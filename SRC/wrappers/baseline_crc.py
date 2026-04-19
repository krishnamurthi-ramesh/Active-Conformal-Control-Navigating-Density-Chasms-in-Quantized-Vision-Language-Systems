import torch
import numpy as np
import sys
import os
import gc
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

# Add Conformal Risk to path
crc_path = Path("/home/cse-sdpl/research/ACC/BASELINES/conformal-risk")
if str(crc_path) not in sys.path:
    sys.path.insert(0, str(crc_path))

try:
    from core.get_lhat import get_lhat
except ImportError:
    print("Warning: Conformal Risk core not found. Using local fallback implementation.")
    def get_lhat(calib_loss_table, lambdas, alpha, B=1):
        n = calib_loss_table.shape[0]
        rhat = calib_loss_table.mean(axis=0)
        lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0)
        return lambdas[lhat_idx]

class CRCAgent:
    """
    CRC: Conformal Risk Control (Angelopoulos et al., ICLR 2024 Spotlight).
    Updated for VLM support (campaign).
    """
    def __init__(self, model_id: str, alpha: float = 0.1, device: str = "cuda"):
        self.model_id = model_id
        self.alpha = alpha
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.lambda_hat = 1.0  # Threshold
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []

    def load_model(self):
        """Load 4-bit student VLM."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Resolve model path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)
        elif "llava" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/DATA/models/student_vlm/llava16_7b")
             if local_path.exists(): model_path = str(local_path)
        elif "phi" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/DATA/models/student_vlm/phi4_multimodal")
             if local_path.exists(): model_path = str(local_path)
             bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens_extend"],
        )
        
        print(f"[CRC] Loading VLM from {model_path}...")
        if "phi" in model_path.lower():
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map={"": self.device},
                trust_remote_code=True,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
            # Stability fixes from student agent
            if hasattr(self.model, "tie_weights"):
                try: self.model.tie_weights()
                except: pass
            if hasattr(self.model, "model") and not hasattr(self.model.model, "prepare_inputs_for_generation"):
                if hasattr(self.model, "prepare_inputs_for_generation"):
                    self.model.model.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map={"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def unload_model(self):
        """Release VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        """Run CRC inference with multimodal support."""
        if self.model is None:
            self.load_model()

        # Build multimodal inputs
        if image is not None:
            if isinstance(image, Path):
                image = str(image)
            
            if "phi" in self.model_id.lower():
                # Phi-4 format
                from PIL import Image
                if isinstance(image, (str, Path)):
                    image = Image.open(image).convert("RGB")
                
                messages = [
                    {"role": "system", "content": "You are a concise assistant. Provide the answer immediately without conversational filler."},
                    {"role": "user", "content": "<|image_1|>\n" + prompt}
                ]
                text_input = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.processor(
                    text=text_input,
                    images=image,
                    return_tensors="pt",
                ).to(self.device)
            elif "qwen" in self.model_id.lower():
                messages = [
                    {"role": "system", "content": "You are a concise assistant. Provide the answer immediately without conversational filler."},
                    {"role": "user", "content": [
                        {"type": "image", "image": image, "max_pixels": 1003520},
                        {"type": "text", "text": prompt},
                    ]}
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                model_inputs = self.processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            else:
                # LLaVA style
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.processor(
                    text=text_input,
                    images=image,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
        else:
            model_inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.device)

        input_ids = model_inputs["input_ids"]
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []
        chasm_detected = False

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=False, # standard CRC usually uses greedy/nucleus
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
            )
            
            # Post-hoc entropy check (simplified CRC score)
            first_pass = self.model(**model_inputs)
            logits = first_pass.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
            self.drift_history = [entropy]

        if entropy > self.lambda_hat:
            chasm_detected = True

        gen_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.processor.decode(gen_ids, skip_special_tokens=True)
        return generated_text, chasm_detected
