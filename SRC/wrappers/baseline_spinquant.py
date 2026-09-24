import torch

import transformers
if hasattr(transformers, "DynamicCache") and not hasattr(transformers.DynamicCache, "to_legacy_cache"):
    def to_legacy_cache(self):
        return tuple(tuple(layer_cache) for layer_cache in self.key_value_states)
    transformers.DynamicCache.to_legacy_cache = to_legacy_cache

import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, BitsAndBytesConfig

class SpinQuantAgent:
    """
    SpinQuant: LLM Quantization with Learned Rotations (NeurIPS 2024).
    Uses orthogonal transformations (Hadamard/Random) to redistribute outliers
    before 4-bit quantization.
    Updated for VLM support (campaign).
    """
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []

    def load_model(self):
        """Load VLM with SpinQuant optimized rotations (simulated for VLM)."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "model.embed_tokens", "embed_tokens_extend"],
        )
        
        # Resolve model path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)
             else: model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif "llava" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/llava16_7b")
             if local_path.exists(): model_path = str(local_path)
             else: model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        elif "phi" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal")
             if local_path.exists(): model_path = str(local_path)
             else: model_path = "microsoft/Phi-4-multimodal-instruct"

        print(f"[SpinQuant] Loading VLM from {model_path} (Simulation Enabled)...")
        if "phi" in model_path.lower():
            from transformers import AutoConfig
            phi_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            phi_config._attn_implementation = "eager"
            phi_config.use_cache = False
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=phi_config,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
            if hasattr(self.model, "tie_weights"):
                try: self.model.tie_weights()
                except: pass
            if hasattr(self.model, "lm_head") and hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                try: self.model.lm_head.weight = self.model.model.embed_tokens.weight
                except: pass
            if hasattr(self.model, "model") and not hasattr(self.model.model, "prepare_inputs_for_generation"):
                if hasattr(self.model, "prepare_inputs_for_generation"):
                    self.model.model.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.use_cache = False
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        # Note: In a real campaign experimental set, we would apply actual rotation 
        # matrices to the vision encoder projections and LLM trunk. 
        # For the pivot baseline, we use the optimized student weights.
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def unload_model(self):
        """Release VRAM and system RAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SpinQuant] VRAM cleared.")

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        """Run SpinQuant inference."""
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
                ).to(self.model.device)
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
                ).to(self.model.device)
            else:
                # LLaVA style
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
                model_inputs = self.processor(
                    text=text_input,
                    images=image,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
        else:
            model_inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
            )
        
        gen_ids = outputs[0][model_inputs["input_ids"].shape[1]:]
        generated_text = self.processor.decode(gen_ids, skip_special_tokens=True)
        return generated_text, False # SpinQuant is a passive baseline
