import torch

import transformers
if hasattr(transformers, "DynamicCache") and not hasattr(transformers.DynamicCache, "to_legacy_cache"):
    def to_legacy_cache(self):
        return tuple(tuple(layer_cache) for layer_cache in self.key_value_states)
    transformers.DynamicCache.to_legacy_cache = to_legacy_cache

import os
import time
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, List, Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to sys.path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class Any4Agent:
    """
    Any4: Aggressive 4-bit quantization (Passive Baseline).
    This baseline uses standard NF4 quantization and acts as the 'no safety' control.
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
        """Load 4-bit VLM model into memory."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Check if model_id is a short name or path
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
             bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "model.embed_tokens", "embed_tokens_extend"],
            )

        print(f"[Any4] Loading VLM from {model_path}...")
        if "phi" in model_path.lower():
            from transformers import AutoConfig
            # Load config first and force eager attention BEFORE model init.
            # Transformers 5.x tries to auto-select FA2 in the model's __init__
            # before our attn_implementation kwarg reaches it, so we must patch
            # the config directly to prevent the ValueError.
            phi_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            phi_config._attn_implementation = "eager"
            phi_config.use_cache = False
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=phi_config,
                quantization_config=bnb_cfg,
                device_map={"": self.device},
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            # Stability fixes from student agent
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.use_cache = False
            if hasattr(self.model, "tie_weights"):
                try: self.model.tie_weights()
                except: pass
            if hasattr(self.model, "lm_head") and hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                try: self.model.lm_head.weight = self.model.model.embed_tokens.weight
                except: pass
            if hasattr(self.model, "model") and not hasattr(self.model.model, "prepare_inputs_for_generation"):
                if hasattr(self.model, "prepare_inputs_for_generation"):
                    self.model.model.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map={"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
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
        print("[Any4] VRAM cleared.")

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        """Run pure 4-bit inference without any interventions."""
        # [A*-Tier Fix] Clear cache before processing
        torch.cuda.empty_cache()
        
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
                ).to(self.device)
        else:
            # Text-only fallback
            model_inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.device)

        input_ids = model_inputs["input_ids"]
        initial_len = input_ids.shape[1]
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []

        if "phi" in self.model_id.lower():
            # Use manual forward loop for Phi-4 due to .generate() NF4 bug
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache()
            attention_mask = model_inputs["attention_mask"]
            # Phi-4's processor puts input_mode in model_inputs (None for text-only).
            # We must pop it from model_inputs to avoid 'multiple values' conflict,
            # then pass the correct value: InputMode.LANGUAGE = 0.
            phi4_input_mode = model_inputs.pop("input_mode", 0)  # 0 = LANGUAGE
            if phi4_input_mode is None:
                phi4_input_mode = 0  # force LANGUAGE mode for text-only

            with torch.no_grad():
                for step in range(max_tokens):
                    if step > 0:
                        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
                    
                    if step == 0:
                        outputs = self.model(
                            **model_inputs,
                            input_mode=phi4_input_mode,
                            past_key_values=past_key_values,
                            use_cache=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    else:
                        outputs = self.model(
                            input_ids=next_token_id,
                            input_mode=phi4_input_mode,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1).to(self.device)
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
                    
                    if input_ids[0, -1].item() in (
                        self.processor.tokenizer.eos_token_id,
                        self.processor.tokenizer.pad_token_id or -1,
                    ):
                        break
            
            gen_ids = input_ids[0, initial_len:]
            generated_text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
            return generated_text, False
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
                )
            
            # Decode only the NEW tokens
            gen_ids = outputs[0][initial_len:]
            generated_text = self.processor.decode(gen_ids, skip_special_tokens=True)
            return generated_text, False # No chasm detection in Any4
