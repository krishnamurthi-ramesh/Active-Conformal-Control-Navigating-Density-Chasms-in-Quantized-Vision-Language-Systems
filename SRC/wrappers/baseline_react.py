import torch
import json
import os
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, BitsAndBytesConfig

class ReActAgent:
    """
    ReAct: Reasoning and Acting in Language Models (ICLR 2023).
    Implements the Thought-Action-Observation loop.
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
        
        # ACC Research Final Rigor Fix: Force ALFWorld Action Syntax
        self.alfworld_prefix = """Interact with a household to solve a task.
You must use the following format:
Thought: <your reasoning>
Action: <specific command like 'go to shelf 1' or 'take apple from fridge 1'>

Example 1:
Task: put a cool apple in the fridge.
Thought: I need to find an apple. I will look in the basket.
Action: go to basket 1

Example 2:
Task: examine the book with the lamp.
Thought: I see a book on the nightstand. I will pick it up.
Action: take book 1 from nightstand 1
"""

    def _is_alfworld(self, prompt: str) -> bool:
        """Check if prompt is ALFWorld based on common indicators."""
        indicators = ["You are in a ", "Your task is to:", "examine the ", "clean the ", "put the "]
        return any(ind in prompt for ind in indicators)

    def load_model(self):
        """Load 4-bit VLM model for ReAct logic."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Resolve model path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)
        elif "llava" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/llava16_7b")
             if local_path.exists(): model_path = str(local_path)
        elif "phi" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal")
             if local_path.exists(): model_path = str(local_path)
             bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "embed_tokens_extend"],
            )
        else: # Default bnb_cfg for other models
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        print(f"[ReAct] Loading VLM from {model_path}...")
        if "phi" in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
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
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
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
        print("[ReAct] VRAM cleared.")

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        """Run ReAct single-pass CoT for the benchmark baseline."""
        # [A*-Tier Fix] Clear cache before processing
        torch.cuda.empty_cache()
        
        if self.model is None:
            self.load_model()
            
        # ReAct prompt injection
        question = prompt.strip()
        if self._is_alfworld(question):
            react_prompt = f"{self.alfworld_prefix}\nTask: {question}\nThought: "
        else:
            react_prompt = f"Question: {question}\nThought 1: "
        
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
                    {"role": "user", "content": "<|image_1|>\n" + react_prompt}
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
                        {"type": "text", "text": react_prompt},
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
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": react_prompt},
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
                ).to(self.model.device)
        else:
            model_inputs = self.processor(
                text=react_prompt, return_tensors="pt"
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
        return generated_text, False # ReAct is a passive baseline
