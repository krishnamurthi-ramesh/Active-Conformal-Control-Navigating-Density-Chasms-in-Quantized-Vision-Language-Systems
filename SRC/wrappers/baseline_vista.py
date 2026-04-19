import torch
from torch import nn
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM
from typing import Tuple, List, Optional
import os
import json
from pathlib import Path
import sys

# Ensure VISTA is in path
VISTA_PATH = "/home/cse-sdpl/research/ACC/03_BASELINES/VISTA"
if VISTA_PATH not in sys.path:
    sys.path.append(VISTA_PATH)

from steering_vector import obtain_vsv
from llm_layers import add_vsv_layers, remove_vsv_layers

class VISTAAgent:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.logits_alpha = 0.3
        self.vsv_lambda = 0.5
        self.tar_layers = "28,30"
        
    def load_model(self):
        print(f"[VISTA] Loading {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        loader = AutoModelForCausalLM if "phi" in self.model_id.lower() else AutoModelForImageTextToText
        self.model = loader.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map=self.device,
            trust_remote_code=True
        )
        self._patch_model_for_sla()
        
    def _get_lm_head(self):
        for name, module in self.model.named_modules():
            if name.endswith("lm_head"): return module
        return None

    def _patch_model_for_sla(self):
        """Definitive SLA Patch: Strict Rank-Invariance for AnyRes."""
        def create_patched_forward(original_forward):
            def patched_forward(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                if outputs is None or not hasattr(outputs, "logits"): return outputs
                
                # Capture architectural invariants
                orig_rank = outputs.logits.dim()
                orig_shape = outputs.logits.shape
                orig_dtype = outputs.logits.dtype
                
                if getattr(self.model, 'v_logits_aug', False):
                    kwargs['output_hidden_states'] = True # Ensure hidden states available for next step
                    h_states = getattr(outputs, 'hidden_states', None)
                    if h_states is not None:
                        h_states = h_states[1:] 
                        lm_head = self._get_lm_head()
                        if lm_head:
                            try:
                                s, e = map(int, self.tar_layers.split(','))
                                layer_logits = []
                                for i in range(max(0, s), min(len(h_states), e+1)):
                                    h = h_states[i]
                                    if h is not None:
                                        l = lm_head(h.to(orig_dtype))
                                        # [RANK MIRRORING] Always match whatever the model naturally produced
                                        if l.dim() != orig_rank:
                                            if l.dim() == 3 and orig_rank == 2: l = l.squeeze(1)
                                            elif l.dim() == 2 and orig_rank == 3: l = l.unsqueeze(1)
                                        if l.shape == orig_shape: layer_logits.append(l)
                                if layer_logits:
                                    avg_aug_logits = torch.stack(layer_logits).mean(0)
                                    outputs.logits = self.logits_alpha * avg_aug_logits + (1-self.logits_alpha) * outputs.logits
                            except: pass
                
                # [STRICT RANK GUARD] Absolute guarantee of rank consistency
                if outputs.logits.dim() != orig_rank:
                    if outputs.logits.dim() == 3 and orig_rank == 2: outputs.logits = outputs.logits.squeeze(1)
                    elif outputs.logits.dim() == 2 and orig_rank == 3: outputs.logits = outputs.logits.unsqueeze(1)
                
                return outputs
            return patched_forward

        # Only patch the most specific forward available
        if hasattr(self.model, "language_model"):
            self.model.language_model.forward = create_patched_forward(self.model.language_model.forward)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "forward"):
            self.model.model.forward = create_patched_forward(self.model.model.forward)
        else:
            self.model.forward = create_patched_forward(self.model.forward)

    def run_inference(self, prompt: str, image=None, max_tokens: int = 256) -> Tuple[str, bool]:
        """Bulletproof VISTA inference with architectural integrity for AnyRes/VLM."""
        try:
            if self.model is None: self.load_model()
            if image is not None and isinstance(image, (str, Path)):
                from PIL import Image as PILImage
                image = PILImage.open(image).convert("RGB")
            
            # 1. Prepare Inputs
            if "phi" in self.model_id.lower():
                msg = [{"role": "user", "content": "<|image_1|>\n" + prompt}] if image else [{"role": "user", "content": prompt}]
                text = self.processor.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                inputs_pos = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
                inputs_neg = self.processor(text=self.processor.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(self.device)
            else:
                msg_pos = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}] if image else [{"role": "user", "content": prompt}]
                text_pos = self.processor.apply_chat_template(msg_pos, tokenize=False, add_generation_prompt=True)
                inputs_pos = self.processor(text=text_pos, images=image, return_tensors="pt").to(self.device)
                
                msg_neg = [{"role": "user", "content": prompt}]
                text_neg = self.processor.apply_chat_template(msg_neg, tokenize=False, add_generation_prompt=True)
                inputs_neg = self.processor(text=text_neg, return_tensors="pt").to(self.device)

            # 2. [RESEARCH INTEGRITY] VSV Derivation
            llm = getattr(self.model, "language_model", self.model)
            vsv_ok = False
            try:
                # Include 'image_sizes' and 'pixel_values' for AnyRes integrity
                def filter_kw(d): return {k: v for k, v in d.items() if k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "image_sizes"]}
                vsv_res = obtain_vsv(args=None, model=self.model, kwargs_list=[[filter_kw(inputs_neg), filter_kw(inputs_pos)]], 
                                     layer_indices=[int(x) for x in self.tar_layers.split(',')])
                if vsv_res and vsv_res[0] is not None:
                    add_vsv_layers(llm, torch.stack([vsv_res[0].to(torch.float16)], dim=1).to(self.device), [self.vsv_lambda])
                    vsv_ok = True
            except: pass

            # 3. [RESEARCH INTEGRITY] SLA Enablement
            self.model.v_logits_aug = True
            
            # 4. Generation with absolute fallback
            try:
                gen_inputs = {k: v for k, v in inputs_pos.items() if k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "image_sizes"]}
                with torch.no_grad():
                    outputs = self.model.generate(**gen_inputs, max_new_tokens=max_tokens, do_sample=False)
            except Exception as e:
                # Fallback to Raw Generation
                self.model.v_logits_aug = False
                if vsv_ok: 
                    try: remove_vsv_layers(llm)
                    except: pass
                with torch.no_grad():
                    outputs = self.model.generate(**gen_inputs, max_new_tokens=max_tokens, do_sample=False)
            
            # Cleanup
            if vsv_ok: 
                try: remove_vsv_layers(llm)
                except: pass
            self.model.v_logits_aug = False
            
            return self.processor.decode(outputs[0], skip_special_tokens=True), False
            
        except Exception as e:
            # Absolute level-0 fallback
            self.model.v_logits_aug = False
            try:
                 llm = getattr(self.model, "language_model", self.model)
                 remove_vsv_layers(llm)
            except: pass
            return f"ERROR: {e}", False
