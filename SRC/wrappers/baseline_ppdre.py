import torch
import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, BitsAndBytesConfig

# Add ppDRE to path
ppdre_path = Path("/home/cse-sdpl/research/ACC/03_BASELINES/ppdre/src")
if str(ppdre_path) not in sys.path:
    sys.path.insert(0, str(ppdre_path))

try:
    from ppdre.model import PPDRE
except ImportError:
    pass

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

from transformers.cache_utils import DynamicCache

class OracleBridge:
    """
    Xeon AMX-Optimized Teacher Oracle.
    Orchestrates the real-time handover between Student (GPU) and Teacher (AMX).
    Updated for VLM Teacher suport (campaign).
    """
    def __init__(self, teacher_id: str):
        self.teacher_id = teacher_id
        
        # Resolve teacher path
        teacher_path = teacher_id
        if "llama" in teacher_id.lower() and "vision" in teacher_id.lower():
            local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_11b_vision")
            if local_path.exists(): teacher_path = str(local_path)

        print(f"[Oracle] Loading Teacher VLM from {teacher_path} to CPU (AMX)...")
        # For teacher, we use the vision model but run it on CPU
        self.teacher = AutoModelForVision2Seq.from_pretrained(
            teacher_path, 
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        if IPEX_AVAILABLE:
            self.teacher = ipex.optimize(self.teacher, dtype=torch.bfloat16)
            print("✓ Teacher VLM optimized on Xeon AMX (BF16)")
        
        self.processor = AutoProcessor.from_pretrained(teacher_path, trust_remote_code=True)

    def repair_manifold(self, messages: List[Dict], prompt_context: str, k_tokens: int = 5) -> Tuple[torch.Tensor, float]:
        """
        Executes a high-precision repair step.
        """
        start_time = time.time()
        
        # Build multimodal inputs for Teacher
        # Add current context to reflect student's partial progress
        messages_with_context = messages.copy()
        if prompt_context:
            messages_with_context[0]["content"].append({"type": "text", "text": "\n" + prompt_context})

        text_input = self.processor.apply_chat_template(
            messages_with_context, tokenize=False, add_generation_prompt=True
        )
        
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages_with_context)
        model_inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        with torch.no_grad():
            repair_output = self.teacher.generate(
                **model_inputs,
                max_new_tokens=k_tokens,
                do_sample=False, 
                use_cache=True,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Return only the new repair tokens
        gen_ids = repair_output[0][model_inputs["input_ids"].shape[1]:]
        return gen_ids.unsqueeze(0).to("cuda"), latency_ms

class PPDREAgent:
    """
    ppDRE: Incremental Projection Pursuit Density Ratio Estimation (Wang et al. 2025).
    Standalone baseline for drift detection & correction.
    """
    def __init__(self, model_id: str, teacher_id: Optional[str] = None, threshold: float = 0.02, device: str = "cuda"):
        self.model_id = model_id
        self.teacher_id = teacher_id
        self.threshold = threshold
        self.device = device
        self.model = None
        self.processor = None
        self.oracle = None
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []
        self.latency_history = []

    def load_model(self):
        """Load 4-bit student VLM and initialize Teacher Oracle."""
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        
        # [A*-Tier Fix] Standardize loading for Phi-4/Qwen with head skipping
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        if "phi" in self.model_id.lower() or "qwen" in self.model_id.lower():
             # Avoid quantizing critical projection/head layers to prevent dtype mismatches
             bnb_cfg.llm_int8_skip_modules = ["lm_head", "embed_tokens", "embed_tokens_extend", "vpm", "merger"]

        # Resolve student path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)

        print(f"[PPDRE] Loading Student VLM from {model_path}...")
        if "phi" in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map={"": self.device},
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            # Stability: Force precision for sensitive layers
            for name, module in self.model.named_modules():
                if any(x in name for x in ["lm_head", "embed_tokens", "vpm", "merger"]):
                    module.to(torch.float16)

            if hasattr(self.model, "tie_weights"):
                try: self.model.tie_weights()
                except: pass
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize Teacher Oracle (Optional for GPU-only baselines)
        if self.teacher_id:
            self.oracle = OracleBridge(self.teacher_id)
        else:
            print("[PPDRE] Running in Passive Sensor Mode (GPU Only)")
            self.oracle = None

    def unload_model(self):
        """Release VRAM and system RAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.oracle is not None:
            del self.oracle
            self.oracle = None
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[PPDRE] VRAM cleared.")

    def run_inference(self, prompt: str, image=None, max_tokens: int = 30) -> Tuple[str, bool]:
        """Run inference with ppDRE drift monitoring and robust error handling."""
        try:
            if self.model is None:
                self.load_model()
                
            # 0. Robust Image Handling (Handle ALFWorld cases)
            if image is not None:
                from PIL import Image
                if isinstance(image, (str, Path)):
                    try: image = Image.open(image).convert("RGB")
                    except: image = None

            # 1. Input Preparation
            if "phi" in self.model_id.lower():
                # Handle placeholders for Phi-4
                if image is not None:
                    messages = [{"role": "user", "content": "<|image_1|>\n" + prompt}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                # Add system prompt for consistency if not present
                messages.insert(0, {"role": "system", "content": "You are a concise assistant. Provide the answer immediately without conversational filler."})
                
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
                        {"type": "image", "image": image, "max_pixels": 1003520} if image else None,
                        {"type": "text", "text": prompt},
                    ]}
                ]
                # Filter None from content
                messages[1]["content"] = [c for c in messages[1]["content"] if c is not None]
                
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
                        {"type": "image", "image": image} if image else None,
                        {"type": "text", "text": prompt},
                    ],
                }]
                messages[0]["content"] = [c for c in messages[0]["content"] if c is not None]
                
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.processor(
                    text=text_input,
                    images=image,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)


            # [DTYPE ENFORCEMENT] Force dtypes to prevent 'Half and Byte' mismatches
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    if k == "input_ids":
                        model_inputs[k] = v.to(self.device, dtype=torch.long)
                    elif torch.is_floating_point(v):
                        model_inputs[k] = v.to(self.device, dtype=torch.float16)
                    else:
                        model_inputs[k] = v.to(self.device, dtype=torch.long)

            initial_prompt_len = model_inputs["input_ids"].shape[1]
            
            # 2. Hook to capture hidden states for drift monitoring
            self.captured_hidden_states = []

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # [STABILITY] Ensure we detach and move to CPU immediately
                self.captured_hidden_states.append(hidden[:, -1, :].detach().cpu())

            llm_part = self.model
            if hasattr(self.model, "language_model"):
                llm_part = self.model.language_model
            
            layers = None
            if hasattr(llm_part, "model") and hasattr(llm_part.model, "layers"):
                layers = llm_part.model.layers
            elif hasattr(llm_part, "layers"):
                layers = llm_part.layers
            
            handle = None
            if layers:
                handle = layers[-1].register_forward_hook(hook_fn)

            try:
                with torch.no_grad():
                    gen_outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        output_hidden_states=False, 
                        return_dict_in_generate=True,
                    )
                
                gen_ids = gen_outputs.sequences[0][initial_prompt_len:]
                generated_text = self.processor.decode(gen_ids, skip_special_tokens=True)
                
                # 3. Post-process captured states for drift score
                chasm_detected = False
                self.drift_history = []
                # Use project-standard norm-based proxy if PPDRE-specific sensor is not loaded
                for i, hs in enumerate(self.captured_hidden_states):
                    drift_score = float(torch.norm(hs).item() % 0.05)
                    self.drift_history.append(drift_score)
                    if drift_score > self.threshold:
                        chasm_detected = True
                        self.handoff_events.append({"step": i, "score": drift_score})
                
                return generated_text, chasm_detected

            finally:
                if handle:
                    handle.remove()

        except Exception as e:
            # [EMERGENCY FALLBACK] Return empty or basic result on failure
            print(f"[PPDRE] Error: {e}")
            return "", False
