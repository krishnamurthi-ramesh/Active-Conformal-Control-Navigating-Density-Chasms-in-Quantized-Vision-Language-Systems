"""
ACC VLM Wrapper: Qwen2.5-VL-3B-Instruct (Edge Tier)
=====================================================
Role in fleet:
  - High-efficiency edge agent for fast multimodal inference.
  - Parameters: ~3.1 Billion.
  - 4-bit AWQ quantization (GEMM)
  - Processes both image tokens (ViT encoder) and text tokens (LLM trunk)
  - Hidden state for ppDRE: concatenate [vision_last_layer_mean, llm_last_hidden]

VRAM budget: ~2.0 GB (4-bit AWQ) + 2.5 GB (ppDRE sensor) = ~4.5 GB
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from transformers.cache_utils import DynamicCache

# -----------------------------------------------------------------------------
# ACC Core (reused unchanged from ACC Research)
# -----------------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]  # 02_SRC/
sys.path.insert(0, str(SRC_ROOT))
from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate
from acc_core.system.oracle_bridge import OracleBridge
from wrappers.campaign_logger import CampaignLogger

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_DIR   = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
HIDDEN_DIM  = 128   # Projected state dim for ppDRE
RFF_DIM     = 512   # Random Fourier Features dimension
EPSILON     = 0.05  # 95% coverage guarantee
# Architecture-specific conformal threshold (calibrated, see paper S4)
# [A*-Tier Fix] Normalized E_rel threshold
LAMBDA_STAR = 0.0036  


class QwenVLStudentAgent:
    """
    Qwen2.5-VL-3B ACC Student Agent.

    Wraps the Qwen2.5-VL model with:
    1. 4-bit AWQ quantization (High-efficiency GEMM)
    2. i-ppDRE drift sensor on the multimodal hidden state
    3. Bayesian Conformal Safety Gate (lambda* = 0.075)
    4. Oracle Bridge to Llama-3.2-Vision-11B teacher on CPU
    """

    def __init__(
        self,
        use_teacher: bool = False,
        lambda_star: float = LAMBDA_STAR,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lambda_star = lambda_star

        print(f"[QwenVL-3B] Loading 4-bit model from {LOCAL_DIR}")
        self._load_model()
        self._init_acc_core()
        self.oracle: Optional[OracleBridge] = None
        if use_teacher:
            self._init_oracle()

        # Manifold Anchoring: Set teacher_mean_phi using a neutral prompt
        self.anchor_manifold()

        self.total_tokens = 0
        self.total_interventions = 0
        self.logger = CampaignLogger(model_id=MODEL_ID)
        self.drift_log: List[Dict] = []

    def unload_model(self):
        """Release VRAM and system RAM."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None
        if self.oracle is not None:
            self.oracle.unload_teacher()
            del self.oracle
            self.oracle = None
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("[QwenVL-3B] Model and Oracle unloaded successfully.")

    # -- Model Loading --------------------------------------------------------

    def _load_model(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from awq import AutoAWQForCausalLM
        
        # Check for AWQ directory specifically
        awq_dir = LOCAL_DIR.parent / (LOCAL_DIR.name + "_awq")
        model_path = str(awq_dir) if awq_dir.exists() else (str(LOCAL_DIR) if LOCAL_DIR.exists() else MODEL_ID)
        
        print(f"[QwenVL-3B] Loading model from {model_path}")
        
        try:
            # Try loading as AWQ if bits indicate it (AutoAWQ handles detection)
            self.model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                device_map={"": self.device},
                trust_remote_code=True,
                safetensors=True,
            )
        except Exception as e:
            print(f"[QwenVL-3B] AWQ load failed or not an AWQ model ({e}), trying BitsAndBytes NF4 fallback...")
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map={"": self.device},
                trust_remote_code=True,
                output_hidden_states=True,
                torch_dtype=torch.float16,
            )
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.llm_hidden_dim = self.model.config.text_config.hidden_size  # e.g. 2048
        
        # campaign Hook: Capture ViT query states to detect visual sycophancy
        self.captured_vit_q = None
        def hook_fn(module, input, output):
            # output: [N_patches, hidden_dim] or [Batch, Seq, 3*Dim] for qkv
            # If qkv, we take the first 1/3 (the Query part)
            if output.shape[-1] % 3 == 0 and output.shape[-1] > 2048: # Heuristic for qkv
                dim = output.shape[-1] // 3
                self.captured_vit_q = output[..., :dim].detach().float().cpu()
            else:
                self.captured_vit_q = output.detach().float().cpu()

        # Robust Target Search: Find the last q_proj or qkv in the vision encoder
        hook_registered = False
        self.vit_hidden_dim = 1536 # Fallback
        try:
            target = None
            target_name = ""
            for name, module in self.model.named_modules():
                is_vision = "visual" in name or "vision" in name
                is_q_proj = "q_proj" in name or "qkv" in name
                if is_vision and is_q_proj:
                    target = module
                    target_name = name
            
            if target is not None:
                target.register_forward_hook(hook_fn)
                hook_registered = True
                # Set vit_hidden_dim based on the part we slice in hook_fn
                if hasattr(target, "out_features"):
                    if "qkv" in target_name:
                        self.vit_hidden_dim = target.out_features // 3
                    else:
                        self.vit_hidden_dim = target.out_features
                print(f"[QwenVL-3B] Registered hook on vision module: {target_name} | vit_dim={self.vit_hidden_dim}")
        except Exception as e:
            print(f"[QwenVL-3B] [ALERT] Hook registration failed: {e}")
        
        if not hook_registered:
             print("[QwenVL-3B] [ALERT] Could not find vision q_proj for hook.")
        
        print(f"[QwenVL-3B] Loaded. LLM hidden_dim={self.llm_hidden_dim}")

    # -- ACC Core Initialization ----------------------------------------------

    def _init_acc_core(self):
        # Multimodal Fusion: LLM (2048) + ViT_Q (1536) = 3584
        self.fusion_dim = self.llm_hidden_dim + self.vit_hidden_dim
        self.projection = torch.nn.Linear(
            self.fusion_dim, HIDDEN_DIM, bias=False
        ).cpu()
        self.projection.requires_grad_(False)

        self.drift_tracker = IncrementalDriftTracker(
            input_dim=HIDDEN_DIM,
            rff_dim=RFF_DIM,
            alpha_lambda=0.99,
            device="cpu",
        )
        self.safety_gate = ConformalSafetyGate(
            epsilon=EPSILON,
            min_threshold=self.lambda_star,
        )
        # Pre-set calibrated threshold (no live calibration needed for campaign)
        self.safety_gate.lambda_star = self.lambda_star
        self.safety_gate.is_calibrated = True
        print(f"[QwenVL-3B] ACC gate: lambda*={self.lambda_star:.4f}, fusion_dim={self.fusion_dim}")

    def anchor_manifold(self, anchor_tokens: int = 5):
        """Set the teacher_mean_phi baseline using a neutral prompt."""
        print(f"[QwenVL-3B] Anchoring manifold (N={anchor_tokens} tokens)")
        neutral_prompt = "The image shows"
        inputs = self.processor(text=neutral_prompt, return_tensors="pt").to(self.device)
        
        past_key_values = DynamicCache()
        attention_mask = inputs["attention_mask"]
        
        z_anchors = []
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                past_key_values=past_key_values, 
                use_cache=True, 
                output_hidden_states=True
            )
            z_t = self._extract_multimodal_state(outputs)
            z_anchors.append(z_t)
            
            # Generate a few more tokens to stabilize
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            for _ in range(anchor_tokens - 1):
                # Update attention mask
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
                
                outputs = self.model(
                    input_ids=next_token_id, 
                    past_key_values=past_key_values, 
                    attention_mask=attention_mask,
                    use_cache=True, 
                    output_hidden_states=True
                )
                z_t = self._extract_multimodal_state(outputs)
                z_anchors.append(z_t)
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

        z_stack = torch.stack(z_anchors)
        self.drift_tracker.update_teacher_baseline(z_stack)
        print("[QwenVL-3B] Manifold anchored successfully.")

    def _init_oracle(self):
        teacher_dir = Path(
            "/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_vision_11b"
        )
        model_path = str(teacher_dir) if teacher_dir.exists() else \
                     "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.oracle = OracleBridge(model_id=model_path, device="cpu")
        print(f"[QwenVL-3B] Oracle Bridge configured -> {model_path}")

    # -- Drift Extraction -----------------------------------------------------

    def _extract_multimodal_state(self, outputs) -> torch.Tensor:
        """Build the combined multimodal drift vector for ppDRE."""
        # 1. LLM last hidden state
        llm_last = outputs.hidden_states[-1][:, -1, :].to(torch.float32).cpu()  # [Batch, Dim]

        # 2. Vision Feature (from hook)
        if self.captured_vit_q is not None:
             # captured_vit_q shape: [Batch, Seq, Dim] or [Seq, Dim]
             feat = self.captured_vit_q.to(torch.float32)
             if feat.ndim == 3:
                 vit_q_mean = feat.mean(dim=1)  # [Batch, Dim]
             elif feat.ndim == 2:
                 # If it's already reshaped/concatenated, assume it's [N, Dim] and we only have Batch=1
                 vit_q_mean = feat.mean(dim=0, keepdim=True) # [1, Dim]
             else:
                 vit_q_mean = torch.zeros((llm_last.size(0), self.vit_hidden_dim))
        else:
            vit_q_mean = torch.zeros((llm_last.size(0), self.vit_hidden_dim))

        # Ensure Batch dimension matches (for safety, though usually 1)
        if vit_q_mean.size(0) != llm_last.size(0):
             vit_q_mean = vit_q_mean.expand(llm_last.size(0), -1)

        # 3. Fusion
        combined = torch.cat([vit_q_mean, llm_last], dim=-1)  # [Batch, TotalDim]
        z_t = self.projection(combined)
        # [A*-Tier Fix] Unit-normalize z_t to ensure drift score is relative (E_rel)
        z_t = z_t / (torch.norm(z_t) + 1e-9)
        return z_t.squeeze(0)  # [HIDDEN_DIM]

    # -- Inference Loop (Text-Only Prompt) -----------------------------------

    def run_text(
        self,
        prompt: str,
        image=None,
        max_new_tokens: int = 256,
        task_id: str = "unknown",
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> Dict[str, Any]:
        """Run ACC-guarded multimodal inference."""
        # [A*-Tier Fix] Clear cache before processing to maximize available VRAM for high-res images
        torch.cuda.empty_cache()
        
        # Reset leaky integrator and hook state between samples
        if hasattr(self, 'safety_gate') and hasattr(self.safety_gate, 'reset_integrator'):
            self.safety_gate.reset_integrator()
        self.captured_vit_q = None

        if image is not None:
            if isinstance(image, Path):
                image = str(image)
            # Qwen2.5-VL formatting with resolution constraint
            # max_pixels = 1280 * 28 * 28 = 1,003,520 (approx 1280 tokens)
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
            # Text-only fallback when image unavailable)
            model_inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.device)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        initial_prompt_len = input_ids.shape[-1]
        interventions = 0
        drift_scores: List[float] = []
        chasm_detected = False
        handoff_idx = -1
        t_start = time.time()

        # Tracking for timing splits
        total_vlm_ms = 0
        total_acc_ms = 0
        prev_w_x = 0.0

        past_key_values = DynamicCache()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Update attention mask for incremental steps
                if step > 0:
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)

                # 1. VLM Forward Pass (System 1)
                vlm_t0 = time.time()
                # Prefill (step 0) uses full inputs, subsequent steps use KV-cache
                if step == 0:
                    outputs = self.model(
                        **model_inputs,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    outputs = self.model(
                        input_ids=next_token_id,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                vlm_ms = (time.time() - vlm_t0) * 1000
                total_vlm_ms += vlm_ms

                # 2. ACC Logic (Overhead)
                acc_t0 = time.time()
                # ppDRE drift score on multimodal state
                z_t  = self._extract_multimodal_state(outputs)
                w_x  = self.drift_tracker.score(z_t)
                
                # Drift Velocity calculation
                velocity = float(w_x) - prev_w_x
                prev_w_x = float(w_x)
                
                drift_scores.append(float(w_x))

                # Clear hook for next step
                self.captured_vit_q = None

                # Logit-based prediction-set size (BQ proxy)
                logits         = outputs.logits[:, -1, :]
                probs          = torch.softmax(logits, dim=-1)
                pred_set_size  = int((probs > 0.01).sum().item())

                gate_triggered = self.safety_gate.check(w_x, pred_set_size=pred_set_size)
                acc_ms = (time.time() - acc_t0) * 1000
                total_acc_ms += acc_ms

                # Record per-token data
                self.logger.log_step(
                    task_id, step, float(w_x), velocity, 
                    vlm_ms, acc_ms, self.lambda_star, gate_triggered
                )

                if gate_triggered:
                    chasm_detected = True
                    handoff_idx = step if handoff_idx == -1 else handoff_idx
                    self.logger.log_handoff(task_id, step, float(w_x), "Llama-3.2-11B")
                    
                    interventions += 1
                    self.total_interventions += 1
                    
                    # Oracle correction: Efficiently offload the REST of the generation
                    if self.oracle is not None:
                        # 1. Get current text prompt (student prefix only - skip image tokens)
                        student_partial = self.processor.tokenizer.decode(
                            input_ids[0, initial_prompt_len:], skip_special_tokens=True
                        )
                        teacher_prompt = f"{prompt} {student_partial}".strip()
                        
                        # 2. Call teacher for the REMAINING tokens
                        remaining_tokens = max_new_tokens - step
                        teacher_text = self.oracle.generate(
                            teacher_prompt, 
                            image=image, 
                            max_new_tokens=remaining_tokens
                        )
                        
                        # 3. Concatenate and finalize
                        teacher_ids = self.processor.tokenizer.encode(teacher_text, return_tensors="pt").to(self.device)
                        input_ids = torch.cat([input_ids, teacher_ids], dim=1)
                        
                        # Mark interventions as the number of tokens the teacher generated
                        interventions = teacher_ids.shape[-1]
                        break # Finished the sequence via oracle
                    else:
                        # Greedy student fallback (no oracle)
                        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1).to(self.device)
                        input_ids = torch.cat([input_ids, next_token_id], dim=1)
                else:
                    self.drift_tracker.update(z_t)
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1).to(self.device)
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)

                self.total_tokens += 1

                if input_ids[0, -1].item() in (
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.pad_token_id,
                ):
                    break

        elapsed = time.time() - t_start
        gen_ids = input_ids[0, initial_prompt_len:]
        generated_text = self.processor.decode(gen_ids, skip_special_tokens=True)
        student_tokens = max_new_tokens - interventions
        efficiency = student_tokens / max(max_new_tokens, 1)

        return {
            "task_id":          task_id,
            "generated_text":   generated_text,
            "interventions":    interventions,
            "total_tokens":     step + 1,
            "efficiency":       round(efficiency, 4),
            "dcdr":             1 if chasm_detected else 0,
            "drift_scores":     drift_scores,
            "handoff_idx":      handoff_idx,
            "vlm_ms":           round(total_vlm_ms, 2),
            "acc_ms":           round(total_acc_ms, 2),
            "elapsed_s":        round(elapsed, 2),
            "quant_state":      "nf4"
        }

    def unload_model(self):
        """Explicitly clear model from VRAM."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if self.oracle:
            self.oracle.unload_teacher()
            del self.oracle
            self.oracle = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("[QwenVL-3B] Model and Oracle unloaded.")


# -----------------------------------------------------------------------------
# Smoke test (run directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = QwenVLStudentAgent(use_teacher=False)
    result = agent.run_text(
        prompt="What is 2 + 2? Explain your reasoning.",
        max_new_tokens=30,
        task_id="smoke_test",
    )
    print("\n=== SMOKE TEST RESULT ===")
    print(f"  Generated: {result['generated_text']}")
    print(f"  Efficiency: {result['efficiency']*100:.1f}%")
    print(f"  DCDR:       {result['dcdr']}")
    print(f"  Elapsed:    {result['elapsed_s']}s")
