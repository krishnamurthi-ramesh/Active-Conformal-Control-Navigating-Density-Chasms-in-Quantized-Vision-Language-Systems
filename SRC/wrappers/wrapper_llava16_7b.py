"""
ACC VLM Wrapper: LLaVA-v1.6-Vicuna-7B (Standard Tier)
======================================================
Role in fleet:
  - Fleet baseline for cross-architecture validation.
  - Parameters: ~7.0 Billion.
  - 4-bit NF4 quantization (via BitsAndBytes)
  - Architecture: Separate CLIP ViT vision encoder + Vicuna-7B LLM
  - ViT self-attention distortion is visible here - key finding
  - Hidden state for ppDRE: LLM last-layer + optional ViT last-layer mean

VRAM budget: ~4.5 GB (NF4 7B) + 2.5 GB (ppDRE) = ~7 GB
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
from transformers.cache_utils import DynamicCache

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))
from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate
from acc_core.system.oracle_bridge import OracleBridge
from wrappers.campaign_logger import CampaignLogger

MODEL_ID    = "llava-hf/llava-v1.6-vicuna-7b-hf"
LOCAL_DIR   = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/llava16_7b")
HIDDEN_DIM  = 128
RFF_DIM     = 512
EPSILON     = 0.05  # Architecture-specific conformal threshold (calibrated, see paper S4)
# [A*-Tier Fix] Normalized E_rel threshold (Sensitive Vision-Projection)
LAMBDA_STAR = 0.0019
# LLaVA-7B: general-purpose, sharp manifold boundary


class LLaVA16StudentAgent:
    """
    LLaVA-v1.6-Vicuna-7B ACC Student Agent (Standard Tier).

    Key insight: LLaVA has a CLIP ViT vision encoder whose
    self-attention maps show significant distribution shift under 4-bit
    quantization. This is the primary source of "visual sycophancy" -
    the model ignores contradictory visual evidence because ViT attention
    maps collapse to spurious patterns.

    ACC monitors latent drift from BOTH the ViT encoder and LLM trunk.
    """

    def __init__(
        self,
        use_teacher: bool = False,
        lambda_star: float = LAMBDA_STAR,
        use_vit_drift: bool = True,     # ViT-aware drift
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lambda_star = lambda_star
        self.use_vit_drift = use_vit_drift

        print(f"[LLaVA-1.6-7B] Loading NF4 model from {LOCAL_DIR}")
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("[LLaVA-1.6-7B] Model and Oracle unloaded successfully.")

    # -- Model Loading --------------------------------------------------------

    def _load_model(self):
        from transformers import (
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
            BitsAndBytesConfig,
        )
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True, # Allow offloading
        )
        model_path = str(LOCAL_DIR) if LOCAL_DIR.exists() else MODEL_ID
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.llm_hidden_dim  = self.model.config.text_config.hidden_size  # 4096
        self.vit_hidden_dim  = self.model.config.vision_config.hidden_size  # 1024
        print(f"[LLaVA-1.6-7B] LLM hidden={self.llm_hidden_dim}, ViT hidden={self.vit_hidden_dim}")

        # campaign Hook: Capture ViT query states (visual sycophancy signal)
        self.captured_vit_q = None
        def hook_fn(module, input, output):
            self.captured_vit_q = output.detach().float().cpu()

        # Target: CLIP ViT encoder last layer query projection
        try:
            self.model.model.vision_tower.vision_model.encoder.layers[-1].self_attn.q_proj.register_forward_hook(hook_fn)
            print("[LLaVA-1.6-7B] Registered hook on vision_tower...layers[-1].self_attn.q_proj")
        except Exception as e:
            print(f"[LLaVA-1.6-7B] [ALERT] Hook registration failed: {e}")

    # -- ACC Core -------------------------------------------------------------

    def _init_acc_core(self):
        # Multimodal Fusion: LLM (4096) + ViT_Q (1024) = 5120
        self.fusion_dim = self.llm_hidden_dim + self.vit_hidden_dim
        self.projection = torch.nn.Linear(self.fusion_dim, HIDDEN_DIM, bias=False).cpu()
        self.projection.requires_grad_(False)

        self.drift_tracker = IncrementalDriftTracker(
            input_dim=HIDDEN_DIM, rff_dim=RFF_DIM, alpha_lambda=0.99, device="cpu"
        )
        self.safety_gate = ConformalSafetyGate(
            epsilon=EPSILON, min_threshold=self.lambda_star
        )
        self.safety_gate.lambda_star = self.lambda_star
        self.safety_gate.is_calibrated = True
        print(f"[LLaVA-1.6-7B] ACC gate: lambda*={self.lambda_star:.4f}, fusion_dim={self.fusion_dim}")

    def anchor_manifold(self, anchor_tokens: int = 5):
        """Set the teacher_mean_phi baseline using a neutral prompt."""
        print(f"[LLaVA-1.6-7B] Anchoring manifold (N={anchor_tokens} tokens)")
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
        print("[LLaVA-1.6-7B] Manifold anchored successfully.")

    def _init_oracle(self):
        teacher_dir = Path(
            "/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_vision_11b"
        )
        model_path = str(teacher_dir) if teacher_dir.exists() else \
                     "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.oracle = OracleBridge(model_id=model_path, device="cpu")

    def _extract_multimodal_state(self, outputs) -> torch.Tensor:
        """Build the combined multimodal drift vector for ppDRE."""
        # 1. LLM last hidden state
        llm_last = outputs.hidden_states[-1][:, -1, :].to(torch.float32).cpu()  # [Batch, Dim]

        # 2. Vision Feature (from hook)
        if self.captured_vit_q is not None:
             # captured_vit_q shape: [Batch, Seq, Dim] or [Batch*Seq, Dim]
             feat = self.captured_vit_q.to(torch.float32)
             if feat.ndim == 3:
                 vit_q_mean = feat.mean(dim=1)  # [Batch, Dim]
             elif feat.ndim == 2:
                 # If it's already reshaped, assume it's [N, Dim] and we only have Batch=1
                 vit_q_mean = feat.mean(dim=0, keepdim=True) # [1, Dim]
             else:
                 vit_q_mean = torch.zeros((llm_last.size(0), self.vit_hidden_dim))
        else:
            vit_q_mean = torch.zeros((llm_last.size(0), self.vit_hidden_dim))

        # Ensure Batch dimension matches (for safety, though usually 1)
        # LLaVA-1.6 AnyRes can produce multiple "batches" if crops are treated as batch
        if vit_q_mean.size(0) != llm_last.size(0):
             if vit_q_mean.size(0) > 1 and llm_last.size(0) == 1:
                 # Pool multiple crops (AnyRes) into a single feature vector
                 vit_q_mean = vit_q_mean.mean(dim=0, keepdim=True)
             else:
                 vit_q_mean = vit_q_mean.expand(llm_last.size(0), -1)

        # 3. Fusion
        combined = torch.cat([vit_q_mean, llm_last], dim=-1)  # [Batch, TotalDim]
        z_t = self.projection(combined)
        # [A*-Tier Fix] Unit-normalize z_t to ensure drift score is relative (E_rel)
        z_t = z_t / (torch.norm(z_t) + 1e-9)
        return z_t.squeeze(0)  # [HIDDEN_DIM]

    # -- Inference Loop -------------------------------------------------------

    def run_text(
        self,
        prompt: str,
        image=None,
        max_new_tokens: int = 256,
        task_id: str = "unknown",
    ) -> Dict[str, Any]:
        """Run ACC-guarded multimodal inference on a single VLM sample."""
        # [A*-Tier Fix] Clear cache before processing to maximize available VRAM
        torch.cuda.empty_cache()
        
        # Reset hook state and leaky integrator between samples
        self.captured_vit_q = None
        if hasattr(self, 'safety_gate') and hasattr(self.safety_gate, 'reset_integrator'):
            self.safety_gate.reset_integrator()

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a concise assistant. Provide the answer immediately without conversational filler."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        if image is not None:
             conversation[0]["content"].append({"type": "image"})
             if isinstance(image, (str, Path)):
                 image = Image.open(image).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        model_inputs = self.processor(
            text=text_prompt, images=image, return_tensors="pt"
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
                # Multimodal drift score
                z_t  = self._extract_multimodal_state(outputs)
                w_x  = self.drift_tracker.score(z_t)
                
                # Drift Velocity calculation
                velocity = float(w_x) - prev_w_x
                prev_w_x = float(w_x)
                
                drift_scores.append(float(w_x))

                # Clear hook
                self.captured_vit_q = None

                logits = outputs.logits[:, -1, :]
                probs  = torch.softmax(logits, dim=-1)
                pred_set_size = int((probs > 0.01).sum().item())
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
                        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
                        input_ids  = torch.cat([input_ids, next_token_id], dim=1)
                else:
                    self.drift_tracker.update(z_t)
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1).to(self.device)
                    input_ids  = torch.cat([input_ids, next_token_id], dim=1)

                self.total_tokens += 1

                if input_ids[0, -1].item() in (
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.pad_token_id or -1,
                ):
                    break

        elapsed = time.time() - t_start
        gen_ids = input_ids[0, initial_prompt_len:]
        generated_text = self.processor.tokenizer.decode(
            gen_ids, skip_special_tokens=True
        )
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
        print("[LLaVA-1.6-7B] Model and Oracle unloaded.")


if __name__ == "__main__":
    agent = LLaVA16StudentAgent(use_teacher=False, use_vit_drift=True)
    result = agent.run_text(
        "Is there a cat in this image?", max_new_tokens=20, task_id="smoke_test"
    )
    print(f"Generated:  {result['generated_text']}")
    print(f"Efficiency: {result['efficiency']*100:.1f}%  DCDR: {result['dcdr']}")
