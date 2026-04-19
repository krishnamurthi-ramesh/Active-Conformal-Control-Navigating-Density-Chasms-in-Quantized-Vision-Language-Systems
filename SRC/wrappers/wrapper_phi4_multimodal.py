"""
ACC VLM Wrapper: Phi-4 Multimodal-Instruct (Logic Tier)
=========================================================
Role in fleet:
  - Proving ACC prevents the 4.7% accuracy collapse in high-reasoning tasks.
  - Parameters: 5.6 Billion.
  - Dense latent manifold (lambda* ~ 0.200) - lowest drift sensitivity
  - 4-bit NF4 quantization (with Safe LoRA Patch)
  - Architecture: Phi-4 with interleaved vision + language processing
  - Hidden state for ppDRE: LLM trunk last-layer hidden state

VRAM budget: ~4.5 GB (NF4 4-bit) + 2.5 GB (ppDRE sensor) = ~7 GB
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

MODEL_ID    = "microsoft/Phi-4-multimodal-instruct"
LOCAL_DIR   = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal")
HIDDEN_DIM  = 128
RFF_DIM     = 512
EPSILON     = 0.05
# [A*-Tier Fix] Normalized E_rel threshold
LAMBDA_STAR = 0.0032


class Phi4MultimodalStudentAgent:
    """
    Phi-4 Multimodal ACC Student Agent (Logic Tier).

    Key finding (carried into final campaign):
    Phi-4's dense synthetic pre-training produces a naturally compact latent
    manifold that is highly resilient to 4-bit quantization noise.
    This requires a threshold of lambda*=0.075 to balance safety and efficiency.
    """

    def __init__(
        self,
        use_teacher: bool = False,
        lambda_star: float = LAMBDA_STAR,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lambda_star = lambda_star

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
        print("[Phi4-MM] Model and Oracle unloaded successfully.")

    # Model Loading
    def _load_model(self):
        """
        Load Phi-4 Multimodal in NF4.
"""
        model_name = str(LOCAL_DIR) if LOCAL_DIR.exists() else MODEL_ID

        print(f"[Phi4-MM] Loading model from {model_name} …")
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens_extend"],
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation="sdpa",
            output_hidden_states=True,
        )
        
        # Tie model weights if possible (fixes warning)
        if hasattr(self.model, "tie_weights"):
            try:
                self.model.tie_weights()
                print("[Phi4-MM] Model weights tied successfully.")
            except Exception as e:
                print(f"[Phi4-MM] [WARNING] tie_weights() failed: {e}")

        # Log VRAM and RAM usage after model load
        try:
            import psutil
            vram = None
            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1024**3
            ram = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            print(f"[Phi4-MM] Memory usage after model load: RAM={{ram:.2f}} GB" + (f", VRAM={{vram:.2f}} GB" if vram is not None else ""))
        except Exception as e:
            print(f"[Phi4-MM] [WARNING] Could not log memory usage: {e}")

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.llm_hidden_dim = getattr(self.model.config, "hidden_size", 3072)
        print(f"[Phi4-MM] Loaded. hidden_dim={self.llm_hidden_dim}")
        self.vit_hidden_dim = 1152

        # Fix prepare_inputs_for_generation attribute error if needed
        if hasattr(self.model, "model") and not hasattr(self.model.model, "prepare_inputs_for_generation"):
            if hasattr(self.model, "prepare_inputs_for_generation"):
                 self.model.model.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
                 print("[Phi4-MM] Delegated prepare_inputs_for_generation to model trunk.")

        # Hook: Capture ViT query states
        self.captured_vit_q = None
        def hook_fn(module, input, output):
            self.captured_vit_q = output.detach().float().cpu()

        # Find the last q_proj in the vision encoder for the hook
        hook_registered = False
        try:
            target = None
            for name, module in self.model.named_modules():
                if ("image_embed" in name or "visual" in name) and "q_proj" in name:
                    target = module 
            
            if target is not None:
                target.register_forward_hook(hook_fn)
                hook_registered = True
                print("[Phi4-MM] Registered hook on vision module.")
        except Exception as e:
            print(f"[Phi4-MM] [ALERT] Hook registration failed: {e}")
        
        if not hook_registered:
             print("[Phi4-MM] [ALERT] Could not find vision q_proj for hook.")

    # --- ACC Core -----------------------------------------------------------

    def _init_acc_core(self):
        # Multimodal Fusion: LLM (3072) + ViT_Q (1152) = 4224
        self.fusion_dim = self.llm_hidden_dim + self.vit_hidden_dim
        self.projection = torch.nn.Linear(
            self.fusion_dim, HIDDEN_DIM, bias=False
        ).cpu()
        self.projection.requires_grad_(False)

        self.drift_tracker = IncrementalDriftTracker(
            input_dim=HIDDEN_DIM, rff_dim=RFF_DIM, alpha_lambda=0.99, device="cpu"
        )
        self.safety_gate = ConformalSafetyGate(
            epsilon=EPSILON, min_threshold=self.lambda_star
        )
        self.safety_gate.lambda_star = self.lambda_star
        self.safety_gate.is_calibrated = True
        print(f"[Phi4-MM] ACC gate: lambda*={self.lambda_star:.4f}, fusion_dim={self.fusion_dim}")

    def anchor_manifold(self, anchor_tokens: int = 5):
        """Set the teacher_mean_phi baseline using a neutral prompt."""
        print(f"[Phi4-MM] Anchoring manifold (N={anchor_tokens} tokens)")
        neutral_prompt = "The image shows"
        inputs = self.processor(text=neutral_prompt, return_tensors="pt").to(self.device)
        
        past_key_values = DynamicCache()
        attention_mask = inputs["attention_mask"]
        
        z_anchors = []
        with torch.no_grad():
            # Initial forward pass (prefill)
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
                # Update attention mask for next step
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
        print("[Phi4-MM] Manifold anchored successfully.")

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
             # captured_vit_q shape: [Batch, Seq, Dim] or [Seq, Dim]
             feat = self.captured_vit_q.to(torch.float32)
             if feat.ndim == 3:
                 vit_q_mean = feat.mean(dim=1)  # [Batch, Dim]
             elif feat.ndim == 2:
                 # If it's already reshaped/concatenated, assume it's [N, Dim] and we only have Batch=1
                 vit_q_mean = feat.mean(dim=0, keepdim=True) # [1, Dim]
             else:
                 # Fallback
                 vit_q_mean = torch.zeros((llm_last.size(0), self.vit_hidden_dim))
        else:
            vit_q_mean = torch.zeros((llm_last.size(0), self.vit_hidden_dim))

        # Ensure Batch dimension matches (for safety, though usually 1)
        # Robust pooling for multi-crop or batch mismatches
        if vit_q_mean.size(0) != llm_last.size(0):
             if vit_q_mean.size(0) > 1 and llm_last.size(0) == 1:
                 # AnyRes/Multi-crop pooling
                 vit_q_mean = vit_q_mean.mean(dim=0, keepdim=True)
             else:
                 vit_q_mean = vit_q_mean.expand(llm_last.size(0), -1)

        # 3. Fusion
        combined = torch.cat([vit_q_mean, llm_last], dim=-1)  # [Batch, TotalDim]
        z_t = self.projection(combined)
        # [A*-Tier Fix] Unit-normalize z_t to ensure drift score is relative (E_rel)
        z_t = z_t / (torch.norm(z_t) + 1e-9)
        return z_t.squeeze(0)  # [HIDDEN_DIM]

    # --- Inference Loop -----------------------------------------------------

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
        # [A*-Tier Fix] Clear cache before processing to maximize available VRAM
        torch.cuda.empty_cache()
        
        # Reset hook state and leaky integrator between samples
        self.captured_vit_q = None
        if hasattr(self, 'safety_gate') and hasattr(self.safety_gate, 'reset_integrator'):
            self.safety_gate.reset_integrator()

        # Build prompt with image placeholder if image provided
        content = []
        if image is not None:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            content = "<|image_1|>\n" + prompt
        else:
            content = prompt

        messages = [
            {"role": "system", "content": "You are a concise assistant. Provide the answer immediately without conversational filler."},
            {"role": "user", "content": content}
        ]

        text_input = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.processor(
            text=text_input,
            images=image,
            return_tensors="pt",
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
                try:
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
                except (TypeError, ValueError) as e:
                    # Fallback for models or configurations that might fail with explicit cache args
                    outputs = self.model(**model_inputs, return_dict=True)
                    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                        # Fallback: skip drift scoring this step
                        logits = outputs.logits[:, -1, :]
                        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
                        input_ids = torch.cat([input_ids, next_token_id], dim=1)
                        continue
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
                        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1).to(self.device)
                        input_ids = torch.cat([input_ids, next_token_id], dim=1)
                else:
                    self.drift_tracker.update(z_t)
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1).to(self.device)
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)

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

if __name__ == "__main__":
    agent = Phi4MultimodalStudentAgent(use_teacher=False)
    result = agent.run_text("What is 3 + 4?", max_new_tokens=20, task_id="smoke_test")
    print(f"Generated: {result['generated_text']}")
    print(f"Efficiency: {result['efficiency']*100:.1f}%  DCDR: {result['dcdr']}")
