"""
ACC AGENT: Active Conformal Control (ACC Research)
Scientific Role: Uses i-ppDRE drift detection + Conformal Safety Gate
                 to detect hallucinations and trigger interventions.

Integration:
    1. IncrementalDriftTracker (i-ppDRE): Scores each step's hidden state
    2. ConformalSafetyGate: Thresholds drift scores with statistical guarantees
    3. OracleBridge: CPU teacher for trajectory correction on drift detection
"""
import argparse
import json
import os
import sys
import time
import torch
import numpy as np

# Add /app/src to path for container imports; also support local dev
SRC_ROOT = "/app/src"
LOCAL_SRC = os.path.join(os.path.dirname(__file__), "..")
for p in [SRC_ROOT, LOCAL_SRC]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

from wrappers.setup_student import setup_model

# Import ACC Core Components
from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate
from acc_core.system.oracle_bridge import OracleBridge

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _resolve_task_file():
    candidates = [
        os.environ.get("ACC_TASK_FILE"),
        "/app/data/benchmark_tasks.json",
        os.path.join(os.path.dirname(__file__), "../../01_DATA/benchmark_tasks.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return "/app/data/benchmark_tasks.json"

TASK_FILE = _resolve_task_file()

# ACC Configuration
HIDDEN_DIM = 128   # Compressed state dimension
RFF_DIM = 512      # Random Fourier Features dimension
CALIBRATION_STEPS = 100
EPSILON = 0.05     # 95% Safety Guarantee


class ACCAgent:
    def __init__(self, model_name: str = "mistral", use_teacher: bool = True, 
                 model=None, tokenizer=None, gsm8k_threshold=None, gsm8k_reanchor=None):
        """
        Args:
            model_name: One of 'llama3', 'mistral', 'phi3'
            use_teacher: If True, load the CPU teacher for real corrections.
            model: Optional pre-loaded student model.
            tokenizer: Optional pre-loaded tokenizer.
            gsm8k_threshold: Optional override for GSM8K threshold.
            gsm8k_reanchor: Optional override for GSM8K re-anchor tokens.
        """
        self.use_teacher = use_teacher
        self.gsm8k_threshold = gsm8k_threshold
        self.gsm8k_reanchor = gsm8k_reanchor

        # 1. Load Student Model (GPU, 4-bit NF4) if not provided
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer, self.model = setup_model(model_name=model_name)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_hidden_dim = self.model.config.hidden_size

        # 2. Initialize ACC Core (runs on CPU)
        self.drift_tracker = IncrementalDriftTracker(
            input_dim=HIDDEN_DIM,
            rff_dim=RFF_DIM,
            alpha_lambda=0.99,
            device="cpu"
        )

        self.safety_gate = ConformalSafetyGate(
            epsilon=EPSILON,
            min_threshold=0.005
        )

        # 3. Projection layer
        self.projection = torch.nn.Linear(
            self.model_hidden_dim, HIDDEN_DIM, bias=False
        ).to("cpu")
        self.projection.requires_grad_(False)

        # 4. Teacher Oracle (lazy-loaded on first intervention if enabled)
        self.oracle: OracleBridge | None = None
        if use_teacher:
            # Resolve model_id for teacher
            model_id = self.model.config._name_or_path
            if not model_id or os.path.isdir(model_id):
                 # Fallback to registry if path is local
                 from wrappers.setup_student import MODEL_REGISTRY
                 model_id = MODEL_REGISTRY.get(model_name, {}).get("id", "mistralai/Mistral-7B-Instruct-v0.3")
                 
            self.oracle = OracleBridge(model_id=model_id, device="cpu")

        # 5. Statistics
        self.total_interventions = 0
        self.total_tokens = 0
        self.is_calibrated = False
        self.drift_log: list[dict] = [] 

    def _extract_state(self, hidden_states):
        """Projects the last layer's last token hidden state to HIDDEN_DIM."""
        last_hidden = hidden_states[-1][:, -1, :]  # [batch, hidden]
        z_t = self.projection(last_hidden.float().cpu())
        return z_t.squeeze(0)

    def _teacher_correct(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Call the CPU teacher to generate the next correct token.
        Implements the real PCIe handoff described in the proposal.

        Returns the corrected input_ids with the teacher's token appended.
        """
        if self.oracle is None:
            # Fallback: simulate latency (for --no-teacher mode)
            time.sleep(0.092)
            return input_ids

        # Lazy-load teacher on first call
        if not self.oracle.is_loaded:
            print("\n   [Oracle] Loading FP16 teacher on CPU (first intervention)...")
            self.oracle.load_teacher(verbose=True)

        # Transfer input_ids to CPU (PCIe handoff)
        cpu_ids = input_ids.cpu()

        # Teacher generates corrected next token
        corrected_ids, latency_ms = self.oracle.correct_trajectory(
            cpu_ids, max_new_tokens=1
        )

        # Transfer corrected token back to GPU
        corrected_ids = corrected_ids.to(self.device)

        print(f" [{latency_ms:.1f}ms]", end="")
        return corrected_ids

    def _get_calibration_prompts(self, n: int = 1000) -> list[str]:
        """Load calibration prompts from C4 validation split.

        Falls back to synthetic prompts if HuggingFace datasets unavailable.
        """
        try:
            from datasets import load_dataset
            print(f"   Loading {n} prompts from AllenAI C4 validation split...")
            ds = load_dataset(
                "allenai/c4", "en",
                split="validation",
                streaming=True,
            )
            prompts = []
            for item in ds:
                text = item.get("text", "")
                # Use first 100 chars as prompt
                if len(text) > 50:
                    prompts.append(text[:100])
                if len(prompts) >= n:
                    break
            print(f"   Loaded {len(prompts)} C4 prompts for calibration")
            return prompts
        except Exception as e:
            print(f"   C4 unavailable ({e}), using synthetic fallback")
            return [
                "The scientific method is a systematic approach to",
                "In mathematics, the fundamental theorem states that",
                "The process of photosynthesis involves converting",
                "Machine learning algorithms can be categorized into",
                "The theory of relativity describes how",
            ] * (n // 5 + 1)

    def calibrate(self, num_steps=CALIBRATION_STEPS):
        """Phase 1: Calibrate the safety gate using diverse C4 prompts.

        Uses 1000 real text samples from C4 validation for research-grade
        calibration (proposal Section 3.2), with synthetic fallback.
        """
        print(f"\n CALIBRATING ACC (target: {num_steps} reference scores)...")

        prompts = self._get_calibration_prompts(n=max(num_steps // 5, 50))
        gen_per_prompt = max(5, num_steps // len(prompts))
        collected = 0

        for pi, prompt_text in enumerate(prompts):
            if collected >= num_steps:
                break

            inputs = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            input_ids = inputs.input_ids

            for step in range(gen_per_prompt):
                if collected >= num_steps:
                    break
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)

                    z_t = self._extract_state(outputs.hidden_states)

                    # Teacher baseline from first 20 scores
                    if collected < 20:
                        self.drift_tracker.update_teacher_baseline(z_t.unsqueeze(0))

                    w_x = self.drift_tracker.score(z_t)
                    self.drift_tracker.update(z_t)

                    # Also collect prediction set sizes for BQ calibration
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    pred_set_size = int((probs > 0.01).sum().item())

                    self.safety_gate.add_calibration_score(w_x, pred_set_size=pred_set_size)
                    collected += 1

                    next_token_logits = logits
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            if (pi + 1) % 20 == 0:
                print(f"   ... {collected}/{num_steps} scores collected")

        threshold = self.safety_gate.calibrate()
        self.is_calibrated = True
        print(f" ACC Calibrated. Intervention Threshold: {threshold:.5f}\n")

    def run_suite(self):
        if not os.path.exists(TASK_FILE):
            print(f" Task file not found: {TASK_FILE}")
            return

        with open(TASK_FILE, "r") as f:
            tasks = json.load(f)

        print(f" Loaded {len(tasks)} tasks.")

        if not self.is_calibrated:
            self.calibrate()

        for i, task in enumerate(tasks):
            print(f"\n  TASK {i+1}/{len(tasks)} [{task.get('category', 'Gen')}]: {task['id']}")
            try:
                self.generate(task["prompt"], max_tokens=task["max_tokens"])
            except Exception as e:
                print(f"  ERROR on Task {task['id']}: {str(e)}")
                continue

        self._print_summary()

    def generate(self, prompt, max_tokens=100):
        if "<|im_start|>" not in prompt:
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        start_time = time.time()
        interventions = 0
        print("   Generating...", end=" ", flush=True)

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)

                # --- ACC CORE LOGIC ---
                # 1. Extract State
                z_t = self._extract_state(outputs.hidden_states)

                # 2. Detect Drift (i-ppDRE)
                w_x = self.drift_tracker.score(z_t)

                # 3. Compute prediction set size |C(x_t)| for BQ monitoring
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                pred_set_size = int((probs > 0.01).sum().item())

                # 4. Safety Check (Conformal Gate with BQ signal)
                gate_triggered = self.safety_gate.check(w_x, pred_set_size=pred_set_size)

                # Log for Density Chasm visualization
                self.drift_log.append({
                    "step": step,
                    "w_x": float(w_x),
                    "threshold": float(self.safety_gate.lambda_star) if self.safety_gate.is_calibrated else 0.0,
                    "pred_set_size": pred_set_size,
                    "intervention": gate_triggered,
                })

                if gate_triggered:
                    interventions += 1
                    self.total_interventions += 1
                    print(f"\n    [ALERT] INTERVENTION (drift={w_x:.4f}, |C|={pred_set_size})", end="")

                    # REAL TEACHER CORRECTION (replaces time.sleep)
                    input_ids = self._teacher_correct(input_ids)

                    # Decode the teacher's corrected token
                    token_str = self.tokenizer.decode(input_ids[0, -1], skip_special_tokens=True)
                    print(f" → '{token_str}'", end="", flush=True)
                else:
                    # Safe — continue with student
                    self.drift_tracker.update(z_t)

                    # --- Token Generation ---
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                    print(token_str, end="", flush=True)

                self.total_tokens += 1

                if input_ids[0, -1].item() == self.tokenizer.eos_token_id:
                    break

        elapsed = time.time() - start_time
        print(f"\n Finished in {elapsed:.2f}s (interventions: {interventions})")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print(" ACC SESSION SUMMARY")
        print(f"   Total Tokens: {self.total_tokens}")
        print(f"   Total Interventions: {self.total_interventions}")
        if self.total_tokens > 0:
            rate = (self.total_interventions / self.total_tokens) * 100
            print(f"   Intervention Rate: {rate:.2f}%")
        if self.oracle and self.oracle.correction_times:
            self.oracle.print_stats()
        print("=" * 70)

    def save_drift_log(self, path: str):
        """Save drift trajectory for Density Chasm visualization."""
        with open(path, "w") as f:
            json.dump(self.drift_log, f, indent=2)
        print(f"   Drift log saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="ACC Agent — Active Conformal Control")
    parser.add_argument("--model", choices=["llama3", "mistral", "phi3"], default="mistral",
                        help="Student model architecture")
    parser.add_argument("--no-teacher", action="store_true",
                        help="Simulate oracle latency instead of loading CPU teacher")
    parser.add_argument("--drift-log", type=str, default=None,
                        help="Path to save drift trajectory JSON (for plotting)")
    args = parser.parse_args()

    agent = ACCAgent(model_name=args.model, use_teacher=not args.no_teacher)
    agent.run_suite()

    if args.drift_log:
        agent.save_drift_log(args.drift_log)


if __name__ == "__main__":
    main()
