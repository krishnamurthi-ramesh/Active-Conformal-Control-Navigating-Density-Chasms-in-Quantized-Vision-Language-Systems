import sys
import time
import torch
import numpy as np
from typing import Optional

sys.path.append('/home/cse-sdpl/research/ACC/02_SRC')
from acc_core.system.ring_buffer import ACCOpsBridge
from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate
from acc_core.system.lazy_sync import KVCacheSync
from wrappers.campaign_logger import CampaignLogger


class OracleCampaignManager:
    """
    Orchestrates local-oracle handover when the safety gate triggers.
    Uses KVCacheSync to migrate KV cache between GPU (student) and CPU (oracle).
    """

    def __init__(self, latency_budget_ms: float = 92.0):
        self.synchronizer = KVCacheSync(latency_budget_ms=latency_budget_ms)
        self._kv_cache_fetcher = None
        self._teacher_recover = None
        self._student_resume = None

    def configure_handover(self, kv_cache_fetcher, teacher_recover, student_resume) -> None:
        """
        Register callbacks to enable real KV-cache handover.

        - kv_cache_fetcher() -> torch.Tensor
        - teacher_recover(kv_cache_on_cpu) -> torch.Tensor
        - student_resume(corrected_cache_on_gpu) -> None
        """
        self._kv_cache_fetcher = kv_cache_fetcher
        self._teacher_recover = teacher_recover
        self._student_resume = student_resume

    def handle_offload(self) -> bool:
        """
        Performs KV-cache migration and teacher recovery if callbacks are set.
        Returns True if offload was requested, False otherwise.
        """
        if not (self._kv_cache_fetcher and self._teacher_recover and self._student_resume):
            return True

        kv_cache = self._kv_cache_fetcher()
        cpu_cache = self.synchronizer.transfer_to_oracle(kv_cache)
        corrected_cache = self._teacher_recover(cpu_cache)
        gpu_cache = self.synchronizer.resume_student(corrected_cache)
        self._student_resume(gpu_cache)
        return True


def run_campaign():
    print("  ACC CAMPAIGN SUPERVISOR: STARTED")

    # 1. Setup Infrastructure
    logger = CampaignLogger()
    bridge = ACCOpsBridge(create=True)
    manager = OracleCampaignManager()

    # 2. State Management
    tracker: Optional[IncrementalDriftTracker] = None
    gate: Optional[ConformalSafetyGate] = None
    calibration_mode = True
    current_task_id = "WAITING"
    last_step = -1

    try:
        while True:
            timestamp, flag, step = bridge.read_latest_state()

            # --- NEW TASK DETECTION ---
            # If step count resets to 1, we assume Student started a new prompt
            if step == 1 and last_step > 1:
                print(f"\n NEW TASK DETECTED (Was: {last_step}) - RESETTING STATE")
                tracker = IncrementalDriftTracker(input_dim=128, rff_dim=256, device='cpu')
                gate = ConformalSafetyGate(epsilon=0.05, min_threshold=0.005)
                calibration_mode = True
                current_task_id = f"task_{int(time.time())}"

            # First Run Init
            if tracker is None or gate is None:
                tracker = IncrementalDriftTracker(input_dim=128, rff_dim=256, device='cpu')
                gate = ConformalSafetyGate(epsilon=0.05, min_threshold=0.005)

            # --- PROCESSING LOOP ---
            if step != last_step and step > 0:
                # 1. Read Vector
                raw_bytes = bridge.buffer[16:16+512]
                z_t = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                z_tensor = torch.from_numpy(z_t).float()

                # 2. Measure & Update
                w_x = tracker.score(z_tensor)
                tracker.update(z_tensor)

                threshold = 0.0
                intervention = False

                if calibration_mode:
                    gate.add_calibration_score(w_x)
                    if step % 10 == 0:
                        print(f"   [{current_task_id}] Calibrating Step {step}: Score {w_x:.5f}")

                    if step >= 30:
                        calibration_mode = False

                        # --- ROBUSTNESS TWEAK FOR PAPER ---
                        # Remove top 5% outliers from calibration to avoid loose thresholds
                        scores = np.array(gate.calibration_scores)
                        cutoff = np.percentile(scores, 95)
                        gate.calibration_scores = [s for s in scores if s <= cutoff]

                        threshold = gate.calibrate()
                        print(f"   SAFETY GATE LOCKED: lambda* = {threshold:.5f}")
                else:
                    threshold = gate.lambda_star
                    if gate.check(w_x):
                        intervention = True
                        bridge.trigger_intervention()
                        manager.handle_offload()
                        print(f"   INTERVENTION | Step {step} | Score {w_x:.4f} > {threshold:.4f}")
                    else:
                        bridge.clear_intervention()

                # 3. Log Data
                logger.log_step(current_task_id, step, w_x, threshold, intervention)
                last_step = step

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n CAMPAIGN ABORTED.")
        bridge.close()
        bridge.shm.unlink()


if __name__ == "__main__":
    run_campaign()
