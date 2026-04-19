import csv
import time
import os
from datetime import datetime
from pathlib import Path

class CampaignLogger:
    def __init__(self, model_id="unknown"):
        # Create a unique folder for this experiment run
        self.run_id = datetime.now().strftime("campaign_%Y%m%d_%H%M")
        self.model_tag = model_id.replace("/", "_").replace(".", "")
        self.log_dir = Path(f"/home/cse-sdpl/research/ACC/06_RESULTS/logs/{self.run_id}_{self.model_tag}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 1. Trajectory Log (For Density Chasm Plot)
        # Expanded: Includes velocity and timing splits
        self.traj_file = self.log_dir / "trajectory_data.csv"
        with open(self.traj_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "task_id", "step", 
                "drift_score", "drift_velocity", 
                "vlm_forward_ms", "acc_overhead_ms", 
                "threshold", "intervention_active"
            ])

        # 2. Handoff Log (For Transparency)
        self.handoff_file = self.log_dir / "handoffs.log"
        with open(self.handoff_file, 'w') as f:
            f.write(f"=== ACC HANDOFF LOG: {self.run_id} | Model: {model_id} ===\n")

        print(f" [ACC Logger] Logging to: {self.log_dir}")

    def log_step(self, task_id, step, score, velocity, vlm_ms, acc_ms, threshold, active):
        """Records token-level metrics for Density Chasm visualization."""
        with open(self.traj_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), task_id, step, 
                round(score, 4), round(velocity, 4), 
                round(vlm_ms, 2), round(acc_ms, 2), 
                threshold, int(active)
            ])

    def log_handoff(self, task_id, token_idx, drift, teacher):
        """Records the exact moment of teacher intervention."""
        with open(self.handoff_file, 'a') as f:
            log_str = f"[HANDOFF] {task_id} | Token Index: {token_idx} | Drift: {drift:.4f} | Teacher: {teacher}\n"
            f.write(log_str)
            # Also print to stdout for real-time monitoring
            print(f"  [ALERT] {log_str.strip()}")
