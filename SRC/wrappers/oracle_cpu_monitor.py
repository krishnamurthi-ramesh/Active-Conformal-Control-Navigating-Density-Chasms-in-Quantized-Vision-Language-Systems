import sys
import time
import numpy as np
import torch

sys.path.append('/home/cse-sdpl/research/ACC/02_SRC')
from acc_core.system.ring_buffer import ACCOpsBridge
from acc_core.detector.ipp_dre import IncrementalDriftTracker
from acc_core.control.conformal import ConformalSafetyGate

def run_oracle():
    print(" ORACLE (Patched): Initializing...")

    bridge = ACCOpsBridge(create=True)
    tracker = IncrementalDriftTracker(input_dim=128, rff_dim=256, device='cpu')

    # Init Gate with Noise Floor of 0.005
    gate = ConformalSafetyGate(epsilon=0.05, min_threshold=0.005)

    print(" ORACLE: Waiting for Student stream...")

    last_step = -1
    calibration_mode = True
    calibration_limit = 30

    try:
        while True:
            timestamp, flag, step = bridge.read_latest_state()

            if step != last_step and step > 0:
                # Read Data
                raw_bytes = bridge.buffer[16:16+512]
                z_t = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                z_tensor = torch.from_numpy(z_t).float()

                # 1. Measure Surprise FIRST
                w_x = tracker.score(z_tensor)

                # 2. Learn SECOND
                tracker.update(z_tensor)

                if calibration_mode:
                    gate.add_calibration_score(w_x)
                    print(f"   {step:04d}    CALIBRATING    (Score: {w_x:.5f})")

                    if step >= calibration_limit:
                        calibration_mode = False
                        drift_threshold = gate.calibrate()
                        print("   --------------------------------------------------")
                        print(f"    SAFETY GATE ACTIVE (Threshold: {drift_threshold:.5f})")
                        print("   --------------------------------------------------")
                else:
                    status = " Normal"
                    if gate.check(w_x):
                        status = " DRIFT DETECTED"
                        bridge.trigger_intervention()

                    print(f"   {step:04d}    {w_x:.5f}        {status}")

                last_step = step

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n ORACLE: Shutdown.")
        bridge.close()
        bridge.shm.unlink()

if __name__ == "__main__":
    run_oracle()
