#!/usr/bin/env python3
"""
Dedicated sweep for Phi-4-Multimodal 5.6B (N=512)
Location: 08_SWEET_SPOT_ANALYSIS/phi4_sweet_spot_rerun.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import gc
from tqdm import tqdm

# Environment Setup
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1" # Force offline to ensure local path is used

# Add Project Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))
sys.path.insert(0, str(ROOT_DIR / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from benchmark_loaders_vlm import load_vlm_benchmark
from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent

# Mix Configuration (N=512)
MIX_CONFIG = {
    "mathvista": 170,
    "pope": 171,
    "alfworld": 171
}

BENCHMARK_PATHS = {
    "mathvista": ROOT_DIR / "01_DATA" / "Benchmarks" / "mathvista",
    "pope": ROOT_DIR / "01_DATA" / "Benchmarks" / "pope",
    "alfworld": ROOT_DIR / "01_DATA" / "Benchmarks" / "alfworld",
}

def get_mixed_pool():
    pool = []
    print(f"Loading {MIX_CONFIG['mathvista']} samples from mathvista...")
    print(f"Loading {MIX_CONFIG['pope']} samples from pope...")
    print(f"Loading {MIX_CONFIG['alfworld']} samples from alfworld...")
    for bench, count in MIX_CONFIG.items():
        samples = load_vlm_benchmark(bench, BENCHMARK_PATHS[bench], num_samples=count)
        pool.extend(samples)
    random.seed(42)
    random.shuffle(pool)
    return pool[:512]

def evaluate_vlm_correctness(task_type, text, gt):
    if not text: return False
    text = text.lower().strip()
    gt = gt.lower().strip()
    if task_type == "mathvista": return gt in text
    elif task_type == "pope": return (gt in text[:10]) 
    elif task_type == "alfworld": return gt in text
    return gt in text

def run_capture(agent, pool):
    print(f"\n[CAPTURE] STARTING VLM model: {agent.__class__.__name__} | pool size: {len(pool)}")
    
    from acc_core.system.oracle_bridge import OracleBridge
    # Use local teacher path if possible
    teacher_path = "/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_vision_11b"
    oracle = OracleBridge(model_id=teacher_path)
    oracle.load_teacher()
    
    captured_data = []
    for i, s in enumerate(tqdm(pool)):
        try:
            # Student Run
            res = agent.run_text(s.question, image=s.image, max_new_tokens=30)
            s_text = res["generated_text"]
            s_drift = res["drift_scores"]
            s_correct = evaluate_vlm_correctness(s.task_type, s_text, s.ground_truth)
            
            # Teacher Run
            t_text_raw = oracle.generate(s.question, image=s.image, max_new_tokens=30)
            t_correct = evaluate_vlm_correctness(s.task_type, t_text_raw, s.ground_truth)
            
            captured_data.append({
                "id": s.sample_id,
                "bench": s.task_type,
                "drift_history": s_drift,
                "student_correct": s_correct,
                "teacher_correct": t_correct,
                "tokens": len(s_drift)
            })
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            # Periodic save to monitor and prevent data loss
            if (i + 1) % 5 == 0:
                with open("phi4_rerun_capture.json", "w") as f:
                    json.dump(captured_data, f, indent=2)
                print(f"[Phi4-MM] Periodic save at sample {i+1}")
        except Exception as e:
            print(f"Error on sample {s.sample_id}: {e}")
            continue
    
    oracle.unload_teacher()
    return captured_data

def run_simulation(captured_data):
    thresholds = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    frontier = []
    for tau in thresholds:
        correct_count = 0
        student_tokens = 0
        total_tokens = 0
        for d in captured_data:
            drift = d["drift_history"]
            handoff_idx = -1
            for idx, val in enumerate(drift):
                if val > tau:
                    handoff_idx = idx
                    break
            if handoff_idx == -1:
                correct_count += 1 if d["student_correct"] else 0
                student_tokens += d["tokens"]
            else:
                correct_count += 1 if d["teacher_correct"] else 0
                student_tokens += handoff_idx
            total_tokens += d["tokens"]
            
        acc = correct_count / len(captured_data) if captured_data else 0
        eff = student_tokens / total_tokens if total_tokens > 0 else 0
        
        frontier.append({
            "threshold": tau,
            "accuracy": acc,
            "efficiency": eff,
            "handoff_rate": sum(1 for d in captured_data if any(v > tau for v in d["drift_history"])) / len(captured_data) if captured_data else 0
        })
    return frontier

def main():
    pool = get_mixed_pool()
    print("\n" + "="*60)
    print("--- Calibrating VLM: phi4 ---")
    print("="*60)
    
    try:
        agent = Phi4MultimodalStudentAgent()
        data = run_capture(agent, pool)
        
        if data:
            frontier = run_simulation(data)
            
            SCRIPT_DIR = Path(__file__).parent
            with open(SCRIPT_DIR / "phi4_rerun_capture.json", "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"\n[RESULTS] phi4 | Sweet Spot Candidates:")
            print("| Tau   | Acc   | Eff   | H/O % |")
            for f in frontier:
                star = " *" if f['threshold'] == 1.0 else ""
                print(f"| {f['threshold']:.3f} | {f['accuracy']:.3f} | {f['efficiency']:.3f} | {f['handoff_rate']:.1%}{star} |")
            
            print(f"[Phi4-MM] Model and Oracle unloaded successfully.")
            print(f"--- Finished phi4 and cleared VRAM ---")

        if hasattr(agent, 'unload_model'):
            agent.unload_model()
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
