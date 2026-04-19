#!/usr/bin/env python3
"""
campaign: Architecture-aware VLM Manifold Sweep (N=512)
Location: 08_SWEET_SPOT_ANALYSIS/manifold_sweep_vlm.py

Calibrates the Bayesian Conformal Gate (lambda*) for the VLM fleet:
- Qwen2.5-VL-3B
- Phi-4-Multimodal
- LLaVA-v1.6-7B

Uses a mixed pool of MathVista, VQAv2, POPE, and ALFWorld.
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

# Add Project Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))
sys.path.insert(0, str(ROOT_DIR / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from benchmark_loaders_vlm import load_vlm_benchmark
# Original wrappers are commented out as they will be imported dynamically inside the loop
# from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
# from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
# from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent

# Mix Configuration (N=512)
MIX_CONFIG = {
    "mathvista": 170, # train
    "pope": 171,      # val-subset
    "alfworld": 171   # train
}

NUM_SAMPLES = 512  # campaign full calibration sweep

BENCHMARK_PATHS = {
    "mathvista": ROOT_DIR / "01_DATA" / "Benchmarks" / "mathvista",
    "pope": ROOT_DIR / "01_DATA" / "Benchmarks" / "pope",
    "alfworld": ROOT_DIR / "01_DATA" / "Benchmarks" / "alfworld",
}

def get_mixed_pool():
    pool = []
    for bench, count in MIX_CONFIG.items():
        print(f"Loading {count} samples from {bench}...")
        samples = load_vlm_benchmark(bench, BENCHMARK_PATHS[bench], num_samples=count)
        pool.extend(samples)
    random.seed(42)
    random.shuffle(pool)
    return pool

def evaluate_vlm_correctness(task_type, text, gt):
    """
    Simplified correctness check for VLM tasks.
    In a full run, we would use LLM-as-a-judge or exact match for MathVista.
    """
    if not text: return False
    text = text.lower().strip()
    gt = gt.lower().strip()
    
    if task_type == "mathvista":
        # Check if GT number or label is in text
        return gt in text
    elif task_type == "pope":
        # POPE is usually 'yes' or 'no'
        return (gt in text[:10]) 
    elif task_type == "alfworld":
        return gt in text
    return gt in text

def run_capture_with_agent(agent, pool):
    print(f"\n[CAPTURE] STARTING VLM model: {agent.__class__.__name__} | pool size: {len(pool)}")
    
    # Use the Teacher from OracleBridge for capture
    from acc_core.system.oracle_bridge import OracleBridge
    teacher_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    oracle = OracleBridge(model_id=teacher_id)
    oracle.load_teacher()
    
    captured_data = []
    for i, s in enumerate(tqdm(pool)):
        try:
            # 1. Student Run (Capture drift trajectory)
            res = agent.run_text(s.question, image=s.image, max_new_tokens=30, task_id=s.sample_id)
            s_text = res["generated_text"]
            s_drift = res["drift_scores"]
            s_correct = evaluate_vlm_correctness(s.task_type, s_text, s.ground_truth)
            
            # 2. Teacher Run (Full Accuracy Anchor)
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
            
            # Periodic VRAM cleanup
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error on sample {s.sample_id}: {e}")
            continue
    
    # [RESOURCE SAFETY] Unload Teacher explicitly
    oracle.unload_teacher()
    del oracle
    gc.collect()
            
    return captured_data

def run_simulation(captured_data):
    # Wider range for VLM density chasms (including A*-tier statistical targets)
    thresholds = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.0635, 0.075, 0.0875, 0.10, 0.1069, 0.15, 0.20, 0.30, 0.50, 1.0]
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
                # Student finished
                correct_count += 1 if d["student_correct"] else 0
                student_tokens += d["tokens"]
            else:
                # Handoff occurred
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
    models = ["qwen", "phi4", "llava"]
    
    all_results = {}
    for m in models:
        print(f"\n" + "="*60)
        print(f"--- Calibrating VLM: {m} ---")
        print("="*60)
        
        try:
            # 1. Initialize agent (Load model into VRAM)
            if m == "qwen":
                from wrappers.wrapper_qwen25vl_3b import QwenVLStudentAgent
                agent = QwenVLStudentAgent()
            elif m == "phi4":
                from wrappers.wrapper_phi4_multimodal import Phi4MultimodalStudentAgent
                agent = Phi4MultimodalStudentAgent()
            elif m == "llava":
                from wrappers.wrapper_llava16_7b import LLaVA16StudentAgent
                agent = LLaVA16StudentAgent()
            else:
                raise ValueError(f"Unknown model: {m}")
            
            # 2. Run Capture
            data = run_capture_with_agent(agent, pool)
            
            # 3. Simulation
            if data:
                frontier = run_simulation(data)
                all_results[m] = frontier
                
                # [NEW] Save raw capture for resilience analysis
                # Use path relative to script location
                SCRIPT_DIR = Path(__file__).parent
                raw_filename = SCRIPT_DIR / f"raw_capture_{m}.json"
                with open(raw_filename, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Incremental save of frontier
                results_filename = SCRIPT_DIR / "vlm_manifold_results.json"
                with open(results_filename, "w") as f:
                    json.dump(all_results, f, indent=2)
                
                # Find Sweet Spot
                sweet_spot = None
                for f in reversed(frontier):
                    if f["efficiency"] >= 0.8:
                        sweet_spot = f
                        break
                
                print(f"\n[RESULTS] {m} | Sweet Spot Candidates:")
                print("| Tau   | Acc   | Eff   | H/O % |")
                for f in frontier:
                    star = " *" if f == sweet_spot else ""
                    print(f"| {f['threshold']:.3f} | {f['accuracy']:.3f} | {f['efficiency']:.3f} | {f['handoff_rate']:.1%}{star} |")
            
            # 4. Cleanup (Crucial for VRAM)
            if hasattr(agent, 'unload_model'):
                agent.unload_model()
            del agent
            gc.collect()
            torch.cuda.empty_cache()
            print(f"--- Finished {m} and cleared VRAM ---")
            
        except Exception as e:
            print(f"ERROR calibrating {m}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print(f"\nLocked VLM Sweet Spots to vlm_manifold_results.json in {SCRIPT_DIR}")

if __name__ == "__main__":
    main()
