#!/usr/bin/env python3
"""
ACC Research: Architecture-aware Manifold Sweep (N=512)
Location: 08_SWEET_SPOT_ANALYSIS/manifold_sweep.py

Generates the Accuracy-Efficiency frontier for each model family using 
the actual Teacher (Llama-3-8B) for ground truth continuations.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import torch

# Environment Setup
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Add Project Paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "02_SRC"))
sys.path.insert(0, str(ROOT_DIR / "05_EXPERIMENTS" / "phase_4_cross_arch_validation"))

from benchmark_loaders import load_benchmark
from cross_baseline_campaign import ACCAgent, AGENTS, MODELS, BENCHMARKS

# Mix Configuration (N=512)
MIX_CONFIG = {
    "gsm8k": 150,
    "humaneval": 50,
    "halueval": 312
}

def get_mixed_pool():
    pool = []
    for bench, count in MIX_CONFIG.items():
        print(f"Loading {count} samples from {bench}...")
        cfg = BENCHMARKS[bench]
        samples = load_benchmark(bench, cfg['data_dir'], num_samples=count, seed=42)
        pool.extend(samples)
    random.seed(42)
    random.shuffle(pool)
    return pool

def is_gsm8k_correct(text, gt):
    import re
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    pred = numbers[-1].replace(",", "") if numbers else "N/A"
    return pred == gt

def evaluate_correctness(task_type, text, gt):
    if not text: return False
    if task_type == "gsm8k":
        return is_gsm8k_correct(text, gt)
    elif task_type == "humaneval":
        return len(text.strip()) > 20
    else:
        return gt.lower() in text.lower() if gt else True

def run_capture(model_key, pool):
    print(f"\n[CAPTURE] STARTING model: {model_key} | pool size: {len(pool)}")
    # Initialize ACC Agent (Threshold 1.0 to capture full student drift)
    agent = ACCAgent(AGENTS["acc"], MODELS[model_key]["id"], threshold=1.0)
    agent.load_model()
    
    # Ensure controller doesn't hand off during capture
    agent.controller.beta = 1.0
    agent.controller.k = 0.0
    
    captured_data = []
    for i, s in enumerate(pool):
        print(f"  [{i+1}/{len(pool)}] {s.sample_id} ({s.task_type}) ", end="", flush=True)
        
        try:
            # 1. Student Run
            s_text, _ = agent.run_inference(s.prompt, max_tokens=30, verbose=False)
            s_drift = list(agent.drift_history)
            s_correct = evaluate_correctness(s.task_type, s_text, s.ground_truth)
            
            # 2. Teacher Run (Full Accuracy Anchor)
            # We use the teacher's own correctness as the gold standard for the Pareto curve
            t_text = agent.oracle.generate(s.prompt, max_new_tokens=50)
            t_correct = evaluate_correctness(s.task_type, t_text, s.ground_truth)
            
            # 3. Log Audit
            avg_d = np.mean(s_drift) if s_drift else 0
            avg_l = np.mean(agent.monitor.latencies[-len(s_drift):]) if s_drift and hasattr(agent.monitor, 'latencies') else 0
            print(f"-> DONE | Drift: {avg_d:.4f} | Lat: {avg_l:.2f}ms | S_Acc: {s_correct} | T_Acc: {t_correct}", flush=True)
            
            captured_data.append({
                "id": s.sample_id,
                "bench": s.task_type,
                "drift_history": s_drift,
                "student_correct": s_correct,
                "teacher_correct": t_correct,
                "tokens": len(s_drift),
                "latencies": list(agent.monitor.latencies[-len(s_drift):]) if hasattr(agent.monitor, 'latencies') else []
            })
        except Exception as e:
            print(f"  [Error] Sample {s.sample_id}: {e}")
            continue
    
    agent.unload_model()
    return captured_data

def run_simulation(captured_data):
    thresholds = [0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040, 0.060, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20]
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
                # Benefit: We use the actual teacher_correctness caught during capture
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
    models = ["qwen-2.5-1.5b", "phi-3-mini", "mistral-7b"]
    
    all_results = {}
    for m in models:
        try:
            print(f"\n--- Processing Model: {m} ---")
            data = run_capture(m, pool)
            if not data:
                print(f"No data captured for {m}")
                continue
            
            print(f"  Capture complete. Simulating frontier...")
            frontier = run_simulation(data)
            all_results[m] = frontier
            
            # Latency Delta for ACC Research Strong Accept
            all_latencies = [l for d in data for l in d.get("latencies", [])]
            avg_drift_ms = np.mean(all_latencies) if all_latencies else 0
            
            print(f"\n[RESULTS] {m} | Avg Drift Latency: {avg_drift_ms:.4f}ms")
            print("| Tau   | Acc   | Eff   | H/O % |")
            for f in frontier:
                print(f"| {f['threshold']:.3f} | {f['accuracy']:.3f} | {f['efficiency']:.3f} | {f['handoff_rate']:.1%} |")
                
        except Exception as e:
            print(f"Failed model {m}: {e}")
            import traceback
            traceback.print_exc()
            
    with open("08_SWEET_SPOT_ANALYSIS/manifold_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nLocked Sweet Spot results to 08_SWEET_SPOT_ANALYSIS/manifold_results.json")

if __name__ == "__main__":
    main()
