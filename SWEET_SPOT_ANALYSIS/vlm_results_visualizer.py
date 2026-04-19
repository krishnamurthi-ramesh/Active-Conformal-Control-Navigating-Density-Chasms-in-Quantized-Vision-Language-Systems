#!/usr/bin/env python3
"""
campaign: VLM Result Visualizer
Generates publication-quality figures for:
1. Efficiency-Accuracy Pareto Frontier
2. Density Chasm Traces (Drift history)
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Setup aesthetics (Matplotlib only)
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6)
})

RESULTS_FILE = "vlm_manifold_results.json"
MODELS = ["qwen", "phi4", "llava"]
MODEL_NAMES = {
    "qwen": "Qwen2.5-VL-3B",
    "phi4": "Phi-4-Multimodal",
    "llava": "LLaVA-v1.6-7B"
}
COLORS = {"qwen": "#d62728", "phi4": "#1f77b4", "llava": "#2ca02c"}

def plot_pareto_frontier():
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
    
    plt.figure(figsize=(10, 7))
    
    for m in MODELS:
        df = pd.DataFrame(data[m])
        # Sort by efficiency for smooth plotting
        df = df.sort_values("threshold")
        
        plt.plot(df["threshold"], df["accuracy"], marker='o', label=f"{MODEL_NAMES[m]} (Acc)", color=COLORS[m], linewidth=2)
        # plt.plot(df["threshold"], df["efficiency"], marker='s', linestyle='--', label=f"{MODEL_NAMES[m]} (Eff)", color=COLORS[m], alpha=0.6)

    plt.xscale('log')
    plt.xlabel("Conformal Threshold (lambda)")
    plt.ylabel("Accuracy")
    plt.title("VLM Accuracy-Efficiency Sweep (N=512)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("vlm_acc_sweep.png", dpi=300, bbox_inches='tight')
    print("Saved vlm_acc_sweep.png")

    # Second plot: Efficiency vs Accuracy
    plt.figure(figsize=(10, 7))
    for m in MODELS:
        df = pd.DataFrame(data[m])
        plt.plot(df["efficiency"], df["accuracy"], marker='o', label=MODEL_NAMES[m], color=COLORS[m], linewidth=2)
    
    plt.xlabel("Efficiency (Student Token %)")
    plt.ylabel("Accuracy")
    plt.title("ACC Pareto Frontier (VLM Fleet)")
    plt.grid(True, ls="-", alpha=0.5)
    plt.legend()
    plt.savefig("vlm_pareto_frontier.png", dpi=300, bbox_inches='tight')
    print("Saved vlm_pareto_frontier.png")

def plot_density_chasms():
    for m in MODELS:
        raw_file = f"raw_capture_{m}.json"
            
        if not Path(raw_file).exists():
            print(f"Skipping traces for {m}: {raw_file} not found")
            continue
            
        with open(raw_file, "r") as f:
            data = json.load(f)
            
        plt.figure(figsize=(10, 6))
        
        # Pick a few examples: some correct, some incorrect
        correct_ex = [d for d in data if d["student_correct"]][:3]
        # Pick the most "chasmic" incorrect samples to illustrate the gate
        incorrect_ex = sorted([d for d in data if not d["student_correct"]], 
                             key=lambda x: max(x["drift_history"]), reverse=True)[:3]
        
        for i, d in enumerate(correct_ex):
            label = f"Correct (Ex {i+1})" if i == 0 else ""
            plt.plot(d["drift_history"], color="gray", alpha=0.3, label=label)
            
        for i, d in enumerate(incorrect_ex):
            label = "Incorrect (Density Chasm)" if i == 0 else ""
            plt.plot(d["drift_history"], color=COLORS[m], alpha=0.8, linewidth=2, label=label)
            
        # Final Calibrated Thresholds (A*-tier: mu + 2.3*sigma from normalized sweep)
        # Corrected values from vlm_fleet_calibration.json
        THRESHOLDS = {
            "qwen": 0.0036,
            "phi4": 0.0032,
            "llava": 0.0019
        }
        tau = THRESHOLDS.get(m, 0.0) # Default to 0.0 if model not found, though it shouldn't happen
            
        plt.axhline(y=tau, color="black", linestyle="--", label=f"lambda* = {tau:.4f}")
        
        plt.xlabel("Token Position")
        plt.ylabel("i-ppDRE Drift Score")
        plt.title(f"Density Chasm Traces: {MODEL_NAMES[m]}")
        plt.legend()
        # Scale for compact manifolds (ensure detailed visibility)
        plt.ylim(0, 0.15)
        plt.savefig(f"vlm_chasm_{m}.png", dpi=300, bbox_inches='tight')
        print(f"Saved vlm_chasm_{m}.png")

if __name__ == "__main__":
    plot_pareto_frontier()
    plot_density_chasms()
