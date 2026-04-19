# ACC: Sweet Spot Analysis & Calibration

This directory contains the scripts, data, and visualizations used to calibrate the **Active Conformal Control (ACC)** thresholds and analyze the **Pareto Frontier** between latency and accuracy.

## Contents

### 1. Calibration Scripts
- `manifold_sweep.py`: Generates the teacher manifold (ground truth distribution) for density ratio estimation.
- `manifold_sweep_vlm.py`: VLM-specific adaptation for vision-language latent spaces.
- `phi4_sweet_spot_rerun.py`: Reproduction script for Phi-4-Multimodal threshold optimization.

### 2. Result Data (JSON)
- `manifold_results.json` / `vlm_manifold_results.json`: Cached density distribution statistics.
- `raw_capture_*.json`: Raw logit and hidden state traces from model sweeps, used to generate the Pareto plots.

### 3. Visualizations (PNG)
- `vlm_pareto_frontier.png`: **Primary Result** showing the Sweet Spot (λ*) optimization.
- `vlm_chasm_*.png`: Visual evidence of "Density Chasms" — regions where quantized student models lose logic integrity compared to the teacher.
- `vlm_acc_sweep.png`: Accuracy vs. Threshold sensitivity analysis.

### 4. Utilities
- `vlm_results_visualizer.py`: Tool for plotting raw capture data onto the research figures.

## Usage

To reproduce the Pareto Frontier analysis:
```bash
python vlm_results_visualizer.py
```
This will ingest the `raw_capture_*.json` files and regenerate the Pareto plots found in the paper.
