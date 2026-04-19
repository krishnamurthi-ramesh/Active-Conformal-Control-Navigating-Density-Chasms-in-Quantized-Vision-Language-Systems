<div align="center">

<h1>⚡ Active Conformal Control (ACC)</h1>

<h3>Hardware-Aware Safety Monitoring for Vision-Language Models<br>via Density Chasm Detection</h3>

<p>
  <a href="https://eccv.ecva.net/"><img src="https://img.shields.io/badge/ECCV-2026-blue?style=flat-square&logo=academia&logoColor=white" alt="ECCV 2026"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/CUDA-12.1+-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.1+"/>
  <img src="https://img.shields.io/badge/Intel_AMX-Accelerated-0071C5?style=flat-square&logo=intel&logoColor=white" alt="Intel AMX"/>
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#key-contributions">Contributions</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#reproducing-results">Results</a> ·
  <a href="#citation">Citation</a>
</p>

<img src="ACC_ARCHITECTURE_OVERVIEW.md" width="0" height="0"/> <!-- anchor -->

</div>

---

## Overview

**Active Conformal Control (ACC)** is a real-time safety framework that detects and mitigates hallucinations in quantized Vision-Language Models (VLMs) through a hardware-aware, dual-system cascade architecture.

Modern deployment of VLMs on edge hardware requires aggressive quantization (INT4/FP16), which introduces **Density Chasms** — regions in latent space where a quantized student model's behavior diverges from its full-precision teacher. ACC monitors this drift in real-time and intervenes through a secondary, CPU-accelerated oracle — achieving safety without sacrificing throughput.

> **Submitted to ECCV 2026** — European Conference on Computer Vision.

---

## Key Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **ppDRE Density Sensor** | Non-parametric density ratio estimator using Random Fourier Features (RFF) for real-time drift detection on live token streams |
| 2 | **Hardware-Aware Cascade** | Exploits Intel AMX (Advanced Matrix Extensions) for teacher inference on CPU alongside GPU student inference — no second GPU required |
| 3 | **Conformal Safety Guarantees** | Distribution-free coverage guarantees via conformal prediction; statistically rigorous intervention thresholds |
| 4 | **Comprehensive Evaluation** | 9 baselines · 3 VLM families · 4 benchmarks · 2 evaluation scales (N=100, N=500) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ACC: Dual-System Safety Pipeline                   │
│                                                                     │
│  ┌─────────────────────┐         ┌──────────────────────────────┐  │
│  │  SYSTEM 1 (GPU)     │         │  ACC CORE: Safety Monitor    │  │
│  │  ─────────────────  │         │  ────────────────────────    │  │
│  │  Student VLM        │─ z_t ──▶│  ppDRE Density Sensor     │  │
│  │  (4-bit quantized)  │         │  Conformal Set Estimator     │  │
│  │  LLaVA / Qwen / Phi │─logits─▶│  Threshold λ* Controller    │  │
│  └─────────────────────┘         └──────────┬───────────────────┘  │
│                                             │                       │
│                                     drift > λ* ?                   │
│                                    ╱              ╲                 │
│                              ✅ SAFE           ⚠️ CHASM              │
│                                ↓                   ↓               │
│                           Commit Token    ┌──────────────────────┐ │
│                                           │  SYSTEM 2 (CPU)      │ │
│                                           │  ──────────────────  │ │
│                                           │  Teacher LLM         │ │
│                                           │  (FP16 · Intel AMX)  │ │
│                                           │  Corrected Output    │ │
│                                           └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
ACC/
├── SRC/                        # Core source code
│   ├── acc_core/               #   ACC algorithm (density sensor, control, cascade)
│   │   ├── detector/           #     ppDRE density ratio estimator
│   │   ├── control/            #     Conformal control & threshold logic
│   │   └── system/             #     System 1/2 cascade manager
│   └── wrappers/               #   VLM model wrappers & baseline agents
│
├── BASELINES/                  # 8 baseline implementations
│   ├── any4/                   #   Any4 (quantization baseline)
│   ├── crc/                    #   CRC (conformal risk control)
│   ├── opera/                  #   OPERA (attention-based)
│   ├── spinquant/              #   SpinQuant
│   ├── react/                  #   ReAct (reasoning agent)
│   ├── ppdre/                  #   ppDRE (density estimation)
│   ├── semantic_entropy/       #   Semantic Entropy
│   └── vista/                  #   VISTA
│
├── RESULTS/                    # All experimental results (JSON/JSONL)
│   ├── phase_1/                #   Calibration phase (N=500)
│   └── phase_2/                #   Full evaluation (N=100)
│
├── SWEET_SPOT_ANALYSIS/        # Threshold Pareto analysis & tuning
│
├── ENV/                        # Environment setup scripts
│
├── DATA/                       # ⬇️  NOT included — see Data Setup below
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| CUDA | 12.1+ |
| GPU VRAM | 16 GB+ recommended |
| CPU | Intel Sapphire Rapids+ (for AMX) |
| RAM | 64 GB+ |

### Setup

```bash
# 1. Clone the repository
git clone git@github.com:krishnamurthi-ramesh/Active-Conformal-Control-Navigating-Density-Chasms-in-Quantized-Vision-Language-Systems.git
cd Active-Conformal-Control-Navigating-Density-Chasms-in-Quantized-Vision-Language-Systems

# 2. Create a conda environment
conda create -n acc python=3.11 -y
conda activate acc

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data Setup

The benchmark datasets are **not included** in this repository due to size. Download them from their official sources:

| Benchmark | Domain | Official Source |
|---|---|---|
| **POPE** | Object Hallucination | [GitHub: AoiDragon/POPE](https://github.com/AoiDragon/POPE) |
| **VQAv2** | Visual Question Answering | [visualqa.org](https://visualqa.org/) |
| **MathVista** | Mathematical Reasoning | [HuggingFace: AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) |
| **ALFWorld** | Embodied Reasoning | [GitHub: alfworld/alfworld](https://github.com/alfworld/alfworld) |

Place downloaded data under `DATA/Benchmarks/<benchmark_name>/`.

### Model Weights

Download the required model weights to `DATA/models/`:

```bash
# Student VLMs (4-bit quantized)
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download microsoft/Phi-4-multimodal-instruct

# Teacher LLM (FP16 for AMX CPU inference)
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Reproducing Results

### Phase 1 — Calibration (N=500)

```bash
python SRC/wrappers/run_acc_student.py --phase 1 --benchmark pope --n 500
python SRC/wrappers/run_acc_student.py --phase 1 --benchmark vqav2 --n 500
```

### Phase 2 — Full Evaluation (N=100)

```bash
# Run the full cross-baseline campaign
python SRC/cross_baseline_campaign.py --phase 2 --n 100

# Run specific baseline for comparison
python SRC/wrappers/run_acc_student.py --baseline any4 --benchmark mathvista --n 100
```

### Analysis & Figures

```bash
# Threshold Pareto analysis
python SWEET_SPOT_ANALYSIS/pareto_analysis.py

# Generate paper figures
python generate_teacher_manifolds.py
```

---

## Results Overview

### Models Evaluated

| Model | Family | Quantization | Hardware |
|---|---|---|---|
| LLaVA-v1.6-7B | LLaVA | 4-bit (AWQ) | GPU |
| Qwen2.5-VL-3B | Qwen | 4-bit (GPTQ) | GPU |
| Phi-4-Multimodal | Phi | 4-bit (BnB) | GPU |
| LLaMA-3-8B | Teacher | FP16 | CPU (AMX) |

### Baselines Compared

`ACC` · `Any4` · `CRC` · `OPERA` · `SpinQuant` · `ReAct` · `ppDRE` · `Semantic Entropy` · `VISTA`

> Full quantitative results are available in [`RESULTS/`](RESULTS/) as JSON/JSONL files.

---

## Hardware

All experiments were conducted on:

- **GPU**: NVIDIA RTX (16GB+ VRAM) for quantized student inference
- **CPU**: Intel Xeon w5-3435X (Sapphire Rapids) with **AMX** for teacher inference
- **RAM**: 256 GB DDR5
- **OS**: Ubuntu 22.04 LTS

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{acc2026eccv,
  title     = {Active Conformal Control: Hardware-Aware Safety Monitoring for
               Vision-Language Models via Density Chasm Detection},
  author    = {Anonymous},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>ECCV 2026 · Computer Vision · Safety · Conformal Prediction · Edge AI</sub>
</div>
