"""
ACC Fleet Deployment Configuration
Generated from VLM Manifold Sweep (N=512)
Includes final calibrated thresholds (lambda*) for cross-machine sync.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Calibrated Conformal Thresholds (lambda*)
# ─────────────────────────────────────────────────────────────────────────────
ACC_VLM_FLEET = {
    "qwen2.5-vl-3b": {
        "lambda_star": 0.0036,
        "base_accuracy": 0.301,
        "acc_accuracy": 0.340,
        "efficiency": 0.654,  # Median efficiency in collapse band
        "quantization": "4-bit AWQ",
    },
    "phi-4-multimodal": {
        "lambda_star": 0.0032,
        "base_accuracy": 0.363,
        "acc_accuracy": 0.365,
        "efficiency": 0.985,
        "quantization": "4-bit NF4",
    },
    "llava-v1.6-7b": {
        "lambda_star": 0.0019,
        "base_accuracy": 0.336,
        "acc_accuracy": 0.350,
        "efficiency": 0.816,
        "quantization": "4-bit NF4",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Oracle Bridge (Teacher) Configuration
# ─────────────────────────────────────────────────────────────────────────────
ORACLE_CONFIG = {
    "model_id": "/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_vision_11b",
    "device": "cpu",          # Intel Xeon W5-2565X
    "dtype": "bfloat16",      # BF16 for AMX acceleration
    "ipex_enabled": True,      # Intel PyTorch Extension
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Final Campaign: The "Power Trio" + Action Benchmarks (Total N=5,137)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_SPLITS = {
    "pope": {
        "num_samples": 500,
        "path": "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/pope",
        "description": "Visual hallucination and object probing (Centerpiece)"
    },
    "mathvista": {
        "num_samples": 500,  
        "path": "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/mathvista",
        "description": "Multi-step visual reasoning and mathematical logic"
    },
    "vqav2": {
        "num_samples": 500, 
        "path": "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/vqav2",
        "description": "Standard open-ended visual question answering"
    },
    "alfworld": {
        "num_samples": 500,
        "path": "/home/cse-sdpl/research/ACC/01_DATA/Benchmarks/alfworld",
        "description": "Embodied AI and sequential action tasks (Generalization)"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. System Guarantees
# ─────────────────────────────────────────────────────────────────────────────
EPSILON = 0.05  # 95% Manifold Coverage Guarantee
PROJECTION_DIM = 128
RFF_DIM = 512
