"""
Student Model Setup — Multi-Architecture Support

Loads a 4-bit NF4 quantized student model for the ACC pipeline.
Supports the three architectures from the ACC Research proposal:
  - llama3: Meta-Llama-3-8B-Instruct (~5.5 GB VRAM)
  - mistral: Mistral-7B-Instruct-v0.3 (~4.5 GB VRAM)
  - phi3:    Phi-3-mini-4k-instruct (~2.5 GB VRAM)

Hardware: RTX 2000 Ada (16 GB VRAM)
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Model registry — matches proposal Section 3.1 and cross_baseline_campaign.py
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llama3_1b": {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "vram_estimate_gb": 1.5,
    },
    "llama3": {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "vram_estimate_gb": 5.5,
    },
    "mistral": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "vram_estimate_gb": 4.5,
    },
    "phi3": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "vram_estimate_gb": 2.5,
    },
}


def _resolve_cache_dir():
    """Resolve HuggingFace cache directory."""
    candidates = [
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/home/cse-sdpl/research/ACC/DATA/models",
        "/app/data/models/student_any4",
    ]
    for path in candidates:
        if path:
            parent = os.path.dirname(path) if os.path.isfile(path) else path
            try:
                if os.path.exists(parent) or os.access(os.path.dirname(parent), os.W_OK):
                    return path
            except Exception:
                pass
    return os.path.expanduser("~/.cache/huggingface/hub")


CACHE_DIR = _resolve_cache_dir()


def setup_model(model_name: str | None = None):
    """
    Load a 4-bit NF4 quantized student model.

    Args:
        model_name: One of 'llama3', 'mistral', 'phi3', or a full HuggingFace
                    model ID.  Defaults to the ACC_MODEL env variable, then 'mistral'.

    Returns:
        (tokenizer, model) tuple.
    """
    # Resolve model name
    model_name = model_name or os.environ.get("ACC_MODEL", "mistral")

    if model_name in MODEL_REGISTRY:
        model_info = MODEL_REGISTRY[model_name]
        model_id = model_info["id"]
        vram = model_info["vram_estimate_gb"]
    else:
        # Assume it's a raw HuggingFace model ID
        model_id = model_name
        vram = None

    print(f" Initiating Download/Load for: {model_id}")
    print(f"   Using cache directory: {CACHE_DIR}")
    print("   Quantization: 4-bit NF4 (bitsandbytes)")
    if vram:
        print(f"   Estimated VRAM: ~{vram} GB")

    # 1. NF4 Quantization Config (research-grade, matches Any4 ICML 2025 setup)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Token
    hf_token = os.environ.get("HF_TOKEN")

    # 3. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, cache_dir=CACHE_DIR
        )
        print("   ✓ Loaded tokenizer")
    except Exception as e:
        print(f"   ERROR loading tokenizer: {e}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Model (4-bit NF4, device_map='auto' places on GPU)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=nf4_config,
            device_map="auto",
            cache_dir=CACHE_DIR,
            token=hf_token,
        )
        print("   ✓ Loaded model")
    except Exception as e:
        print(f"   ERROR loading model: {e}")
        raise

    print(f" Model loaded in 4-bit NF4 successfully.")
    print(f"   Memory Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    return tokenizer, model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        default="mistral", help="Model architecture to load")
    args = parser.parse_args()
    setup_model(model_name=args.model)
