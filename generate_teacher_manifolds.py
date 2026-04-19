#!/usr/bin/env python3
"""
Generate Teacher Manifold Baselines for UAI 2026

Creates reference manifolds from the Teacher model (Llama-3-8B-Instruct)
for ppDRE drift detection. These manifolds represent the "territory of truth"
that the Student must stay within.

Mathematical Foundation:
- Extract hidden states from Teacher on calibration data
- Project to 512d via Orthogonal RFF (Random Fourier Features)
- Train i-ppDRE density ratio estimator
- Save manifold for runtime drift detection

Usage:
    python generate_teacher_manifolds.py --teacher meta-llama/Meta-Llama-3-8B-Instruct
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'SRC')))

from acc_core.detector.rff_kernel import RandomFourierFeatures


def generate_calibration_prompts() -> List[str]:
    """Generate diverse calibration prompts for manifold extraction.
    
    These should cover the reasoning domains we care about:
    - Mathematical reasoning (GSM8K-style)
    - Logical reasoning
    - Factual knowledge
    - Multi-step reasoning
    - Wikipedia (general knowledge)
    - C4 (general English)
    """
    prompts = []
    
    # 1. Load GSM8K dataset (200 samples)
    print("Loading GSM8K dataset...")
    try:
        from datasets import load_dataset
        gsm8k = load_dataset("gsm8k", "main", split="train")
        gsm8k_prompts = [item["question"] for item in gsm8k.select(range(min(200, len(gsm8k))))]
        prompts.extend(gsm8k_prompts)
        print(f"  ✓ Added {len(gsm8k_prompts)} GSM8K prompts")
    except Exception as e:
        print(f"  [ALERT]️  Could not load GSM8K: {e}")
        # Fallback math prompts
        gsm8k_prompts = [
            f"If a train travels at {60+i} mph for {2+i%3} hours, how far does it go?"
            for i in range(200)
        ]
        prompts.extend(gsm8k_prompts)
        print(f"  ✓ Added {len(gsm8k_prompts)} fallback math prompts")
    
    # 2. Logical Reasoning (100 samples)
    print("Generating logical reasoning prompts...")
    logical_templates = [
        "All {A} have {B}. {C} are {A}. Do {C} have {B}?",
        "If {A}, then {B}. {B} is true. Did {A} happen?",
        "Either {A} or {B} is true. {A} is false. What about {B}?",
        "If {A} implies {B}, and {B} implies {C}, does {A} imply {C}?",
        "No {A} are {B}. Some {C} are {A}. Are some {C} {B}?",
    ]
    logical_prompts = []
    for i in range(100):
        template = logical_templates[i % len(logical_templates)]
        prompt = template.format(A=f"X{i}", B=f"Y{i}", C=f"Z{i}")
        logical_prompts.append(prompt)
    prompts.extend(logical_prompts)
    print(f"  ✓ Added {len(logical_prompts)} logical reasoning prompts")
    
    # 3. Factual Knowledge (50 samples)
    print("Generating factual knowledge prompts...")
    factual_prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for water?",
        "When did World War II end?",
        "What is the largest planet in our solar system?",
    ] * 10
    prompts.extend(factual_prompts)
    print(f"  ✓ Added {len(factual_prompts)} factual prompts")
    
    # 4. Multi-step Reasoning (50 samples)
    print("Generating multi-step reasoning prompts...")
    multistep_prompts = [
        f"John has {3+i} apples. Mary gives him {2+i} more. Then he eats {1+i%2}. How many does he have?"
        for i in range(50)
    ]
    prompts.extend(multistep_prompts)
    print(f"  ✓ Added {len(multistep_prompts)} multi-step prompts")
    
    # 5. Wikipedia samples (50 samples)
    print("Loading Wikipedia samples (20231101.en)...")
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wiki_prompts = []
        for i, item in enumerate(wiki):
            if i >= 50:
                break
            text = item["text"]
            # Take first 500 characters as prompt (more context)
            wiki_prompts.append(text[:500].strip())
        prompts.extend(wiki_prompts)
        print(f"  ✓ Added {len(wiki_prompts)} Wikipedia prompts")
    except Exception as e:
        print(f"  [ALERT]️  Could not load Wikipedia: {e}")
        wiki_prompts = [f"Wikipedia article {i}: General knowledge text." for i in range(50)]
        prompts.extend(wiki_prompts)
        print(f"  ✓ Added {len(wiki_prompts)} fallback Wikipedia prompts")
    
    # 6. C4 samples (50 samples)
    print("Loading C4 samples (en subset)...")
    try:
        from datasets import load_dataset
        c4 = load_dataset("allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz", split="train", streaming=True)
        c4_prompts = []
        for i, item in enumerate(c4):
            if i >= 50:
                break
            text = item["text"]
            # Take first 500 characters (more context)
            c4_prompts.append(text[:500].strip())
        prompts.extend(c4_prompts)
        print(f"  ✓ Added {len(c4_prompts)} C4 prompts")
    except Exception as e:
        print(f"  [ALERT]️  Could not load C4: {e}")
        c4_prompts = [f"C4 text sample {i}: General English content." for i in range(50)]
        prompts.extend(c4_prompts)
        print(f"  ✓ Added {len(c4_prompts)} fallback C4 prompts")
    
    # 7. Edge Cases (12 samples)
    print("Generating edge case prompts...")
    edge_prompts = [
        "The", "Once upon a time", "In conclusion,", "Therefore,",
        "However,", "Furthermore,", "Nevertheless,", "Consequently,",
        "Meanwhile,", "Ultimately,", "First,", "Finally,",
    ]
    prompts.extend(edge_prompts)
    print(f"  ✓ Added {len(edge_prompts)} edge case prompts")
    
    print()
    print(f"✓ Generated {len(prompts)} total prompts")
    print()
    
    return prompts


def extract_teacher_activations(
    model_id: str,
    prompts: List[str],
    num_samples: int = 500,
    hf_token: str = None,
) -> Tuple[np.ndarray, dict]:
    """Extract hidden state activations from teacher model.
    
    Args:
        model_id: HuggingFace model ID (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
        prompts: List of calibration prompts
        num_samples: Number of activation samples to extract
        hf_token: HuggingFace API token for gated models
        
    Returns:
        (activations, metadata): Activations array [num_samples, hidden_dim] and metadata dict
    """
    print("=" * 80)
    print("TEACHER MANIFOLD EXTRACTION")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Target samples: {num_samples}")
    print()
    
    # Load teacher model
    print("Loading teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Use CPU to avoid VRAM issues
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.eval()
    
    hidden_dim = model.config.hidden_size
    print(f"✓ Model loaded (hidden_dim={hidden_dim})")
    print()
    
    # Extract activations
    print("Extracting activations...")
    activations = []
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if len(activations) >= num_samples:
                break
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Forward pass with output_hidden_states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract last layer hidden states [batch, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Take mean over sequence length to get [batch, hidden_dim]
            # This represents the "semantic center" of the prompt
            mean_hidden = hidden_states.mean(dim=1).float().cpu().numpy()  # Convert BF16 to FP32 first
            
            activations.append(mean_hidden[0])  # Remove batch dim
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(prompts)} prompts ({len(activations)} samples)")
    
    activations = np.array(activations[:num_samples])
    
    print()
    print(f"✓ Extracted {len(activations)} activation samples")
    print(f"  Shape: {activations.shape}")
    print(f"  Dtype: {activations.dtype}")
    print()
    
    metadata = {
        "model_id": model_id,
        "hidden_dim": hidden_dim,
        "num_samples": len(activations),
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    return activations, metadata


def project_to_rff(
    activations: np.ndarray,
    rff_dim: int = 512,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, RandomFourierFeatures]:
    """Project activations to RFF feature space.
    
    Args:
        activations: Raw activations [num_samples, hidden_dim]
        rff_dim: Target RFF dimension (512 is the "Goldilocks" zone)
        sigma: RFF bandwidth (kernel width)
        
    Returns:
        (projected, rff_kernel): Projected features and fitted RFF kernel
    """
    print("=" * 80)
    print("RFF PROJECTION")
    print("=" * 80)
    print(f"Input dim: {activations.shape[1]}")
    print(f"RFF dim: {rff_dim}")
    print(f"Bandwidth σ: {sigma}")
    print()
    
    # Initialize RFF kernel
    rff = RandomFourierFeatures(
        input_dim=activations.shape[1],
        rff_dim=rff_dim,
        sigma=sigma,
    )
    
    # Project
    print("Projecting to RFF space...")
    projected = rff.forward(torch.from_numpy(activations).float()).numpy()
    
    print(f"✓ Projected to shape: {projected.shape}")
    print()
    
    return projected, rff


def save_manifold(
    activations: np.ndarray,
    metadata: dict,
    output_path: Path,
):
    """Save manifold baseline to disk.
    
    Args:
        activations: Activation samples [num_samples, dim]
        metadata: Metadata dict
        output_path: Output file path (.npy)
    """
    print("=" * 80)
    print("SAVING MANIFOLD")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Shape: {activations.shape}")
    print(f"Size: {activations.nbytes / 1024 / 1024:.2f} MB")
    print()
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save activations
    np.save(output_path, activations)
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved manifold: {output_path}")
    print(f"✓ Saved metadata: {metadata_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate teacher manifold baselines")
    parser.add_argument(
        "--teacher",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Teacher model ID"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of activation samples"
    )
    parser.add_argument(
        "--rff-dim",
        type=int,
        default=512,
        help="RFF projection dimension"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="DATA/models/teacher_manifolds",
        help="Output directory"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token"
    )
    
    args = parser.parse_args()
    
    # Use environment variable if token not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    print()
    print("=" * 80)
    print("TEACHER MANIFOLD GENERATION - UAI 2026")
    print("=" * 80)
    print()
    
    # Generate calibration prompts
    prompts = generate_calibration_prompts()
    print(f"Generated {len(prompts)} calibration prompts")
    print()
    
    # Extract activations
    activations, metadata = extract_teacher_activations(
        model_id=args.teacher,
        prompts=prompts,
        num_samples=args.num_samples,
        hf_token=hf_token,
    )
    
    # Project to RFF (optional - can save raw activations too)
    # projected, rff_kernel = project_to_rff(activations, rff_dim=args.rff_dim)
    
    # Save manifold
    output_dir = Path(args.output_dir)
    
    # Create clean filename from model ID
    model_name = args.teacher.split('/')[-1].lower().replace('-', '_')
    output_path = output_dir / f"{model_name}_manifold.npy"
    
    save_manifold(activations, metadata, output_path)
    
    print("=" * 80)
    print("MANIFOLD GENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Train i-ppDRE sensor on this manifold")
    print("2. Calibrate conformal threshold lambda*")
    print("3. Run campaign with drift detection")
    print()


if __name__ == "__main__":
    main()
