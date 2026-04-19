"""
Oracle Bridge: CPU Teacher Interface for ACC Handoff

This module provides the interface between the GPU student and CPU teacher
during KV-cache handoff events. It manages:
1. Loading the FP16 teacher model on CPU
2. Accepting KV-cache from GPU student
3. Computing corrected next token with teacher
4. Returning control to student

Hardware: Intel Xeon W5-2565X with 64GB RAM
Model: Llama-3-8B-Instruct FP16 (~16GB)
"""

import time
from typing import Optional, Tuple, cast, Any
import os
from pathlib import Path

import os
# Silencing Transformers/Hub telemetry to prevent background thread JSONDecodeErrors
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1" # Assuming weights are cached after first run

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ipex version mismatch (2.8 requested, 2.10 found) causes hard crash in __init__
# We fall back to native BF16 CPU support
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: IPEX import failed: {e}")
    IPEX_AVAILABLE = False


class OracleBridge:
    """
    Manages the CPU teacher model for drift correction.
    
    This provides the "ground truth" continuation when the 4-bit student
    drifts beyond the safety threshold. The teacher runs exclusively on
    CPU to avoid VRAM contention with the student.
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cpu",
        hf_token: Optional[str] = None,
    ):
        """
        Args:
            model_id: HuggingFace model ID (must match student architecture)
            device: cpu for Xeon W5
            hf_token: HuggingFace authentication token
        """
        # Force local VLM teacher path for campaign fleet
        self.model_id = "/home/cse-sdpl/research/ACC/01_DATA/models/teacher_vlm/llama32_vision_11b"
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        # Disable online check
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        
        self.tokenizer: Optional[Any] = None
        self.teacher: Optional[Any] = None
        self.is_loaded = False
        
        # Metrics
        self.correction_count = 0
        self.correction_times = []
        self.ipex_available = False
    
    def load_teacher(self, verbose: bool = True):
        """
        Load VLM teacher model (Llama-3.2-Vision-11B) on CPU with BF16/AMX.
        """
        if self.is_loaded:
            if verbose:
                print("Teacher already loaded.")
            return

        # VLM Teacher Configuration
        if verbose:
            print(f"Loading BF16 VLM teacher: {self.model_id}")
            print(f"Device: {self.device} (System RAM: ~22.6GB)")

        start = time.time()

        from transformers import MllamaForConditionalGeneration, AutoProcessor

        # Load processor (handles images and text)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, token=self.hf_token
        )
        self.tokenizer = self.processor.tokenizer

        # Load teacher model
        teacher = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Force CPU for AMX acceleration
            low_cpu_mem_usage=True,
            token=self.hf_token,
        )
        teacher.eval()

        # Apply IPEX / AMX optimization (Intel Xeon W5)
        if IPEX_AVAILABLE:
            if verbose:
                print("  Applying IPEX optimization for Intel AMX...")
            teacher = ipex.optimize(teacher, dtype=torch.bfloat16, inplace=True)
            self.ipex_available = True
        else:
            if verbose:
                print("  [ALERT] IPEX unavailable — falling back to native BF16")

        self.teacher = teacher
        self.is_loaded = True

        elapsed = time.time() - start
        if verbose:
            print(f"VLM Teacher loaded in {elapsed:.2f}s")
    def unload_teacher(self):
        """Release teacher model from RAM."""
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None
        self.is_loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Teacher VLM unloaded from CPU.")
        print("OracleBridge: Teacher VLM unloaded from CPU.")

    def generate(self, prompt: str, image: Optional[Any] = None, max_new_tokens: int = 20) -> str:
        """Generation with optional image support (BF16 VLM Teacher)."""
        print(f"\nDEBUG [OracleBridge] prompt length: {len(prompt)}")
        print(f"DEBUG [OracleBridge] prompt preview: {prompt!r}")
        if not self.is_loaded:
            self.load_teacher()
        
        with torch.no_grad():
            if image is not None:
                if isinstance(image, Path):
                    image = str(image)
                # Multimodal prompting via chat template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                teacher_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=teacher_prompt, images=image, return_tensors="pt"
                ).to(self.device)
            else:
                # Text-only
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.teacher.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        prompt_len = inputs.get("input_ids").shape[-1]
        new_tokens = outputs[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def correct_trajectory(
        self,
        input_ids: torch.Tensor,
        image=None,
        max_new_tokens: int = 1,
        re_anchor_tokens: int = 3,
        # Hard cap: teacher never generates more than 20 tokens
        # This cuts MathVista/VQAv2 latency ~4x without hurting short answers
        _TEACHER_TOKEN_CAP: int = 20,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute corrected next token(s) using BF16 VLM teacher.
        """
        if not self.is_loaded:
            self.load_teacher()

        start = time.perf_counter()

        with torch.no_grad():
            if image is not None:
                if isinstance(image, Path):
                    image = str(image)
                # Multimodal correction: Decode student IDs and re-process with image
                text_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ]
                teacher_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                teacher_inputs = self.processor(
                    text=teacher_prompt, images=image, return_tensors="pt"
                ).to("cpu")
            else:
                # Text-only correction
                teacher_inputs = {"input_ids": input_ids.to("cpu")}

            # Generate correction tokens — hard-capped at 20 tokens for edge efficiency
            effective_tokens = min(
                max_new_tokens if max_new_tokens > 1 else re_anchor_tokens,
                _TEACHER_TOKEN_CAP
            )
            outputs = self.teacher.generate(
                **teacher_inputs,
                max_new_tokens=effective_tokens,
                do_sample=False,
                temperature=1.0,  # deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Append teacher's tokens to student context
        prompt_len = teacher_inputs.get("input_ids", input_ids).shape[-1]
        new_tokens = outputs[0, prompt_len:]
        corrected_ids = torch.cat([input_ids.cpu(), new_tokens.unsqueeze(0)], dim=1)

        compute_ms = (time.perf_counter() - start) * 1000
        self.correction_count += 1
        self.correction_times.append(compute_ms)

        return corrected_ids, compute_ms
    
    def print_stats(self):
        """Print correction statistics."""
        if not self.correction_times:
            print("No corrections performed yet.")
            return
        
        times = self.correction_times
        print()
        print("="*60)
        print("ORACLE BRIDGE STATISTICS")
        print("="*60)
        print(f"Total Corrections: {self.correction_count}")
        print(f"Mean Latency: {sum(times)/len(times):.2f}ms")
        print(f"Min Latency: {min(times):.2f}ms")
        print(f"Max Latency: {max(times):.2f}ms")
        print(f"Total CPU Time: {sum(times):.2f}ms")
        print("="*60)
        print()


def test_oracle_bridge():
    """Test the oracle bridge with a simple correction."""
    print("="*60)
    print("ORACLE BRIDGE TEST")
    print("="*60)
    
    bridge = OracleBridge()
    bridge.load_teacher()
    
    # Test prompt
    prompt = "The capital of France is"
    if bridge.tokenizer is None:
        raise RuntimeError("Tokenizer not initialized.")
    inputs = bridge.tokenizer(prompt, return_tensors="pt")
    
    print(f"Test prompt: {prompt}")
    print("Requesting correction from teacher...")
    
    # Get correction
    corrected_ids, latency = bridge.correct_trajectory(
        inputs.input_ids,
        max_new_tokens=5,
    )
    
    corrected_text = bridge.tokenizer.decode(
        corrected_ids[0], 
        skip_special_tokens=True
    )
    
    print(f"Corrected: {corrected_text}")
    print(f"Latency: {latency:.2f}ms")
    
    bridge.print_stats()


if __name__ == "__main__":
    test_oracle_bridge()
