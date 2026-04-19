import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForSequenceClassification, BitsAndBytesConfig
from typing import Tuple, List, Optional, Dict, Any
import sys
from pathlib import Path

class SemanticEntropyAgent:
    """
    Semantic Entropy baseline (Kuhn, Gal & Farquhar, Nature 2024).
    Calculates entropy by clustering multiple output samples using an NLI model.
    Updated for VLM support (campaign).
    """
    def __init__(self, model_id: str, n_samples: int = 2, entropy_threshold: float = 1.2):
        self.model_id = model_id
        self.n_samples = n_samples
        self.entropy_threshold = entropy_threshold
        self.model = None
        self.processor = None
        self.nli_model = None
        self.nli_tokenizer = None
        self.token_history = []
        self.drift_history = []
        self.handoff_events = []

    def load_model(self):
        """Load VLM and NLI models."""
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Resolve model path
        model_path = self.model_id
        if "qwen" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/qwen25vl_3b")
             if local_path.exists(): model_path = str(local_path)
        elif "llava" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/llava16_7b")
             if local_path.exists(): model_path = str(local_path)
        elif "phi" in self.model_id.lower():
             local_path = Path("/home/cse-sdpl/research/ACC/01_DATA/models/student_vlm/phi4_multimodal")
             if local_path.exists(): model_path = str(local_path)
             bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "embed_tokens_extend"],
            )

        print(f"[SemEnt] Loading VLM from {model_path}...")
        if "phi" in model_path.lower():
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            # Stability fixes from student agent
            if hasattr(self.model, "tie_weights"):
                try: self.model.tie_weights()
                except: pass
            if hasattr(self.model, "model") and not hasattr(self.model.model, "prepare_inputs_for_generation"):
                if hasattr(self.model, "prepare_inputs_for_generation"):
                    self.model.model.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load smaller NLI model for semantic clustering to save VRAM
        from transformers import AutoTokenizer
        nli_model_id = "microsoft/deberta-base-mnli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id).to(self.model.device)

    def unload_model(self):
        """Release models and VRAM."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.nli_model is not None:
            del self.nli_model
            self.nli_model = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_semantic_ids(self, sentences: List[str]) -> List[int]:
        """Cluster sentences based on NLI entailment."""
        if not sentences: return []
        semantic_ids = [0] * len(sentences)
        for i in range(1, len(sentences)):
            found_cluster = False
            for j in range(i):
                inputs = self.nli_tokenizer(sentences[i], sentences[j], return_tensors="pt").to(self.nli_model.device)
                with torch.no_grad():
                    logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                
                # Index 2 is entailment for MNLI
                if probs[0, 2] > 0.5:
                    semantic_ids[i] = semantic_ids[j]
                    found_cluster = True
                    break
            if not found_cluster:
                semantic_ids[i] = max(semantic_ids) + 1
        return semantic_ids

    def run_inference(self, prompt: str, image=None, max_tokens: int = 30) -> Tuple[str, bool]:
        """Sample multiple outputs and compute semantic entropy."""
        if self.model is None:
            self.load_model()
            
        # Build multimodal inputs
        if image is not None:
            if isinstance(image, Path):
                image = str(image)
            
            if "phi" in self.model_id.lower():
                # Phi-4 format
                from PIL import Image
                if isinstance(image, (str, Path)):
                    image = Image.open(image).convert("RGB")
                
                messages = [
                    {"role": "system", "content": "You are a concise assistant. Provide the answer immediately without conversational filler."},
                    {"role": "user", "content": "<|image_1|>\n" + prompt}
                ]
                text_input = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.processor(
                    text=text_input,
                    images=image,
                    return_tensors="pt",
                ).to(self.model.device)
            elif "qwen" in self.model_id.lower():
                messages = [
                    {"role": "system", "content": "You are a concise assistant. Provide the answer immediately without conversational filler."},
                    {"role": "user", "content": [
                        {"type": "image", "image": image, "max_pixels": 1003520},
                        {"type": "text", "text": prompt},
                    ]}
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                model_inputs = self.processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                # LLaVA style
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.processor(
                    text=text_input,
                    images=image,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
        else:
            model_inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.model.device)

        samples = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
                )
            gen_ids = output[0][model_inputs["input_ids"].shape[1]:]
            gen_text = self.processor.decode(gen_ids, skip_special_tokens=True)
            samples.append(gen_text.strip())

        # Clustering and Entropy Calculation
        semantic_ids = self._get_semantic_ids(samples)
        counts = np.bincount(semantic_ids)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

        chasm_detected = entropy > self.entropy_threshold
        self.drift_history = [entropy]
        
        return samples[0] if samples else "", chasm_detected
