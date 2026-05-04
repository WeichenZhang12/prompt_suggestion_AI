"""
CodeCompletionModel
-------------------
Loads a LoRA-fine-tuned causal LM and exposes a .generate() method that
returns (completion_text, confidence_score).

Confidence score = exp( mean log-prob of generated tokens )
  → This maps log-space mean back to [0, 1] probability space.

To use your own fine-tuned model, set these env vars:
  BASE_MODEL_NAME   e.g. "codellama/CodeLlama-7b-hf"
  LORA_ADAPTER_PATH e.g. "./checkpoints/lora-adapter"

If LORA_ADAPTER_PATH is not set, the base model is used as-is (for dev/testing).
"""

import os
import math
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "")  # empty = no adapter


class CodeCompletionModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load(self):
        logger.info(f"Base model: {BASE_MODEL_NAME}")
        logger.info(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if LORA_ADAPTER_PATH:
            logger.info(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
        else:
            logger.warning("No LORA_ADAPTER_PATH set — using base model only.")
            self.model = base

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.is_loaded = True

    @torch.no_grad()
    def generate(self, prefix: str, max_new_tokens: int = 64) -> tuple[str, float]:
        """
        Returns:
            completion  : generated text (without the prefix)
            confidence  : exp(mean log-prob) of generated tokens, in (0, 1]
        """
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for determinism
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # ── Decode completion text ────────────────────────────────────────
        generated_ids = outputs.sequences[0][input_len:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # ── Confidence: mean log-prob of generated tokens ─────────────────
        # outputs.scores is a tuple of (vocab_size,) tensors, one per step
        log_probs = []
        for step_idx, score_tensor in enumerate(outputs.scores):
            token_id = generated_ids[step_idx].item()
            if token_id == self.tokenizer.eos_token_id:
                break
            log_prob = torch.log_softmax(score_tensor[0], dim=-1)[token_id].item()
            log_probs.append(log_prob)

        if not log_probs:
            return completion, 0.0

        mean_log_prob = sum(log_probs) / len(log_probs)
        confidence = math.exp(mean_log_prob)  # maps to (0, 1]

        return completion, confidence
