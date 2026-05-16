import argparse
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import build_prompt_completion_dataset  # noqa: E402


def _default_target_modules() -> list[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT on CodeSearchNet")
    p.add_argument(
        "--model_name",
        default=os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B"),
        help="Base model id (same as backend BASE_MODEL_NAME)",
    )
    p.add_argument(
        "--output_dir",
        default=os.getenv("LORA_OUTPUT_DIR", "./checkpoints/lora-adapter"),
        help="Where to save the LoRA adapter (set LORA_ADAPTER_PATH to this in the API)",
    )
    p.add_argument("--language", default="python", help="CodeSearchNet config name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=40)
    p.add_argument("--frac_lo", type=float, default=0.2)
    p.add_argument("--frac_hi", type=float, default=0.8)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--eval_strategy", default="epoch")
    p.add_argument("--dataset_num_proc", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        default=None,
        help="Comma-separated LoRA target module names (default: Qwen-style projections)",
    )
    p.add_argument("--no_eval", action="store_true", help="Disable validation pass")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_prompt_completion_dataset(
        "train",
        language=args.language,
        seed=args.seed,
        min_chars=args.min_chars,
        frac_lo=args.frac_lo,
        frac_hi=args.frac_hi,
        max_samples=args.max_train_samples,
        num_proc=args.dataset_num_proc,
    )

    if args.no_eval:
        eval_ds = None
        eval_strategy = "no"
    else:
        eval_ds = build_prompt_completion_dataset(
            "validation",
            language=args.language,
            seed=args.seed + 1,
            min_chars=args.min_chars,
            frac_lo=args.frac_lo,
            frac_hi=args.frac_hi,
            max_samples=args.max_eval_samples,
            num_proc=args.dataset_num_proc,
        )
        eval_strategy = args.eval_strategy

    use_cuda = torch.cuda.is_available()
    model_init_kwargs: dict = {"trust_remote_code": True}
    if use_cuda:
        model_init_kwargs["device_map"] = "auto"
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_init_kwargs["torch_dtype"] = torch.float16

    target_modules = (
        [s.strip() for s in args.target_modules.split(",") if s.strip()]
        if args.target_modules
        else _default_target_modules()
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=eval_strategy,
        save_total_limit=2,
        load_best_model_at_end=False,
        max_length=args.max_length,
        gradient_checkpointing=True,
        model_init_kwargs=model_init_kwargs,
        bf16=use_cuda and model_init_kwargs.get("torch_dtype") is torch.bfloat16,
        fp16=use_cuda and model_init_kwargs.get("torch_dtype") is torch.float16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.model_name,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
