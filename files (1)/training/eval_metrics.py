import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "backend"))

from data import build_prompt_completion_dataset  # noqa: E402
from metrics import aggregate_metrics, evaluate_completion  # noqa: E402
from model import CodeCompletionModel  # noqa: E402

CONFIDENCE_HIGH = 0.78
CONFIDENCE_LOW = 0.70


def _ui_mode(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH:
        return "inline"
    if confidence >= CONFIDENCE_LOW:
        return "collapsed"
    return "hidden"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate completion quality on CodeSearchNet")
    p.add_argument("--language", default="python")
    p.add_argument("--split", default="validation")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_chars", type=int, default=40)
    p.add_argument("--frac_lo", type=float, default=0.2)
    p.add_argument("--frac_hi", type=float, default=0.8)
    p.add_argument("--save_json", default="", help="Optional output file for summary JSON")
    p.add_argument("--print_examples", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = build_prompt_completion_dataset(
        args.split,
        language=args.language,
        seed=args.seed,
        min_chars=args.min_chars,
        frac_lo=args.frac_lo,
        frac_hi=args.frac_hi,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No examples available after filtering. Try lower min_chars.")

    model = CodeCompletionModel()
    model.load()

    metric_rows: list[dict] = []
    confidences: list[float] = []
    latencies_ms: list[float] = []
    tier_counts = {"inline": 0, "collapsed": 0, "hidden": 0}
    example_rows: list[dict] = []

    for i, row in enumerate(dataset):
        prompt = row["prompt"]
        gold_suffix = row["completion"]

        t0 = time.perf_counter()
        pred_suffix, conf, _toks = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        m = evaluate_completion(pred_suffix, gold_suffix)
        metric_rows.append(m)
        confidences.append(conf)
        latencies_ms.append(elapsed_ms)
        tier = _ui_mode(conf)
        tier_counts[tier] += 1

        if i < args.print_examples:
            example_rows.append(
                {
                    "idx": i,
                    "confidence": round(conf, 4),
                    "ui_mode": tier,
                    "pred_suffix_preview": pred_suffix[:160],
                    "gold_suffix_preview": gold_suffix[:160],
                }
            )

    quality = aggregate_metrics(metric_rows)
    n = len(metric_rows)
    summary = {
        "config": {
            "base_model_name": os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B"),
            "lora_adapter_path": os.getenv("LORA_ADAPTER_PATH", ""),
            "language": args.language,
            "split": args.split,
            "max_samples_requested": args.max_samples,
            "max_samples_used": n,
            "max_new_tokens": args.max_new_tokens,
            "confidence_threshold_high": CONFIDENCE_HIGH,
            "confidence_threshold_low": CONFIDENCE_LOW,
        },
        "quality": quality,
        "confidence": {
            "mean": statistics.fmean(confidences) if confidences else 0.0,
            "median": statistics.median(confidences) if confidences else 0.0,
            "min": min(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0,
        },
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "median": statistics.median(latencies_ms) if latencies_ms else 0.0,
            "min": min(latencies_ms) if latencies_ms else 0.0,
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "ui_mode_counts": tier_counts,
        "ui_mode_rates": {k: (v / n if n else 0.0) for k, v in tier_counts.items()},
        "examples": example_rows,
    }

    print(json.dumps(summary, indent=2))
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
