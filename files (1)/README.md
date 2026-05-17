# Backend — AI Code Completion API

## Quick Start

```bash
cd backend
pip install -r requirements.txt

# Default model is Qwen/Qwen2.5-Coder-1.5B (small, runs on CPU).
# Override via env var if you want a different base model:
# export BASE_MODEL_NAME="codellama/CodeLlama-7b-hf"
# export LORA_ADAPTER_PATH="./checkpoints/lora-adapter"   # omit if not ready yet

uvicorn main:app --reload --port 8000
```

## API

### `POST /api/complete`

**Request:**
```json
{
  "prefix": "def sort_list(arr):\n    ",
  "max_new_tokens": 64
}
```

**Response:**
```json
{
  "completion": "return sorted(arr)",
  "confidence": 0.8731,
  "ui_mode": "inline"
}
```

`ui_mode` values:
| Value | Confidence | Frontend behavior |
|-------|-----------|-------------------|
| `inline` | ≥ 0.78 | Show as ghost text |
| `collapsed` | 0.70 – 0.77 | Show in expandable panel |
| `hidden` | < 0.70 | Suppress entirely |

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true }`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL_NAME` | `Qwen/Qwen2.5-Coder-1.5B` | HuggingFace model ID |
| `LORA_ADAPTER_PATH` | *(empty)* | Path to LoRA adapter dir. Leave empty to use base model. |

## Confidence Score Formula

```
confidence = exp( mean( log P(token_i) for each generated token_i ) )
```

This converts mean log-probability back to a [0, 1] probability value.

---

## Training & evaluation pipeline


`Qwen/Qwen2.5-Coder-1.5B`, 12k samples, 2 epochs, LoRA r=32, eval n=200. If VRAM OOM: lower `--batch_size` / `--max_train_samples` or use 0.5B.


### One-time setup

```powershell
conda create -n cs568 python=3.11 -y
conda activate cs568
cd "~\prompt_suggestion_AI\files (1)"   # change to your clone path
pip install -r requirements.txt
```

### Run from project root

```powershell
$CONDA_ENV = "cs568"
cd "~\prompt_suggestion_AI\files (1)"
conda activate $CONDA_ENV
pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path results | Out-Null
```

### 1. Train LoRA

Output: `./checkpoints/lora-adapter`

```powershell
$env:BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
python -X utf8 training/train_lora.py `
  --output_dir ./checkpoints/lora-adapter `
  --language python `
  --max_train_samples 12000 `
  --num_epochs 2 `
  --batch_size 4 `
  --grad_accum 4 `
  --max_length 512 `
  --lora_r 32 `
  --lora_alpha 64 `
  --no_eval
```

### 2. Eval — base model only

```powershell
$env:BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
Remove-Item Env:\LORA_ADAPTER_PATH -ErrorAction SilentlyContinue
$env:LORA_ADAPTER_PATH = ""
python training/eval_metrics.py `
  --split validation `
  --language python `
  --seed 42 `
  --max_samples 200 `
  --max_new_tokens 128 `
  --save_json results/final_base.json `
  --print_examples 5
```

### 3. Eval — with LoRA

```powershell
$env:LORA_ADAPTER_PATH = "./checkpoints/lora-adapter"
python training/eval_metrics.py `
  --split validation `
  --language python `
  --seed 42 `
  --max_samples 200 `
  --max_new_tokens 128 `
  --save_json results/final_lora.json `
  --print_examples 5
```

### 4. Summaries

Writes `results/final_base_summary.txt` and `results/final_lora_summary.txt`.

```powershell
$summaryScript = @'
import json
from pathlib import Path

def write_summary(src_name, title):
    p = Path(src_name)
    d = json.loads(p.read_text(encoding="utf-8"))
    lines = []
    lines.append(title)
    lines.append(f"n = {d['quality']['n']}")
    lines.append(f"mean_token_accuracy = {d['quality']['mean_token_accuracy']:.6f}")
    lines.append(f"mean_normalized_edit_distance = {d['quality']['mean_normalized_edit_distance']:.6f}")
    lines.append(f"exact_match_rate = {d['quality']['exact_match_rate']:.6f}")
    lines.append(f"mean_edit_distance = {d['quality']['mean_edit_distance']:.3f}")
    lines.append("")
    lines.append("Confidence:")
    for k, v in d["confidence"].items():
        lines.append(f"  {k}: {v:.6f}")
    lines.append("")
    lines.append("Latency (ms):")
    for k, v in d["latency_ms"].items():
        lines.append(f"  {k}: {v:.3f}")
    lines.append("")
    lines.append("UI mode rates:")
    for k, v in d["ui_mode_rates"].items():
        lines.append(f"  {k}: {v:.6f}")
    out = p.with_name(p.stem + "_summary.txt")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out)

write_summary("results/final_base.json", "=== Results (Base) ===")
write_summary("results/final_lora.json", "=== Results (LoRA) ===")
'@
$summaryScript | python -
```