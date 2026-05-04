# Backend — AI Code Completion API

## Quick Start

```bash
cd backend
pip install -r requirements.txt

# Default model is Qwen/Qwen2.5-Coder-0.5B (small, runs on CPU).
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
| `inline` | ≥ 0.80 | Show as ghost text |
| `collapsed` | 0.40 – 0.79 | Show in expandable panel |
| `hidden` | < 0.40 | Suppress entirely |

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true }`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL_NAME` | `Qwen/Qwen2.5-Coder-0.5B` | HuggingFace model ID |
| `LORA_ADAPTER_PATH` | *(empty)* | Path to LoRA adapter dir. Leave empty to use base model. |

## Confidence Score Formula

```
confidence = exp( mean( log P(token_i) for each generated token_i ) )
```

This converts mean log-probability back to a [0, 1] probability value.
