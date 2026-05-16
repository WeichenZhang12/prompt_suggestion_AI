import random
from typing import Any, Optional

from datasets import Dataset, load_dataset


def load_codesearchnet_raw(language: str = "python") -> dict[str, Dataset]:
    ds = load_dataset("code_search_net", language)
    return ds


def _example_to_prompt_completion(example: dict[str, Any],idx: int,*,seed: int,min_chars: int,frac_lo: float,frac_hi: float,) -> dict[str, str]:
    rng = random.Random(seed + idx)
    code = (example.get("func_code_string") or "").strip()
    if len(code) < min_chars:
        return {"prompt": "", "completion": ""}

    lo = max(1, int(len(code) * frac_lo))
    hi = int(len(code) * frac_hi)
    hi = min(hi, len(code) - 1)
    if hi <= lo + 1:
        return {"prompt": "", "completion": ""}

    split = rng.randint(lo, hi)
    prefix, suffix = code[:split], code[split:]
    if not suffix.strip():
        return {"prompt": "", "completion": ""}

    return {"prompt": prefix, "completion": suffix}


def build_prompt_completion_dataset(split: str,*,language: str = "python",seed: int = 42,min_chars: int = 40,frac_lo: float = 0.2,frac_hi: float = 0.8,max_samples: Optional[int] = None,num_proc: Optional[int] = None,) -> Dataset:
    raw = load_dataset("code_search_net", language, split=split)
    if max_samples is not None:
        n = min(max_samples, len(raw))
        raw = raw.select(range(n))

    kwargs: dict[str, Any] = {
        "function": lambda ex, i: _example_to_prompt_completion(
            ex,
            i,
            seed=seed,
            min_chars=min_chars,
            frac_lo=frac_lo,
            frac_hi=frac_hi,
        ),
        "with_indices": True,
        "remove_columns": raw.column_names,
    }
    if num_proc is not None:
        kwargs["num_proc"] = num_proc

    out: Dataset = raw.map(**kwargs)
    out = out.filter(lambda x: bool(x["prompt"]) and bool(x["completion"]))
    return out
