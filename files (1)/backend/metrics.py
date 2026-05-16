import re


def _normalize_text(s: str, *, strip: bool = True, collapse_whitespace: bool = False) -> str:
    if strip:
        s = s.strip()
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s)
    return s


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Two-row DP to save memory
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(
                min(
                    cur[j - 1] + 1,      # insert
                    prev[j] + 1,        # delete
                    prev[j - 1] + cost, # substitute
                )
            )
        prev = cur
    return prev[-1]


def normalized_edit_distance(prediction: str,reference: str,*,strip: bool = True,) -> float:

    p = _normalize_text(prediction, strip=strip)
    r = _normalize_text(reference, strip=strip)
    d = levenshtein_distance(p, r)
    denom = max(len(p), len(r), 1)
    return d / denom


def token_level_accuracy(prediction: str,reference: str,*,strip: bool = True,) -> float:
    p = _normalize_text(prediction, strip=strip)
    r = _normalize_text(reference, strip=strip)
    pred_tokens = p.split()
    ref_tokens = r.split()
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    hits = sum(
        1
        for i, rt in enumerate(ref_tokens)
        if i < len(pred_tokens) and pred_tokens[i] == rt
    )
    return hits / len(ref_tokens)


def exact_match(prediction: str, reference: str, *, strip: bool = True) -> bool:
    p = _normalize_text(prediction, strip=strip)
    r = _normalize_text(reference, strip=strip)
    return p == r


def evaluate_completion(completion: str,ground_truth_suffix: str,*,strip: bool = True,) -> dict:

    p = _normalize_text(completion, strip=strip)
    r = _normalize_text(ground_truth_suffix, strip=strip)
    ed = levenshtein_distance(p, r)
    denom = max(len(p), len(r), 1)
    return {
        "token_accuracy": token_level_accuracy(p, r, strip=False),
        "edit_distance": ed,
        "normalized_edit_distance": ed / denom,
        "exact_match": p == r,
    }


def aggregate_metrics(rows: list[dict]) -> dict:

    if not rows:
        return {
            "n": 0,
            "mean_token_accuracy": 0.0,
            "mean_normalized_edit_distance": 0.0,
            "exact_match_rate": 0.0,
            "mean_edit_distance": 0.0,
        }
    n = len(rows)
    return {
        "n": n,
        "mean_token_accuracy": sum(r["token_accuracy"] for r in rows) / n,
        "mean_normalized_edit_distance": sum(r["normalized_edit_distance"] for r in rows) / n,
        "exact_match_rate": sum(1 for r in rows if r["exact_match"]) / n,
        "mean_edit_distance": sum(r["edit_distance"] for r in rows) / n,
    }
