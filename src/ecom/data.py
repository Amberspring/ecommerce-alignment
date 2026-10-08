import argparse, hashlib, json, re, random
from pathlib import Path


def read(path):
    return [
        json.loads(s)
        for s in Path(path).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def write(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def norm(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", s)
    s = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[EMAIL]", s)
    return s


def prepare(rows, seed=42):
    clean, rejected, seen = [], [], set()
    for i, r in enumerate(rows):
        q, a = norm(r.get("prompt", "")), norm(r.get("completion", ""))
        key = hashlib.sha256(q.casefold().encode()).hexdigest()
        reason = (
            "empty" if not q or not a else ("duplicate_prompt" if key in seen else None)
        )
        if len(q) + len(a) > 12000:
            reason = "too_long"
        if reason:
            rejected.append({"row": i, "reason": reason})
            continue
        seen.add(key)
        clean.append({**r, "prompt": q, "completion": a, "group_id": r.get("policy_id", key)})
    groups = list(dict.fromkeys(r["group_id"] for r in clean))
    random.Random(seed).shuffle(groups)
    n = len(groups)
    ntest = max(1, n // 5) if n >= 5 else 0
    nval = max(1, n // 5) if n >= 5 else 0
    return {
        "train": [r for r in clean if r["group_id"] in groups[: n - ntest - nval]],
        "validation": [r for r in clean if r["group_id"] in groups[n - ntest - nval : n - ntest]],
        "test": [r for r in clean if r["group_id"] in groups[n - ntest :]] if ntest else [],
        "rejected": rejected,
    }


def validate_preferences(rows, allowed_prompts=None):
    out = []
    for r in rows:
        p, c, n = (norm(r.get(k, "")) for k in ("prompt", "chosen", "rejected"))
        if not all((p, c, n)) or c == n:
            raise ValueError("Invalid preference pair")
        if allowed_prompts is not None and p not in allowed_prompts:
            continue  # Keep preference training out of held-out prompt groups.
        out.append({"prompt": p, "chosen": c, "rejected": n})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw.jsonl")
    p.add_argument("--output", default="artifacts/data")
    a = p.parse_args()
    splits = prepare(read(a.input))
    for k, v in splits.items():
        write(Path(a.output) / (k + ".jsonl"), v)
    pairs = validate_preferences(
        read("data/preferences.jsonl"), {r["prompt"] for r in splits["train"]}
    )
    write(Path(a.output) / "preferences.jsonl", pairs)
    report = {
        "status": "measured",
        "input_sha256": hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        "counts": {k: len(v) for k, v in splits.items()},
        "preference_train": len(pairs),
        "seed": 42,
        "limitations": "Synthetic fixtures; no production-data claim. Regex redaction is not exhaustive PII detection.",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/data-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
