import argparse, json, hashlib, re
from pathlib import Path


def score(rows, predictions):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Evaluation must have nonempty, unique IDs")
    by = {p["id"]: p["answer"] for p in predictions}
    truncated = {p["id"] for p in predictions if p.get("finish_reason") == "length"}
    if len(by) != len(predictions) or set(by) != {r["id"] for r in rows}:
        raise ValueError("Prediction IDs must match evaluation IDs exactly")
    cases = []
    for r in rows:
        answer = by[r["id"]]
        if not isinstance(answer, str):
            raise ValueError("Prediction answers must be strings")
        def contains(term):
            return bool(re.search(r"(?<!\d)" + re.escape(term) + r"(?!\d)", answer)) if term.isdigit() else term in answer

        ok = all(contains(x) for x in r["must_include"]) and not any(
            contains(x) for x in r.get("must_not_include", [])
        ) and r["id"] not in truncated
        reasons = []
        if not ok:
            reasons.append("policy_or_fact")
        if re.search(r"(.{4,})\1\1", answer):
            reasons.append("repetition")
        cases.append(
            {"id": r["id"], "pass": ok, "categories": reasons, "answer": answer}
        )
    return {
        "rubric_pass_rate": sum(c["pass"] for c in cases) / len(cases),
        "count": len(cases),
        "badcases": [c for c in cases if c["categories"]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/eval.jsonl")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    load = lambda path: [
        json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    result = score(load(a.dataset), load(a.predictions))
    result.update(
        status="measured",
        dataset_sha256=hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
        prediction_sha256=hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest(),
        metric_warning="Keyword rubric is not human preference win rate or general accuracy.",
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
