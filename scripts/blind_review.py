"""Prepare and score human-blinded Base/SFT/DPO reviews without model judges."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


LABELS = ("base", "sft", "dpo")
SCORES = ("correctness", "support", "helpfulness")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(dataset, predictions, review, key, seed=20261010):
    rows = _read_jsonl(dataset)
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Dataset must have nonempty unique IDs")
    by_model = {}
    for label in LABELS:
        model_rows = _read_jsonl(Path(predictions) / f"{label}.jsonl")
        by_model[label] = {row["id"]: row for row in model_rows}
        if len(by_model[label]) != len(model_rows) or set(by_model[label]) != set(ids):
            raise ValueError(f"{label} prediction IDs must match the dataset exactly")
    review, key = Path(review), Path(key)
    if review.exists() or key.exists():
        raise FileExistsError("Review and key outputs must be new files")
    review.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "question", "evidence", "answer_a", "answer_b", "answer_c", "reviewer_id"]
    fields += [f"{metric}_{letter}" for letter in "abc" for metric in SCORES]
    fields += ["best", "notes"]
    key_rows = []
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            order = sorted(LABELS, key=lambda label: hashlib.sha256(f"{seed}:{row['id']}:{label}".encode()).digest())
            item = {"item_id": row["id"], "question": row.get("question") or row["prompt"], "evidence": json.dumps(row.get("evidence", []), ensure_ascii=False)}
            mapping = {}
            for letter, label in zip("abc", order):
                item[f"answer_{letter}"] = by_model[label][row["id"]]["answer"]
                mapping[letter.upper()] = label
            writer.writerow(item)
            key_rows.append({"item_id": row["id"], "mapping": mapping})
    payload = {"seed": seed, "labels": list(LABELS), "template_sha256": hashlib.sha256(review.read_bytes()).hexdigest(), "items": key_rows}
    key.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def score(reviews, key, output):
    key_data = json.loads(Path(key).read_text(encoding="utf-8"))
    mappings = {row["item_id"]: row["mapping"] for row in key_data["items"]}
    if not mappings:
        raise ValueError("Blind review key contains no items")
    totals = {label: {metric: [] for metric in SCORES} | {"wins": 0} for label in LABELS}
    reviewers, decisions = set(), []
    for review_path in reviews:
        with Path(review_path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if {row["item_id"] for row in rows} != set(mappings) or len(rows) != len(mappings):
            raise ValueError("Every review must contain every item exactly once")
        file_reviewers = {row["reviewer_id"].strip() for row in rows}
        if "" in file_reviewers or len(file_reviewers) != 1:
            raise ValueError("Each completed review file requires one nonempty reviewer_id")
        reviewer = next(iter(file_reviewers))
        if reviewer in reviewers:
            raise ValueError("Each reviewer_id may be counted only once")
        reviewers.add(reviewer)
        for row in rows:
            reviewer = row["reviewer_id"].strip()
            for letter in "abc":
                label = mappings[row["item_id"]][letter.upper()]
                for metric in SCORES:
                    value = row[f"{metric}_{letter}"].strip()
                    if value not in {"0", "1", "2"}:
                        raise ValueError(f"{metric}_{letter} must be 0, 1, or 2")
                    totals[label][metric].append(int(value))
            best = row["best"].strip().upper()
            if best not in {"A", "B", "C", "TIE"}:
                raise ValueError("best must be A, B, C, or TIE")
            winner = None if best == "TIE" else mappings[row["item_id"]][best]
            if winner:
                totals[winner]["wins"] += 1
            decisions.append({"reviewer": reviewer, "item_id": row["item_id"], "winner": winner})
    report = {
        "status": "human_reviewed", "reviewers": sorted(reviewers),
        "review_files": [str(Path(path)) for path in reviews],
        "ratings": {label: {**{metric: sum(totals[label][metric]) / len(totals[label][metric]) for metric in SCORES}, "best_win_rate": totals[label]["wins"] / len(decisions)} for label in LABELS},
        "tie_rate": sum(decision["winner"] is None for decision in decisions) / len(decisions),
        "decision_count": len(decisions),
        "limitations": "Reviewer IDs and entries are self-reported. This script validates completeness and blinding keys but cannot prove reviewer identity or independence.",
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output path to preserve experiment history")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--predictions", required=True)
    prep.add_argument("--review", required=True)
    prep.add_argument("--key", required=True)
    prep.add_argument("--seed", type=int, default=20261010)
    aggregate = sub.add_parser("score")
    aggregate.add_argument("--review", action="append", required=True)
    aggregate.add_argument("--key", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.dataset, args.predictions, args.review, args.key, args.seed) if args.command == "prepare" else score(args.review, args.key, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
