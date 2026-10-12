"""Prepare and score human-blinded Base/SFT(/DPO) reviews without model judges."""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


LABELS = ("base", "sft", "dpo")
SCORES = ("correctness", "support", "helpfulness")
UNANSWERABLE_SCORES = ("correct_refusal", "insufficiency_recognition", "unsupported_assertion", "helpfulness")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(dataset, predictions, review, key, seed=20261010, labels=LABELS):
    rows = _read_jsonl(dataset)
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Dataset must have nonempty unique IDs")
    subsets = {row.get("label", "answerable") for row in rows}
    if len(subsets) != 1 or not subsets <= {"answerable", "unanswerable"}:
        raise ValueError("Review one answerable or unanswerable subset at a time")
    subset = subsets.pop()
    metrics = UNANSWERABLE_SCORES if subset == "unanswerable" else SCORES
    by_model = {}
    labels = tuple(labels)
    if labels not in (("base", "sft"), LABELS):
        raise ValueError("Blind review requires base/sft, optionally followed by dpo")
    letters = "abc"[:len(labels)]
    for label in labels:
        model_rows = _read_jsonl(Path(predictions) / f"{label}.jsonl")
        by_model[label] = {row["id"]: row for row in model_rows}
        if len(by_model[label]) != len(model_rows) or set(by_model[label]) != set(ids):
            raise ValueError(f"{label} prediction IDs must match the dataset exactly")
    review, key = Path(review), Path(key)
    if review.exists() or key.exists():
        raise FileExistsError("Review and key outputs must be new files")
    review.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "question", "evidence"] + [f"answer_{letter}" for letter in letters] + ["reviewer_id"]
    if subset == "unanswerable":
        fields += ["evidence_sufficient"]
    fields += [f"{metric}_{letter}" for letter in letters for metric in metrics]
    fields += ["best", "notes"]
    key_rows = []
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            order = sorted(labels, key=lambda label: hashlib.sha256(f"{seed}:{row['id']}:{label}".encode()).digest())
            evidence = row.get("evidence")
            if evidence is None:
                evidence = json.loads(row["prompt"].split("\n", 1)[1])["reviews"]
            item = {"item_id": row["id"], "question": row.get("question") or row["prompt"], "evidence": json.dumps(evidence, ensure_ascii=False)}
            mapping = {}
            for letter, label in zip(letters, order):
                item[f"answer_{letter}"] = by_model[label][row["id"]]["answer"]
                mapping[letter.upper()] = label
            writer.writerow(item)
            visible_sha = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            key_rows.append({"item_id": row["id"], "mapping": mapping, "visible_sha256": visible_sha})
    payload = {"seed": seed, "labels": list(labels), "subset": subset,
               "template_sha256": hashlib.sha256(review.read_bytes()).hexdigest(), "items": key_rows}
    key.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return payload


def score(reviews, key, output):
    key_data = json.loads(Path(key).read_text(encoding="utf-8"))
    mappings = {row["item_id"]: row["mapping"] for row in key_data["items"]}
    visible_hashes = {row["item_id"]: row.get("visible_sha256") for row in key_data["items"]}
    labels = tuple(key_data["labels"])
    subset = key_data.get("subset", "answerable")
    metrics = UNANSWERABLE_SCORES if subset == "unanswerable" else SCORES
    letters = "abc"[:len(labels)]
    if not mappings:
        raise ValueError("Blind review key contains no items")
    totals = {label: {metric: [] for metric in metrics} | {"wins": 0} for label in labels}
    by_item = {label: {metric: defaultdict(list) for metric in metrics} for label in labels}
    sufficient = []
    sufficient_by_item = defaultdict(list)
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
            visible = {field: row[field] for field in ("item_id", "question", "evidence", *(f"answer_{letter}" for letter in letters))}
            visible_sha = hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if visible_hashes[row["item_id"]] and visible_sha != visible_hashes[row["item_id"]]:
                raise ValueError("Question, evidence, or answer text changed after blinding")
            if subset == "unanswerable":
                evidence_sufficient = row["evidence_sufficient"].strip()
                if evidence_sufficient not in {"0", "1"}:
                    raise ValueError("evidence_sufficient must be 0 or 1")
                sufficient.append(int(evidence_sufficient))
                sufficient_by_item[row["item_id"]].append(int(evidence_sufficient))
            for letter in letters:
                label = mappings[row["item_id"]][letter.upper()]
                for metric in metrics:
                    value = row[f"{metric}_{letter}"].strip()
                    allowed = {"0", "1", "2"} if metric == "helpfulness" or subset == "answerable" else {"0", "1"}
                    if value not in allowed:
                        raise ValueError(f"{metric}_{letter} must be one of {sorted(allowed)}")
                    if subset == "unanswerable" and evidence_sufficient == "1" and metric in {"correct_refusal", "insufficiency_recognition"} and value != "0":
                        raise ValueError(f"{metric}_{letter} cannot be positive when the evidence is sufficient")
                    totals[label][metric].append(int(value))
                    by_item[label][metric][row["item_id"]].append(int(value))
                if subset == "unanswerable" and row[f"correct_refusal_{letter}"] == "1" and (
                    row[f"insufficiency_recognition_{letter}"] != "1" or row[f"unsupported_assertion_{letter}"] != "0"
                ):
                    raise ValueError("Correct refusal requires insufficiency recognition and no unsupported assertion")
            best = row["best"].strip().upper()
            if best not in {*(letter.upper() for letter in letters), "TIE"}:
                raise ValueError("best must name an answer letter or TIE")
            winner = None if best == "TIE" else mappings[row["item_id"]][best]
            if winner:
                totals[winner]["wins"] += 1
            decisions.append({"reviewer": reviewer, "item_id": row["item_id"], "winner": winner})
    paired = {}
    rng = random.Random(20261010)
    for left, right in (("base", "sft"), ("base", "dpo"), ("sft", "dpo")):
        if left not in labels or right not in labels:
            continue
        paired[f"{right}_minus_{left}"] = {}
        for metric in (*metrics, *((f"{metric}_rate_2" for metric in metrics) if subset == "answerable" else ())):
            rating = metric.removesuffix("_rate_2")
            def item_mean(label, item):
                values = by_item[label][rating][item]
                numerator = sum(value == 2 for value in values) if metric.endswith("_rate_2") else sum(values)
                return numerator / len(values)
            deltas = [item_mean(right, item) - item_mean(left, item) for item in sorted(mappings)]
            boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(5000))
            paired[f"{right}_minus_{left}"][metric] = {
                "mean_delta": sum(deltas) / len(deltas),
                "bootstrap_95_ci": [boot[125], boot[4875]],
            }
    report = {
        "status": "reviewer_entries_recorded", "subset": subset, "reviewers": sorted(reviewers),
        "review_files": [str(Path(path)) for path in reviews],
        "review_file_sha256": {str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in reviews},
        "ratings": {label: {**{metric: sum(totals[label][metric]) / len(totals[label][metric]) for metric in metrics},
                            **({f"{metric}_rate_2": sum(value == 2 for value in totals[label][metric]) / len(totals[label][metric]) for metric in metrics} if subset == "answerable" else {}),
                            "best_win_rate": totals[label]["wins"] / len(decisions)} for label in labels},
        "paired_rating_deltas": paired,
        "reviewer_evidence_sufficient_fraction": sum(sufficient) / len(sufficient) if sufficient else None,
        "cases": [{"item_id": item,
                   "reviewer_evidence_sufficient_fraction": sum(sufficient_by_item[item]) / len(sufficient_by_item[item]) if subset == "unanswerable" else None,
                   "models": {label: {metric: sum(by_item[label][metric][item]) / len(by_item[label][metric][item]) for metric in metrics} for label in labels}}
                  for item in sorted(mappings)],
        "tie_rate": sum(decision["winner"] is None for decision in decisions) / len(decisions),
        "decision_count": len(decisions),
        "limitations": "Reviewer IDs and entries are self-reported; the script cannot prove human participation or independence. AmazonQA unanswerable labels are classifier outputs, and reviewers may judge visible evidence sufficient.",
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output path to preserve experiment history")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
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
    prep.add_argument("--labels", default="base,sft,dpo")
    aggregate = sub.add_parser("score")
    aggregate.add_argument("--review", action="append", required=True)
    aggregate.add_argument("--key", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.dataset, args.predictions, args.review, args.key, args.seed, args.labels.split(",")) if args.command == "prepare" else score(args.review, args.key, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
