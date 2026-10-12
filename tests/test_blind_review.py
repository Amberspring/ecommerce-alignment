import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/blind_review.py"
    spec = importlib.util.spec_from_file_location("blind_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blind_review_requires_real_completed_labels(tmp_path):
    module = _module()
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"id": "1", "question": "q", "evidence": ["e"], "prompt": "p"}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for label in module.LABELS:
        (predictions / f"{label}.jsonl").write_text(json.dumps({"id": "1", "answer": label}) + "\n", encoding="utf-8")
    review, key = tmp_path / "review.csv", tmp_path / "key.json"
    module.prepare(dataset, predictions, review, key)
    with pytest.raises(ValueError, match="reviewer_id"):
        module.score([review], key, tmp_path / "invalid.json")
    with review.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    row = rows[0]
    row["reviewer_id"] = "human-1"
    mapping = json.loads(key.read_text(encoding="utf-8"))["items"][0]["mapping"]
    for letter in "abc":
        for metric in module.SCORES:
            row[f"{metric}_{letter}"] = "0" if metric == "support" and mapping[letter.upper()] == "base" else "2"
    row["best"] = "TIE"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    report = module.score([review], key, tmp_path / "report.json")
    assert report["status"] == "reviewer_entries_recorded" and report["tie_rate"] == 1
    assert report["paired_rating_deltas"]["sft_minus_base"]["support_rate_2"]["bootstrap_95_ci"] == [1, 1]
    with pytest.raises(ValueError, match="only once"):
        module.score([review, review], key, tmp_path / "duplicate.json")


def test_two_model_review_includes_visible_comment_evidence(tmp_path):
    module = _module()
    dataset = tmp_path / "eval.jsonl"
    prompt = 'Answer using reviews.\n{"question":"q","reviews":["visible review"]}'
    dataset.write_text(json.dumps({"id": "1", "question": "q", "prompt": prompt}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for label in ("base", "sft"):
        (predictions / f"{label}.jsonl").write_text(json.dumps({"id": "1", "answer": label}) + "\n", encoding="utf-8")
    review, key = tmp_path / "review.csv", tmp_path / "key.json"
    module.prepare(dataset, predictions, review, key, labels=("base", "sft"))
    with review.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "visible review" in row["evidence"]
    assert "answer_c" not in row
    assert json.loads(key.read_text(encoding="utf-8"))["labels"] == ["base", "sft"]


def test_unanswerable_review_uses_separate_rubric_and_rejects_changed_evidence(tmp_path):
    module = _module()
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"id": "1", "label": "unanswerable", "question": "q", "evidence": ["e"]}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for label in ("base", "sft"):
        (predictions / f"{label}.jsonl").write_text(json.dumps({"id": "1", "answer": label}) + "\n", encoding="utf-8")
    review, key = tmp_path / "review.csv", tmp_path / "key.json"
    module.prepare(dataset, predictions, review, key, labels=("base", "sft"))
    with review.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, row = reader.fieldnames, next(reader)
    assert "evidence_sufficient" in fields and "correctness_a" not in fields
    row.update(reviewer_id="reviewer-1", evidence_sufficient="0", best="TIE")
    for letter in "ab":
        row.update({f"correct_refusal_{letter}": "1", f"insufficiency_recognition_{letter}": "1",
                    f"unsupported_assertion_{letter}": "0", f"helpfulness_{letter}": "2"})
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    report = module.score([review], key, tmp_path / "report.json")
    assert report["subset"] == "unanswerable"
    assert report["reviewer_evidence_sufficient_fraction"] == 0
    assert report["paired_rating_deltas"]["sft_minus_base"]["unsupported_assertion"]["bootstrap_95_ci"] == [0, 0]
    row["evidence"] = "changed"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="changed after blinding"):
        module.score([review], key, tmp_path / "invalid.json")
