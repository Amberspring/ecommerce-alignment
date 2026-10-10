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
    for letter in "abc":
        for metric in module.SCORES:
            row[f"{metric}_{letter}"] = "2"
    row["best"] = "TIE"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    report = module.score([review], key, tmp_path / "report.json")
    assert report["status"] == "human_reviewed" and report["tie_rate"] == 1
    with pytest.raises(ValueError, match="only once"):
        module.score([review, review], key, tmp_path / "duplicate.json")
