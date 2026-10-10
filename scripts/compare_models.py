"""Same held-out prompts and generation settings for Base/SFT/DPO endpoints."""
import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import httpx
from ecom.data import read, write
from ecom.evaluate import score, score_references


def _generation_summary(predictions):
    latencies = [row["latency_ms"] for row in predictions]
    completion_tokens = [row["usage"].get("completion_tokens") for row in predictions if isinstance(row.get("usage"), dict) and isinstance(row["usage"].get("completion_tokens"), int)]
    return {
        "truncated_count": sum(row.get("finish_reason") == "length" for row in predictions),
        "median_latency_ms": statistics.median(latencies),
        "mean_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
    }


def _paired_deltas(report, seed=20261010, samples=5000):
    rng = random.Random(seed)
    result = {}
    for left, right in (("base", "sft"), ("base", "dpo"), ("sft", "dpo")):
        if left not in report["models"] or right not in report["models"]:
            continue
        a = {row["id"]: row["token_f1"] for row in report["models"][left]["cases"]}
        b = {row["id"]: row["token_f1"] for row in report["models"][right]["cases"]}
        if set(a) != set(b):
            raise ValueError("Paired model results must contain identical IDs")
        deltas = [b[item] - a[item] for item in sorted(a)]
        boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
        result[f"{right}_minus_{left}"] = {
            "mean_token_f1_delta": sum(deltas) / len(deltas),
            "bootstrap_95_ci": [boot[int(samples * 0.025)], boot[int(samples * 0.975)]],
            "paired_case_win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        }
    return result


def run(dataset, endpoints, output):
    rows = read(dataset)
    if not rows:
        raise ValueError("Empty evaluation dataset")
    if not {"base", "sft", "dpo"}.issubset(endpoints):
        raise ValueError("A comparable run requires base, sft, and dpo endpoints")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory to preserve experiment history")
    output.mkdir(parents=True)
    report = {"status": "running", "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
              "generation": {"temperature": 0, "max_tokens": 256, "enable_thinking": False}, "models": {}}
    for label, endpoint in endpoints.items():
        if label not in ("base", "sft", "dpo", "grpo"):
            raise ValueError("Unknown experiment label")
        predictions = []
        key = os.getenv(endpoint.get("key_env", "UPSTREAM_API_KEY"), "")
        with httpx.Client(timeout=120, trust_env=False, headers={"Authorization": "Bearer " + key} if key else {}) as client:
            for row in rows:
                start = time.perf_counter()
                response = client.post(endpoint["url"].rstrip("/") + "/chat/completions", json={
                    "model": endpoint["model"], "messages": [{"role": "user", "content": row["prompt"]}],
                    "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}})
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                predictions.append({"id": row["id"], "answer": choice["message"]["content"],
                                    "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                                    "model": data.get("model"), "latency_ms": (time.perf_counter() - start) * 1000})
                write(output / f"{label}.jsonl", predictions)
        scorer = score_references if "references" in rows[0] else score
        report["models"][label] = {**scorer(rows, predictions), **_generation_summary(predictions), "model": endpoint["model"],
                                    "predictions_sha256": hashlib.sha256((output / f"{label}.jsonl").read_bytes()).hexdigest()}
    if "references" in rows[0]:
        report["paired_token_f1_deltas"] = _paired_deltas(report)
    report.update(status="measured", warning="Reference lexical overlap on public human product QA; not factual accuracy, preference or Chinese policy performance." if "references" in rows[0] else "Keyword rubric on synthetic policies; not human preference, hallucination rate or production accuracy.")
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/ecommerce/eval.jsonl")
    p.add_argument("--endpoints", required=True, help="JSON mapping base/sft/dpo to url, model, optional key_env")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.dataset, json.loads(Path(a.endpoints).read_text(encoding="utf-8")), a.output), ensure_ascii=False))
