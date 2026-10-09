"""Same held-out prompts and generation settings for Base/SFT/DPO endpoints."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from ecom.data import read, write
from ecom.evaluate import score, score_references


def run(dataset, endpoints, output):
    rows = read(dataset)
    if not rows:
        raise ValueError("Empty evaluation dataset")
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
        report["models"][label] = {**scorer(rows, predictions), "model": endpoint["model"],
                                    "predictions_sha256": hashlib.sha256((output / f"{label}.jsonl").read_bytes()).hexdigest()}
    report.update(status="measured", warning="Reference lexical overlap on public human product QA; not factual accuracy, preference or Chinese policy performance." if "references" in rows[0] else "Keyword rubric on synthetic policies; not human preference, hallucination rate or production accuracy.")
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/ecommerce/eval.jsonl")
    p.add_argument("--endpoints", required=True, help="JSON mapping base/sft/dpo to url, model, optional key_env")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.dataset, json.loads(Path(a.endpoints).read_text(encoding="utf-8")), a.output), ensure_ascii=False))
