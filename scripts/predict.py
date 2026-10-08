"""Collect actual model responses without producing or inventing evaluation scores."""

import argparse, json
from pathlib import Path
import httpx

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000/v1")
p.add_argument("--model", default="ecommerce-model")
p.add_argument("--dataset", default="data/eval.jsonl")
p.add_argument("--output", required=True)
a = p.parse_args()
rows = [json.loads(s) for s in Path(a.dataset).read_text(encoding="utf-8").splitlines()]
out = []
with httpx.Client(timeout=120, trust_env=False) as client:
    for r in rows:
        response = client.post(
            a.url.rstrip("/") + "/chat/completions",
            json={
                "model": a.model,
                "messages": [{"role": "user", "content": r["prompt"]}],
                "temperature": 0,
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        body = response.json()
        out.append(
            {
                "id": r["id"],
                "answer": body["choices"][0]["message"]["content"],
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
                "model": body.get("model"),
            }
        )
target = Path(a.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8"
)
