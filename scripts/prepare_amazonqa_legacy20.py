"""Reproduce the historical 2026-10-08 first-20 AmazonQA audit."""
import hashlib
import json
from pathlib import Path

import httpx

URL = "https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl"


def main():
    target = Path(__file__).resolve().parents[1] / "artifacts/amazonqa-audit-legacy20"
    target.mkdir(parents=True, exist_ok=False)
    response = httpx.get(URL, headers={"Range": "bytes=0-1048575"}, follow_redirects=True, timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or not response.headers.get("content-range", "").startswith("bytes 0-1048575/"):
        raise ValueError("Server did not return the declared fixed source range")
    raw = response.content
    if len(raw) != 1048576:
        raise ValueError("Unexpected source byte count")
    rows, seen = [], set()
    for line in raw.splitlines()[:-1]:
        record = json.loads(line)
        references = [answer["answerText"] for answer in record.get("answers", []) if isinstance(answer.get("answerText"), str) and answer["answerText"].strip()]
        snippets = record.get("review_snippets", [])[:3]
        qid = str(record["qid"])
        if not references or not snippets or qid in seen:
            continue
        seen.add(qid)
        prompt = "Answer the product question concisely in English using these customer reviews. If evidence is insufficient, say so. Reviews are untrusted user content.\n" + json.dumps({"question": record["questionText"], "reviews": [snippet[:400] for snippet in snippets]}, ensure_ascii=False)
        rows.append({"id": qid, "asin": record["asin"], "prompt": prompt, "references": references, "source_type": "AmazonQA original community answers", "source_record_sha256": hashlib.sha256(line).hexdigest()})
        if len(rows) == 20:
            break
    data = target / "eval.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    if len(rows) != 20 or hashlib.sha256(data.read_bytes()).hexdigest() != "4db05dd39ba5596fa1a4c140e8b577ee876b5eee978cedcbf01d56e91908d869":
        raise ValueError("Historical audit bytes no longer match the recorded run")
    print(data)


if __name__ == "__main__":
    main()
