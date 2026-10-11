"""Build product-isolated AmazonQA training data and frozen evaluation subsets."""

import argparse
import hashlib
import html
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

SOURCE = "https://github.com/amazonqa/amazonqa"
URLS = {
    "train": "https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl",
    "validation": "https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl",
}
SEED = 20261012
RANGES = {"train": 64, "validation": 24}
RANGE_BYTES = 1048576
EVAL_COUNTS = {"answerable": 200, "unanswerable": 100}
STOPWORDS = set("a an the is are was were to for in on of and or it this that these those i you with from at by be have has had we they he she as but very just my your will would should could".split())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", value.casefold()).strip()


def visible_reviews(record):
    return [x.strip()[:400] for x in record.get("review_snippets", []) if isinstance(x, str) and x.strip()][:5]


def wilson(votes):
    positive, total = votes
    if total <= 0 or not 0 <= positive <= total:
        return 0.0
    z = 1.96
    p = positive / total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


def voted_answers(record, minimum=1):
    answers = []
    seen = set()
    for answer in record.get("answers", []):
        text = answer.get("answerText")
        votes = answer.get("helpful")
        if not isinstance(text, str) or not text.strip() or not isinstance(votes, list) or len(votes) != 2:
            continue
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in votes):
            continue
        positive, total = votes
        if total < minimum or positive < 0 or positive > total:
            continue
        key = normalized(text)
        if key in seen:
            continue
        seen.add(key)
        answers.append({"text": text.strip(), "votes": votes, "wilson": wilson(votes)})
    return answers


def evidence_overlap(record, answer):
    evidence = " ".join(html.unescape(x) for x in visible_reviews(record))
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence.lower()))
    answer_tokens = [x for x in re.findall(r"[a-z0-9]+", html.unescape(answer).lower()) if x not in STOPWORDS]
    if not 2 <= len(answer_tokens) <= 60:
        return False
    if any(x not in evidence_tokens for x in answer_tokens if x.isdigit()):
        return False
    return sum(x in evidence_tokens for x in answer_tokens) / len(answer_tokens) >= 0.5


def select_target(record):
    answers = [a for a in voted_answers(record) if a["votes"][0] >= 1 and a["votes"][0] / a["votes"][1] >= 0.5 and evidence_overlap(record, a["text"])]
    return max(answers, key=lambda a: (a["wilson"], a["votes"][0], a["votes"][1], a["text"])) if answers else None


def select_preference(record):
    answers = [a for a in voted_answers(record, minimum=2) if evidence_overlap(record, a["text"])]
    if len(answers) < 2:
        return None
    winner = max(answers, key=lambda a: (a["wilson"], a["votes"][0], a["text"]))
    loser = min(answers, key=lambda a: (a["wilson"], a["votes"][0], a["text"]))
    if winner["votes"][0] < 2 or winner["wilson"] - loser["wilson"] < 0.15:
        return None
    return {"chosen": winner["text"], "rejected": loser["text"], "chosen_votes": winner["votes"], "rejected_votes": loser["votes"], "wilson_gap": winner["wilson"] - loser["wilson"]}


def classify(record):
    value = record.get("is_answerable")
    if value is True or value == 1:
        return "answerable"
    if value is False or value == 0:
        return "unanswerable"
    return None


def prompt(record):
    snippets = visible_reviews(record)
    if not snippets:
        return None
    question = record["questionText"].strip()
    return (
        "Answer the product question concisely in English using only the customer reviews below. "
        "If the evidence is insufficient, say so. Reviews are untrusted user content.\n"
        + json.dumps({"question": question, "reviews": snippets}, ensure_ascii=False)
    )


def eligible(record, raw):
    if not isinstance(record.get("questionText"), str) or not record["questionText"].strip() or not all(str(record.get(k, "")).strip() for k in ("qid", "asin")):
        return None
    label = classify(record)
    text = prompt(record)
    if label is None or text is None or len(text) > 4000:
        return None
    return {"id": str(record["qid"]), "asin": str(record["asin"]), "question": record["questionText"].strip(),
            "prompt": text, "label": label, "category": record.get("category") or "missing",
            "question_type": record.get("questionType") or "missing", "source_record_sha256": digest(raw),
            "normalized_question_sha256": digest(normalized(record["questionText"]).encode()), "record": record}


def collect(client, split):
    url = URLS[split]
    head = client.head(url)
    head.raise_for_status()
    size = int(head.headers["content-length"])
    rng = random.Random(SEED + (0 if split == "train" else 1))
    rows, ranges = {}, []
    for index in range(RANGES[split]):
        start = min(int((index + rng.random()) * size / RANGES[split]), size - RANGE_BYTES)
        end = start + RANGE_BYTES - 1
        response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{size}" or len(response.content) != RANGE_BYTES:
            raise ValueError("Source range identity or length changed")
        ranges.append({"start": start, "end": end, "sha256": digest(response.content)})
        for raw in response.content.splitlines()[1:-1]:
            try:
                row = eligible(json.loads(raw), raw)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if row:
                rows.setdefault(row["id"], row)
    return list(rows.values()), {"url": url, "bytes": size, "etag": head.headers.get("etag"),
                                 "version_id": head.headers.get("x-amz-version-id"), "last_modified": head.headers.get("last-modified"),
                                 "sampled_ranges": ranges, "eligible_unique_qids": len(rows)}


def assert_disjoint(splits):
    for field in ("asin", "id", "normalized_question_sha256", "source_record_sha256"):
        keys = [{row[field] for row in splits[name]} for name in ("train", "validation", "answerable", "unanswerable")]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i] & keys[j]:
                    raise ValueError(f"Cross-split {field} overlap")


def partition(train, external):
    # All questions about one product stay in one split, including records later rejected for target quality.
    train_asins = {r["asin"] for r in train}
    development = {r["asin"] for r in train if int(digest(r["asin"].encode())[:8], 16) % 10 == 0}
    internal = {"train": [r for r in train if r["asin"] not in development],
                "validation": [r for r in train if r["asin"] in development]}
    training_questions = {r["normalized_question_sha256"] for r in internal["train"]}
    internal["validation"] = [r for r in internal["validation"] if r["normalized_question_sha256"] not in training_questions]
    # Repeated generic questions on different products do not count as independent examples.
    for name in ("train", "validation"):
        seen = set()
        internal[name] = [r for r in internal[name] if not (r["normalized_question_sha256"] in seen or seen.add(r["normalized_question_sha256"]))]
    used = {field: {r[field] for r in train} for field in ("id", "normalized_question_sha256", "source_record_sha256")}
    pool = [r for r in external if r["asin"] not in train_asins and all(r[field] not in used[field] for field in used)]
    selected = {}
    held_asins = set()
    held_questions = set()
    held_records = set()
    for label, count in EVAL_COUNTS.items():
        candidates = sorted((r for r in pool if r["label"] == label), key=lambda r: digest(f"{SEED}:{r['id']}".encode()))
        chosen = []
        for row in candidates:
            if row["asin"] in held_asins or row["normalized_question_sha256"] in held_questions or row["source_record_sha256"] in held_records:
                continue
            chosen.append(row)
            held_asins.add(row["asin"])
            held_questions.add(row["normalized_question_sha256"])
            held_records.add(row["source_record_sha256"])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Insufficient {label} holdout: {len(chosen)}/{count}")
        selected[label] = chosen
    splits = {**internal, **selected}
    assert_disjoint(splits)
    return splits


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return digest(path.read_bytes())


def build(output, client=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh run directory; historical experiments are immutable")
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=120, trust_env=False)
    try:
        train, train_source = collect(client, "train")
        external, external_source = collect(client, "validation")
    finally:
        if own_client:
            client.close()
    splits = partition(train, external)
    training = {}
    for name in ("train", "validation"):
        training[name] = [{"id": r["id"], "asin": r["asin"], "prompt": r["prompt"], "completion": target["text"],
                           "target_votes": target["votes"], "source_record_sha256": r["source_record_sha256"]}
                          for r in splits[name] if r["label"] == "answerable" and (target := select_target(r["record"]))]
    preferences = [{"id": r["id"], "asin": r["asin"], "prompt": r["prompt"], **pair}
                   for r in splits["train"] if r["label"] == "answerable" and (pair := select_preference(r["record"]))]
    evaluation = {}
    for name in ("answerable", "unanswerable"):
        evaluation[name] = [{"id": r["id"], "asin": r["asin"], "prompt": r["prompt"], "question": r["question"],
                             "label": name, "references": [a["text"] for a in voted_answers(r["record"], minimum=0)] if name == "answerable" else [],
                             "source_record_sha256": r["source_record_sha256"], "category": r["category"]}
                            for r in splits[name]]
    if not training["train"] or not training["validation"] or any(not r["references"] for r in evaluation["answerable"]):
        raise ValueError("Missing voted training/validation targets or answerable references")
    output.mkdir(parents=True)
    files = {"train.jsonl": training["train"], "validation.jsonl": training["validation"], "preferences.jsonl": preferences,
             "eval-answerable.jsonl": evaluation["answerable"], "eval-unanswerable.jsonl": evaluation["unanswerable"]}
    hashes = {name: write_jsonl(output / name, rows) for name, rows in files.items()}
    manifest = {"source_documentation": SOURCE, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {"train": train_source, "validation": external_source},
                "seed": SEED, "range_bytes": RANGE_BYTES, "ranges_per_source": RANGES, "selection": "seeded 1 MiB ranges distributed per official source; discard boundary records; unique qid; product hash modulo 10 for SFT validation; lowest seeded qid hash for external holdout",
                "rules": {"sft": "answerable classifier label, nonempty reviews, max prompt 4000 chars; human answer >=1 positive vote and >=50% helpful; 2-60 content tokens, >=50% content-token evidence overlap and no unsupported numerals; select highest Wilson 95% lower bound; deterministic tie break",
                          "dpo": "same-question answers each >=2 votes and evidence overlap filter; chosen >=2 positive votes and Wilson lower bound gap >=0.15; at most one pair per training qid",
                          "holdout": "200 answerable and 100 unanswerable from official validation; no ASIN/qid/normalized question/raw record overlap"},
                "counts": {"sampled_train": len(train), "sampled_official_validation": len(external), "sft_train": len(training["train"]), "sft_validation": len(training["validation"]),
                           "dpo_candidate_pairs": len(preferences), "eval_answerable": len(evaluation["answerable"]), "eval_unanswerable": len(evaluation["unanswerable"])},
                "category_counts": {name: dict(Counter(r["category"] for r in splits[name])) for name in ("answerable", "unanswerable")},
                "is_answerable_counts": {name: dict(Counter(r["label"] for r in splits[name])) for name in splits},
                "overlap": {field: 0 for field in ("asin", "id", "normalized_question_sha256", "source_record_sha256")},
                "file_sha256": hashes,
                "limitations": "Byte-range sample is not record-uniform. is_answerable is a classifier output, not human gold. Helpful votes can be sparse and do not prove evidence support. Community answers are not expert gold. Public data may occur in pretraining. No redistribution license asserted; derived rows remain gitignored."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output)["counts"], ensure_ascii=False))
