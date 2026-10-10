import importlib.util
import json
from pathlib import Path

import httpx


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureClient:
    def __init__(self, raw):
        self.raw = raw

    def head(self, url):
        return httpx.Response(200, headers={"content-length": "10000", "etag": "fixture"}, request=httpx.Request("HEAD", url))

    def get(self, url, headers):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        size = end - start + 1
        return httpx.Response(206, content=self.raw[:size], headers={"content-range": f"bytes {start}-{end}/10000"}, request=httpx.Request("GET", url))


def test_distributed_selection_preserves_sources_and_fingerprints(tmp_path):
    module = _load_script("prepare_amazonqa.py")
    lines = [json.dumps({
        "qid": index, "asin": "fixture", "questionText": f"question {index}?",
        "questionType": "descriptive", "category": "fixture",
        "review_snippets": ["evidence"], "answers": [{"answerText": "original answer"}],
    }) for index in range(20)]
    body = ("partial\n" + "\n".join(lines) + "\ntrailing").encode()
    raw = body + b" " * (4096 - len(body))
    manifest = module.build(tmp_path / "audit", sample_size=10, range_count=2, range_bytes=4096, client=FixtureClient(raw))
    rows = [json.loads(line) for line in (tmp_path / "audit/eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10 and rows[0]["references"] == ["original answer"]
    assert manifest["range_count"] == 2 and len(manifest["range_fingerprints"]) == 2
    assert len(manifest["dataset_sha256"]) == 64
