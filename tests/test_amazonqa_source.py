import importlib.util
import json
import httpx


def test_fixed_range_selection_keeps_original_answers_and_fingerprints(tmp_path, monkeypatch):
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_amazonqa.py"
    spec = importlib.util.spec_from_file_location("prepare_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/prepare.py"))
    lines = [json.dumps({"qid": i, "asin": "fixture", "questionText": "fixture?", "review_snippets": ["x" * 500], "answers": [{"answerText": "original fixture answer"}]}) for i in range(20)]
    raw = ("\n".join(lines) + "\n").encode()
    raw += b" " * (1048576 - len(raw))
    def get(url, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=0-1048575"
        return httpx.Response(206, content=raw, headers={"content-range": "bytes 0-1048575/9999999"}, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)
    module.main()
    target = tmp_path / "artifacts/amazonqa-audit"
    rows = [json.loads(s) for s in (target / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20 and rows[0]["references"] == ["original fixture answer"]
    assert len(json.loads(rows[0]["prompt"].split("\n", 1)[1])["reviews"][0]) == 400
    assert len(json.loads((target / "manifest.json").read_text())["source_prefix_sha256"]) == 64
