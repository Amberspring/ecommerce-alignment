import importlib.util
from pathlib import Path
import json
import httpx
import pytest
from ecom.evaluate import score


def test_rubric_empty_duplicate_numeric_and_truncated():
    with pytest.raises(ValueError):
        score([], [])
    row = {"id": "a", "must_include": ["7"]}
    with pytest.raises(ValueError):
        score([row, row], [{"id": "a", "answer": "7"}])
    assert score([row], [{"id": "a", "answer": "17天"}])["rubric_pass_rate"] == 0
    assert score([row], [{"id": "a", "answer": "7天", "finish_reason": "length"}])["rubric_pass_rate"] == 0


def test_endpoint_comparison_records_real_response(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("comparison", Path(__file__).resolve().parents[1] / "scripts/compare_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_client = httpx.Client
    def handle(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0 and body["chat_template_kwargs"]["enable_thinking"] is False
        return httpx.Response(200, json={"model": body["model"], "choices": [{"message": {"content": "7天"}, "finish_reason": "stop"}], "usage": {"total_tokens": 10}})
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"id": "a", "prompt": "question", "must_include": ["7"]}), encoding="utf-8")
    result = module.run(dataset, {label: {"url": "http://fixture/v1", "model": label} for label in ("base", "sft", "dpo")}, tmp_path / "results")
    assert all(r["rubric_pass_rate"] == 1 for r in result["models"].values())
    assert (tmp_path / "results/comparison.json").exists()
