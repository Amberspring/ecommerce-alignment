import pytest

pytest.importorskip("fastapi")
pytest.importorskip("transformers")
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from ecom import cpu_api


def test_cpu_generation_excludes_unsupported_token_type_ids(monkeypatch):
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "refund": 3}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=None,
        )
    ).eval()
    monkeypatch.setattr(cpu_api, "model", model)
    monkeypatch.setattr(cpu_api, "tokenizer", tok)
    result = cpu_api.completion(
        cpu_api.Completion(
            messages=[cpu_api.Message(role="user", content="refund")], max_tokens=2
        )
    )
    assert result["usage"]["prompt_tokens"] == 1
    assert (
        result["usage"]["completion_tokens"] >= 1
        and result["scope"] == "random_tiny_qwen3_cpu_integration_only"
    )
