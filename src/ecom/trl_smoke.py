"""Offline random tiny Qwen3 + real TRL/PEFT integration test. No pretrained weights."""

import json
from pathlib import Path
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer


def run():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    words = [
        "<pad>",
        "<eos>",
        "<unk>",
        "refund",
        "delivery",
        "please",
        "check",
        "order",
        "policy",
        "guarantee",
        "immediate",
        "yes",
        "no",
        "user",
        "assistant",
    ]
    tokenizer = Tokenizer(
        WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    config = Qwen3Config(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=1,
        pad_token_id=0,
        sliding_window=None,
    )
    model = Qwen3ForCausalLM(config)
    model.config.use_cache = False
    lc = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lc)
    ds = Dataset.from_list(
        [
            {"prompt": p + " ", "completion": "please check order policy <eos>"}
            for p in ["refund", "delivery"]
        ]
        * 4
    )
    before = {
        k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad
    }
    kwargs = dict(
        use_cpu=True,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=8,
        learning_rate=0.01,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        gradient_checkpointing=False,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds,
        args=SFTConfig(
            output_dir="artifacts/trl-smoke/sft",
            max_length=32,
            completion_only_loss=True,
            **kwargs,
        ),
    )
    trainer.train()
    trainer.save_model()
    tok.save_pretrained("artifacts/trl-smoke/sft")
    sft_delta = sum(
        float((v.detach() - before[k]).norm())
        for k, v in model.named_parameters()
        if k in before
    )
    # Reload train/reference adapters exactly as the full GPU entrypoint does.
    # Recreate the identical frozen base by reseeding before construction.
    torch.manual_seed(42)
    base = Qwen3ForCausalLM(config)
    base.config.use_cache = False
    policy = PeftModel.from_pretrained(
        base, "artifacts/trl-smoke/sft", is_trainable=True, adapter_name="policy"
    )
    policy.load_adapter(
        "artifacts/trl-smoke/sft", adapter_name="reference", is_trainable=False
    )
    policy.set_adapter("policy")
    prefs = Dataset.from_list(
        [
            {
                "prompt": p + " ",
                "chosen": "please check order policy <eos>",
                "rejected": "guarantee immediate <eos>",
            }
            for p in ["refund", "delivery"]
        ]
        * 4
    )
    before = {
        k: v.detach().clone() for k, v in policy.named_parameters() if v.requires_grad
    }
    ref_before = {
        k: v.detach().clone()
        for k, v in policy.named_parameters()
        if ".reference." in k
    }
    dpo = DPOTrainer(
        model=policy,
        processing_class=tok,
        train_dataset=prefs,
        args=DPOConfig(
            output_dir="artifacts/trl-smoke/dpo",
            max_length=32,
            max_prompt_length=8,
            beta=0.1,
            model_adapter_name="policy",
            ref_adapter_name="reference",
            **kwargs,
        ),
    )
    dpo.train()
    dpo.save_model()
    dpo_delta = sum(
        float((v.detach() - before[k]).norm())
        for k, v in policy.named_parameters()
        if k in before
    )
    ref_delta = sum(
        float((v.detach() - ref_before[k]).norm())
        for k, v in policy.named_parameters()
        if k in ref_before
    )
    assert sft_delta > 0 and dpo_delta > 0 and ref_delta == 0
    result = {
        "status": "measured",
        "scope": "random_tiny_qwen3_cpu_integration_only",
        "pretrained_weights": False,
        "seed": 42,
        "architecture": config.to_dict(),
        "sft_lora_delta_l2_sum": sft_delta,
        "dpo_lora_delta_l2_sum": dpo_delta,
        "reference_adapter_delta_l2_sum": ref_delta,
        "sft_logs": trainer.state.log_history,
        "dpo_logs": dpo.state.log_history,
        "qwen3_8b_experiment": "not_run",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/trl-integration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    # Export the actual tiny trained policy for optional end-to-end CPU serving.
    policy.set_adapter("policy")
    merged = policy.merge_and_unload(adapter_names=["policy"])
    merged.save_pretrained("artifacts/trl-smoke/merged")
    tok.save_pretrained("artifacts/trl-smoke/merged")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("architecture", "sft_logs", "dpo_logs")
            }
        )
    )


if __name__ == "__main__":
    run()
