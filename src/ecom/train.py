"""Qwen training entrypoint. Requires real weights and CUDA; never substitutes a toy model."""

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text(encoding="utf-8"))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3-8B training requires CUDA in this configuration; use ecom.smoke for CPU verification."
        )
    from transformers import set_seed

    set_seed(42)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

    tok = AutoTokenizer.from_pretrained(c["model"])
    tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kw = {"torch_dtype": dtype}
    if c["qlora"]:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kw["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(c["model"], **kw)
    model.config.use_cache = False
    if c["qlora"]:
        model = prepare_model_for_kbit_training(model)
    lc = LoraConfig(
        r=c["rank"],
        lora_alpha=c["rank"] * 2,
        lora_dropout=0.05,
        target_modules=c["target_modules"],
        task_type="CAUSAL_LM",
    )
    shared = dict(
        output_dir=c["output"] + "/" + a.stage,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=c.get("dpo_learning_rate", 2e-5)
        if a.stage == "dpo"
        else c["learning_rate"],
        num_train_epochs=c["epochs"],
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        seed=42,
        report_to="none",
        logging_steps=1,
        save_strategy="epoch",
    )
    if a.stage == "sft":
        ds = load_dataset(
            "json",
            data_files={
                s: f"artifacts/data/{s}.jsonl" for s in ("train", "validation")
            },
        )

        # Render non-thinking prompts explicitly; only completion tokens contribute to loss.
        def render(r):
            return {
                "prompt": tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "completion": r["completion"] + tok.eos_token,
            }

        ds = ds.map(render, remove_columns=ds["train"].column_names)
        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            peft_config=lc,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            args=SFTConfig(
                **shared,
                max_length=c["max_length"],
                completion_only_loss=True,
                eval_strategy="epoch",
            ),
        )
    else:
        adapter = Path(c.get("sft_adapter", str(Path(c["output"]) / "sft")))
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError("Run SFT before DPO")
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=True, adapter_name="policy"
        )
        model.load_adapter(str(adapter), adapter_name="reference", is_trainable=False)
        model.set_adapter("policy")
        ds = load_dataset(
            "json", data_files="artifacts/data/preferences.jsonl", split="train"
        )
        ds = ds.map(
            lambda r: {
                "prompt": tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            }
        )
        trainer = DPOTrainer(
            model=model,
            processing_class=tok,
            train_dataset=ds,
            args=DPOConfig(
                **shared,
                beta=c["beta"],
                max_length=c["max_length"],
                model_adapter_name="policy",
                ref_adapter_name="reference",
            ),
        )
    before = {
        n: p.detach().cpu().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    reference = {
        n: p.detach().cpu().clone()
        for n, p in trainer.model.named_parameters()
        if ".reference." in n
    }
    trainer.train()
    current = dict(trainer.model.named_parameters())
    changed = sum(
        not torch.equal(value, current[name].detach().cpu())
        for name, value in before.items()
    )
    reference_unchanged = all(
        torch.equal(value, current[name].detach().cpu())
        for name, value in reference.items()
    )
    if not changed or not reference_unchanged:
        raise RuntimeError("Training update or frozen-reference verification failed")
    trainer.save_model()
    tok.save_pretrained(shared["output_dir"])
    meta = {
        "status": "measured",
        "config": c,
        "stage": a.stage,
        "changed_trainable_tensors": changed,
        "reference_unchanged": reference_unchanged if reference else None,
        "cuda": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "history": trainer.state.log_history,
    }
    Path(shared["output_dir"], "run.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
