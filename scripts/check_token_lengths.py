"""Check Qwen chat-template lengths before paying for GPU training."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/qlora-r16-qv.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    data = Path(args.data_dir)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    summary = {}
    for name in ("train", "validation", "eval-answerable", "eval-unanswerable"):
        path = data / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][path.name]:
            raise ValueError(f"Dataset hash mismatch: {path.name}")
        lengths = []
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            completion = row.get("completion", "")
            if completion:
                rendered += completion + tokenizer.eos_token
            lengths.append(len(tokenizer(rendered, add_special_tokens=False).input_ids))
        lengths.sort()
        summary[name] = {"n": len(lengths), "p50": lengths[len(lengths) // 2],
                         "p95": lengths[int(len(lengths) * .95)], "max": lengths[-1],
                         "over_max_length": sum(n > config["max_length"] for n in lengths)}
    report = {"model": config["model"],
              "tokenizer_json_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "max_length": config["max_length"], "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "splits": summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if any(row["over_max_length"] for row in summary.values()):
        raise SystemExit("Training or evaluation examples exceed max_length")


if __name__ == "__main__":
    main()
