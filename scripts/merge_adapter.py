import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen3-8B")
p.add_argument("--adapter", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError(
        "This merge profile requires CUDA and enough memory for unquantized base weights"
    )
base = AutoModelForCausalLM.from_pretrained(
    a.base, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
model.save_pretrained(a.output, safe_serialization=True)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.output)
Path(a.output, "merge.json").write_text(
    json.dumps({"base": a.base, "adapter": a.adapter, "status": "merged"}),
    encoding="utf-8",
)
