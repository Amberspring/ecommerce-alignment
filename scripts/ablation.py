import subprocess, sys, argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--execute", action="store_true")
p.add_argument("--stage", choices=["sft", "dpo"], default="sft")
a = p.parse_args()
for config in sorted(Path("configs").glob("*.json")):
    c = json.loads(config.read_text())
    if "qlora" not in c or config.name.startswith("dpo-") != (a.stage == "dpo"):
        continue
    cmd = [
        sys.executable,
        "-m",
        "ecom.train",
        "--config",
        str(config),
        "--stage",
        a.stage,
    ]
    print(" ".join(cmd))
    if a.execute:
        subprocess.run(cmd, check=True)
