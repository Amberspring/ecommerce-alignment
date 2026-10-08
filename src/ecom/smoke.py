"""Actual CPU autograd SFT/DPO on a tiny conditional model, not Qwen performance."""

import copy, json, hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def dpo_loss(pc, pr, rc, rr, beta=0.1):
    return -F.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()


class TinyPolicy(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.emb = nn.Embedding(n, 16)
        self.out = nn.Linear(16, n)

    def forward(self, x):
        return self.out(self.emb(x))


def run():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    rows = [
        json.loads(s)
        for s in Path("data/preferences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chars = sorted(
        set("".join(r[k] for r in rows for k in ("prompt", "chosen", "rejected")))
    )
    ids = {c: i for i, c in enumerate(chars)}

    def encode(s):
        return torch.tensor([ids[c] for c in s])

    model = TinyPolicy(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)

    def lp(m, p, s):
        # Prompt mean embedding conditions every completion token.
        logits = m.out(m.emb(encode(p)).mean(0))
        return F.log_softmax(logits, dim=-1)[encode(s)].sum()

    def sft():
        return torch.stack(
            [-lp(model, r["prompt"], r["chosen"]) / len(r["chosen"]) for r in rows]
        ).mean()

    first = float(sft().detach())
    grads = []
    for _ in range(30):
        opt.zero_grad()
        loss = sft()
        loss.backward()
        grads.append(float(model.out.weight.grad.norm()))
        opt.step()
    final = float(sft().detach())
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    def loss_dpo():
        values = []
        for r in rows:
            with torch.no_grad():
                rc = lp(ref, r["prompt"], r["chosen"])
                rr = lp(ref, r["prompt"], r["rejected"])
            values.append(
                dpo_loss(
                    lp(model, r["prompt"], r["chosen"]),
                    lp(model, r["prompt"], r["rejected"]),
                    rc,
                    rr,
                )
            )
        return torch.stack(values).mean()

    before = float(loss_dpo().detach())
    snapshot = model.out.weight.detach().clone()
    for _ in range(20):
        opt.zero_grad()
        loss = loss_dpo()
        loss.backward()
        opt.step()
    after = float(loss_dpo().detach())
    delta = float((model.out.weight.detach() - snapshot).norm())
    assert final < first and after < before and delta > 0 and all(g > 0 for g in grads)
    result = {
        "status": "measured",
        "scope": "tiny_cpu_autograd_smoke_only",
        "model": "TinyPolicy (16-dimensional conditional character distribution)",
        "seed": 42,
        "torch": torch.__version__,
        "device": "cpu",
        "examples": len(rows),
        "data_sha256": hashlib.sha256(
            Path("data/preferences.jsonl").read_bytes()
        ).hexdigest(),
        "sft_initial_loss": first,
        "sft_final_loss": final,
        "dpo_initial_loss": before,
        "dpo_final_loss": after,
        "dpo_parameter_delta_l2": delta,
        "min_sft_grad_norm": min(grads),
        "qwen_experiment": "not_run",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cpu-smoke.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    run()
