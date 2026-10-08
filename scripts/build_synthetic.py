"""Reproducible fictional policies; not real platform customer data."""
from pathlib import Path
import json
from ecom.data import prepare, write, validate_preferences

raw, pairs = [], []
for i in range(32):
    policy = f"虚构店铺{i:02d}"
    days, refund, threshold = 7 + i % 4, 2 + i % 3, 100 + i * 10
    context = f"店铺规则：未拆封商品签收后{days}天内可退；退款审核后{refund}个工作日到账；满{threshold}元包邮；优惠券不可叠加；拆封食品不支持无理由退货；发票在订单页申请；物流异常转人工核实。"
    scenarios = [
        ("未拆封的商品怎么退？", f"未拆封商品签收后{days}天内可退，请在订单页申请。", [str(days), "订单页"]),
        ("退款什么时候到账？", f"退款审核后{refund}个工作日到账，请在订单页查看进度。", [str(refund), "审核"]),
        ("买多少可以免运费？", f"满{threshold}元包邮。", [str(threshold)]),
        ("两张优惠券能叠加吗？", "优惠券不可叠加，请按订单页可选优惠使用。", ["不可叠加"]),
        ("食品拆封了还能无理由退吗？", "拆封食品不支持无理由退货；如有质量问题请联系人工核实。", ["不支持", "人工"]),
        ("如何申请发票？", "请在订单页申请发票。", ["订单页", "发票"]),
        ("物流停了三天，我很着急！", "理解您等待的焦虑，请联系人工核实物流异常；不能保证未核实的送达日期。", ["人工", "核实"]),
        ("商品库存有多少？", "规则没有库存信息，请在商品页查看或联系人工核实。", ["没有", "人工"]),
    ]
    for j, (q, answer, required) in enumerate(scenarios):
        prompt = f"{policy}。{context}\n用户：{q}"
        raw.append(dict(id=f"p{i:02d}-{j}", policy_id=policy, prompt=prompt, completion=answer, must_include=required, must_not_include=["保证明天", "一定到账"], source="authored_fictional_policy"))
        pairs.append(dict(prompt=prompt, chosen=answer, rejected="我保证明天全部处理好，所有商品都能退，优惠券随便叠加。"))
splits = prepare(raw)
write('data/ecommerce/raw.jsonl', raw)
write('data/ecommerce/preferences.jsonl', pairs)
write('data/ecommerce/eval.jsonl', splits['test'])
for name, rows in splits.items():
    write(f'artifacts/data/{name}.jsonl', rows)
write('artifacts/data/preferences.jsonl', validate_preferences(pairs, {r['prompt'] for r in splits['train']}))
Path('results').mkdir(exist_ok=True)
Path('results/ecommerce-data-audit.json').write_text(json.dumps(dict(status='measured_data_build_only', policies=32, rows=len(raw), counts={k:len(v) for k,v in splits.items()}, limitation='Synthetic templates with group-isolated policies; not production data or business accuracy.'), ensure_ascii=False, indent=2), encoding='utf-8')
print({k: len(v) for k, v in splits.items()})
