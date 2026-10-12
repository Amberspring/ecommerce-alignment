# 独立评测与真人盲评

## 冻结数据

`scripts/prepare_amazonqa.py` 从 AmazonQA 官方 validation 文件的完整 747,051,157 字节中，按 seed=20261010 在 64 个等距区间内各取 1 MiB，再按 qid 哈希选择 200 条有效问题。每段内容哈希、最终 ID、类别分布与数据哈希写入 manifest；原始片段、参考回答和评测集留在 `artifacts/`，避免在数据许可不清楚时再分发。该抽样覆盖全文件，但按字节区间抽样并非严格的逐记录均匀随机抽样。

```sh
python scripts/prepare_amazonqa.py --output artifacts/amazonqa-audit-20261010
python scripts/compare_models.py --dataset artifacts/amazonqa-audit-20261010/eval.jsonl --endpoints configs/evaluation-endpoints.example.json --output results/amazonqa-<run-id>
```

模型只看到问题和最多五条评论证据，看不到社区参考答案。自动结果报告 token F1、逐题差值、固定 seed 的配对 bootstrap 95% 区间、截断数、响应长度和端到端时延；这些指标用于定位差异，仍不等于事实正确率或偏好胜率。历史 20 题可用 `scripts/prepare_amazonqa_legacy20.py` 精确重建，不能和新 200 题混为同一实验。

## 盲评

先从完整的 Base/SFT/DPO 预测目录生成盲评表和单独保存的映射密钥：

```sh
python scripts/blind_review.py prepare --dataset artifacts/amazonqa-audit-20261010/eval.jsonl --predictions results/amazonqa-<run-id> --review artifacts/review/reviewer-1.csv --key artifacts/review/key.json
```

评审人只能拿到 CSV，不应看到 key、模型名或自动指标。对 answerable 集的每个 A/B/C 回答分别填写 correctness、support、helpfulness，取值 0=不合格、1=部分合格、2=合格；`best` 填 A、B、C 或 TIE，`reviewer_id` 必须是真实参与者的稳定匿名编号。correctness 判断答案是否回应问题，support 的 2 分表示回答完全受给定评论支撑，helpfulness 判断是否简洁、有用且没有无依据承诺。

对 unanswerable 集另行生成评分表。先填 `evidence_sufficient`：仅当可见评论足以回答问题全部关键部分时填 1，否则填 0；这个复核项用于暴露 AmazonQA 分类器可能误标的题。逐个答案填写 `insufficiency_recognition`（明确指出缺少证据）、`unsupported_assertion`（出现评论不支持的具体商品断言）和 `correct_refusal`（证据不足时明确拒绝猜测，且没有无依据断言），三项填 0 或 1；`helpfulness` 仍填 0 至 2。证据其实充分的题不能给 `correct_refusal` 或 `insufficiency_recognition` 加分。两个子集的分数、分母和坏例分别报告，不能混成一个质量数字。

完成后汇总；两个评审人应分别复制原始空表填写，不能互相覆盖：

```sh
python scripts/blind_review.py score --review artifacts/review/reviewer-1.csv --review artifacts/review/reviewer-2.csv --key artifacts/review/key.json --output results/amazonqa-<run-id>/human-review.json
```

脚本拒绝空 reviewer、缺题、重复题、越界分数、篡改后的问题/证据/答案和未填写的 best。报告保存评分文件 SHA-256、逐题分数和按题聚类的 5000 次配对 bootstrap 区间；answerable 的 `support_rate_2` 是评论完全支持的比例，用于预先冻结的验收门槛。报告状态只写 `reviewer_entries_recorded`，因为 reviewer ID 由填写者自报，程序无法证明真人参与；应另存真人评审的参与记录。若没有真人完成 CSV，就只能报告“盲评工具已准备”，不能写成人评成绩。
