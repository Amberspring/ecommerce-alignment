# 独立评测与证据边界

`scripts/compare_models.py` 对 Base/SFT/DPO 使用同一数据集与生成参数，逐条保存响应、usage、结束原因和时延，并记录数据与预测 SHA-256。截断回答不算通过；纯数字关键词使用数字边界匹配，避免“17天”被误当成“7天”。已有训练产物不代表客服质量提升。

```sh
python scripts/compare_models.py --dataset data/ecommerce/eval.jsonl --endpoints configs/evaluation-endpoints.example.json --output results/comparison-<unique-run-id>
```

示例 endpoint 的模型名必须与实际服务一致。可在支持 LoRA 的同一基座服务上注册两个适配器，或者提供三个独立地址。每次使用全新结果目录；请求中断保留已收集响应，不伪造剩余预测。

**当前语料全部为自行编写的合成教学数据。** 即使运行真实 Qwen3-8B，结果也只证明合成政策规则遵循，不能称为真实客户准确率、幻觉率或人工偏好胜率。用户要求的数据可验证原则适用于正式实验：真实客服来源、使用权限、脱敏、独立标签和人工复核未补齐前，不发布真实业务效果结论。GRPO、困难偏好负例、盲评不填预期提升数字。

真实商品问答审计入口 `python scripts/prepare_amazonqa.py` 使用 [AmazonQA 作者仓库](https://github.com/amazonqa/amazonqa) 给出的 validation 下载地址。脚本在完整文件中读取 64 个固定的 1MiB 范围，保存每段 SHA-256，从 8179 条候选中按 qid 哈希冻结 200 题；模型只看问题和最多五条评论片段，参考回答不进入提示。比较脚本同时报告规范化 token F1、逐题配对差值及 bootstrap 区间、截断、响应长度和时延；这些仍不等于事实准确率或业务提升。

历史 2026-10-08 的文件前缀 20 题由 `scripts/prepare_amazonqa_legacy20.py` 重建。新旧样本不合并，任何新实验写入新目录；真人盲评的准备、评分和防冒充检查见 [HUMAN_EVAL.md](HUMAN_EVAL.md)。

源数据与含参考回答的样本保留在被 Git 忽略的 artifacts，未声明原数据拥有无限制再分发许可。公开 manifest 保存源 URL、ETag、范围、样本 ID、哈希和分布，方便研究者自行下载复核；不把作者的自动 answerability 分类标签当成人工事实标签。
