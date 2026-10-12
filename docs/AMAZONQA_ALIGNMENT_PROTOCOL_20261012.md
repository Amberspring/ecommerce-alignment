# AmazonQA 英文证据问答：训练与冻结验收协议（2026-10-12）

这轮修复旧中文合成政策数据与英文商品问答评测的任务错位。数据只来自 [AmazonQA 官方仓库](https://github.com/amazonqa/amazonqa)指向的 `train-qar.jsonl` 与 `val-qar.jsonl`；train 从 64 个等距区间、validation 从 24 个等距区间各取 1 MiB，丢弃范围边界行，保存范围和 SHA-256。该抽样按字节分布，并非全库逐记录均匀抽样；`is_answerable` 是作者分类器输出，不能称为人工真值。

SFT 仅使用官方 train 中被标为 answerable、有评论证据且提示不超过 4000 字符的问题。社区回答须至少有 1 个赞成票、赞成比例不低于 50%，并在模型可见的前五条、每条前 400 字符评论中通过保守词面过滤：2 至 60 个实词，至少一半实词能在证据中找到，回答出现的数字须全部出现在证据中；在合格回答中选择 Wilson 95% 下界最高者，平局按正票数、总票数、文本固定排序。这只是剔除明显无证据答案的代理，不能证明语义蕴含；首版 CPU 抽查发现高票答案与评论相矛盾，因此在任何 GPU 训练前加入此规则，首版 r1 保留为诊断，不用于训练。产品 ASIN 哈希取模 10 决定训练内验证产品，其余为训练产品；同一标准化问题跨产品出现时，从内部验证中移除，并在各子集内去重。DPO 候选只来自训练产品同一问题内同时通过证据过滤的真实社区回答：每条至少 2 票，胜者至少 2 个赞成票，Wilson 下界差至少 0.15；每题最多一对，数量不足时停止 DPO。

正式留出来自官方 validation，按预定 qid 哈希排序选 200 个 answerable 和 100 个 unanswerable，排除与训练来源 ASIN、qid、标准化问题哈希或原始记录哈希重合的样本。两个留出子集之间也不允许交叉。所有原始及派生问答放在 gitignore 的 `artifacts/` 下；仓库只提交构建程序、规则、无文本的 manifest 与文件哈希，数据许可未确认前不再分发数据。

## 冻结模型与门槛

基座为 `Qwen/Qwen3-8B`，首轮只做 QLoRA rank 16 的 SFT，参数沿用 `configs/qlora-r16-qv.json`，新目录保存模型、环境、seed、步数与权重哈希。Base/SFT 在同一 200 题 answerable 集上使用相同证据提示、temperature 0、非 thinking、最多 256 输出 token；截断、超时和空答按失败计，保存全部逐例输出。词面指标报告多参考答案最大 token F1、EM、配对 bootstrap 5000 次的 95% 区间；但 F1 只能作为辅助代理，不能称为事实正确率。

独立质量验收要求双盲人工评分或事先固定且保存逐例解释的独立模型评审，分别评正确性、评论证据支持度和帮助性；模型评审只能称为模型评审。SFT 可写成正面质量结果的门槛预先定为：200 题 F1 均值差的配对 bootstrap 95% 下界大于 0，且人工盲评 `support_rate_2`（评论完全支持的比例）的配对 95% 下界不低于 0，正确性与帮助性均分不能下降；配对区间按题聚类、固定 seed=20261010 抽样 5000 次。评审缺席时只报告词面探索结果。`unanswerable` 100 题另用独立评分表记录证据是否充分、正确拒答、无依据断言、证据不足识别与逐例坏例，不与 answerable F1 混算。DPO 只有在 SFT 通过上述门槛且同一冻结规则至少取得 200 对候选时才训练，并以同样协议比较 SFT/DPO；当前 r2 的 42 对不足，因此本轮不安排 DPO。

公开基准可能进入模型预训练；“本项目未用测试集调参”不等于模型从未见过题。社区投票稀疏，受欢迎回答未必受提供评论支持，训练后必须审查坏例。以上门槛是本项目在看新成绩前固定的验收标准，不是行业统一标准。

## 复现

```sh
python scripts/prepare_amazonqa_alignment.py --output artifacts/amazonqa-alignment-20261012-r2
python -m pytest -q tests/test_amazonqa_alignment.py
python scripts/check_token_lengths.py --data-dir artifacts/amazonqa-alignment-20261012-r2 --output results/amazonqa-alignment-20261012-r2/token-lengths-20261012-v2.json
python -m ecom.train --config configs/qlora-r16-qv.json --stage sft --data-dir artifacts/amazonqa-alignment-20261012-r2 --output artifacts/amazonqa-sft-20261012-r2
```

构建后将 `manifest.json` 中不含问答文本的统计和哈希复制到新的 `results/amazonqa-alignment-20261012-r2/`，记录原始命令与完整日志。旧 `python -m ecom.data` 只构建历史合成数据，不能作为新训练入口；旧适配器、旧 200 题和其原始比较一律保留。

CPU 预检使用 Qwen3-8B 的真实分词器和非 thinking 对话模板，`train` 1700 条最长 705 token、内部验证 189 条最长 682 token、两个留出集最长分别为 613 与 572 token，均未达到 1024 token 截断上限。该结果只验证输入长度，不代表答案质量；分词器与模板哈希保存在预检输出中。

模型服务就绪后分别对 `eval-answerable.jsonl` 和 `eval-unanswerable.jsonl` 执行 `scripts/compare_models.py`，`--endpoints` 固定为 Base/SFT 两个服务，`--output` 使用各自新目录。该脚本对 answerable 计算 F1/EM/配对区间，对 unanswerable 只计算明确表示证据不足的字符串代理；接口失败和截断留在分母。人工盲评表用 `python scripts/blind_review.py prepare --dataset ... --predictions ... --review ... --key ... --labels base,sft` 生成，评审看到问题和评论但看不到模型名；真实评审完成后再运行 `score`，不得预填或自称人工参与。
