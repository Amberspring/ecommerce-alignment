# 简历表述与证据门槛

## 当前可以写

**电商客服大模型后训练与评测｜Qwen3-8B、QLoRA、DPO、TRL、PEFT**

围绕电商政策遵循构建 Base/SFT/DPO 同条件实验链路，完成按店铺隔离的数据切分、completion-only SFT、双适配器 DPO、参数更新与 reference 冻结校验，并保存逐例输出、运行配置和数据哈希。Qwen3-8B 在单张 RTX 4080 SUPER 32GB 上完成 SFT/DPO 各 40 step；48 条合成政策题中 Base、SFT、DPO 分别通过 31、48、48 条，但外部 AmazonQA 20 题 token F1 约为 0.215、0.221、0.219，未观察到 DPO 相对 SFT 的可靠增益。进一步冻结覆盖完整验证文件的 200 题外部集，并实现模型身份盲化、人工评分完整性校验和配对 bootstrap 分析。

面试时必须主动说明：48 题是模板化虚构政策，规则分数受输出长度和 Base 截断影响；AmazonQA 是英文商品问答，不等同于中文客服业务；200 题的新模型结果和真人盲评尚未完成。这个版本展示的是后训练实现、实验设计和负结果分析，不是线上客服效果。

## 完成新验收后才能写

只有在新 200 题 Base/SFT/DPO 预测、至少两名确有参与的盲评人、逐题评分和适配器备份全部落盘后，才能补写事实支持率、偏好胜率、置信区间及成本。DPO 没有超过 SFT 时应保留负结果，并把项目重点放在“为什么训练指标饱和不代表外部泛化”；不得改用训练 reward accuracy、合成规则通过率或模型裁判分数替代真人结果。

## 证据索引

- GPU 训练摘要：`results/gpu-sft-run.json`、`results/gpu-dpo-run.json`
- 合成政策逐例输出：`results/policy-20261008/`
- 历史外部审计：`results/amazonqa-20261008/`
- 新冻结集 manifest：`results/amazonqa-20261010/source-manifest.json`
- 协议与人评：`docs/EVALUATION_CHAIN.md`、`docs/HUMAN_EVAL.md`

本仓库和当前本机证据包不含 8B 适配器。远端原文件存在性未核实前，不能写“模型权重可下载复现”。
