# 电商客服大模型训练与偏好对齐

Qwen3-8B 的 QLoRA SFT、DPO、同条件模型对照与证据审计项目。实现数据清洗和隔离切分、训练参数更新检查、逐例预测留痕、外部数据抽样、配对不确定性分析及真人盲评流程。

## 2026-10-12：真实英文训练数据准备

旧适配器在冻结 AmazonQA 200 题上的 Base/SFT/DPO token F1 分别为 0.20225、0.20851、0.20848；SFT−Base 的配对 95% 区间为 [-0.00322, 0.01568]，DPO−Base 为 [-0.00376, 0.01629]，均不足以证明质量改善。原始 GPU 输出在单独的本机证据包中，旧训练、旧评测和旧适配器备份保持不变。

新流水线改用官方 AmazonQA train 中的英文评论证据问答，依据人类 helpful 投票选目标、过滤明显缺乏评论支持的答案，并按 ASIN 隔离训练、内部验证与官方 validation 留出。新构建 r2 得到 SFT 训练 1700 条、内部验证 189 条、独立 answerable 200 条、unanswerable 100 条，四类泄漏交叉均为零；符合固定投票与证据规则的 DPO 候选仅 42 对，未达到预设的 200 对训练门槛。规则、来源范围哈希、完整复现命令及统计见 [冻结协议](docs/AMAZONQA_ALIGNMENT_PROTOCOL_20261012.md)和 [manifest](results/amazonqa-alignment-20261012-r2/source-manifest.json)。这些只是 CPU 数据准备结果，新 SFT 尚无模型质量成绩；证据过滤是词面代理，不等于人工核实的事实支持。

```sh
python scripts/prepare_amazonqa_alignment.py --output artifacts/amazonqa-alignment-20261012-r2
python -m pytest -q tests/test_amazonqa_alignment.py
python -m ecom.train --config configs/qlora-r16-qv.json --stage sft --data-dir artifacts/amazonqa-alignment-20261012-r2 --output artifacts/amazonqa-sft-20261012-r2
```

原 `python -m ecom.data` 只适用于合成政策教学数据，不能作为新训练入口；`ecom.train` 现要求显式指定新数据目录和唯一输出目录。

[![CPU checks](https://github.com/Amberspring/ecommerce-alignment/actions/workflows/cpu-checks.yml/badge.svg)](https://github.com/Amberspring/ecommerce-alignment/actions/workflows/cpu-checks.yml)

| 当前证据 | 结论 |
|---|---|
| 2026-10-06，RTX 4080 SUPER 32GB，SFT/DPO 各 40 step | 训练链路、policy 更新和 reference 冻结得到验证；不代表业务泛化。 |
| 48 条合成政策题：Base 31/48，SFT 48/48，DPO 48/48 | 模板数据已饱和，且 Base 有 6 条截断；不能据此宣布 DPO 优于 SFT。 |
| 历史 AmazonQA 20 题：F1 约 0.215/0.221/0.219，EM 均为 0 | 没有观察到可靠的 SFT/DPO 外部增益；样本太小。 |
| 旧冻结 AmazonQA 200 题 | 2026-10-12 Base/SFT/DPO 已实测，F1 约 0.202/0.209/0.208，增量区间跨零；真人盲评尚未运行。 |

仓库不含 8B 适配器权重；本机另有已校验的旧适配器备份，但新 SFT 尚未训练。项目当前可证明后训练实验设计和负结果分析，尚不能声称真实客服质量提升；详见[独立评测与真人盲评](docs/HUMAN_EVAL.md)和[下一次模型验收预算](docs/NEXT_RUN_BUDGET.md)。

**当前交付：CPU 工程验证 + Qwen3-8B GPU 主链路实测。** QLoRA SFT 与 DPO 各训练 2 epoch/40 step；SFT 和 DPO 分别记录 9.84GB、11.69GB 的 PyTorch peak allocated memory，均有 144 个可训练张量发生变化，DPO reference 保持不变。
CPU 实验确实调用 TRL、PEFT 和 Qwen3 架构，但模型很小、随机初始化；不能把其 loss、token accuracy 或 reward accuracy 写成 8B 客服准确率。
另已把实际训练后的微型策略模型通过 HTTP 接入项目三，验证训练产物到服务的完整链路；微型模型不能用于实际客服。

## 安装与本机复现

推荐 Python 3.12。以下命令在当前仓库目录运行，python 应指向你自己的虚拟环境。

```sh
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[trl-smoke,eval]" pytest==8.3.5
python -m ecom.data
python -m ecom.smoke
python -m ecom.trl_smoke
python -m pytest -q
```

执行后，数据切分位于 artifacts/data/，训练产物位于 artifacts/trl-smoke/，真实运行记录位于 results/。仅测试数据清洗可安装 requirements.txt；训练验证需额外安装上述 CPU 依赖。

可选真实 CPU 推理：安装 `python -m pip install -e ".[serve]"`，运行 `python -m uvicorn ecom.cpu_api:app --host 127.0.0.1 --port 8000`。它加载 artifacts/trl-smoke/merged 的实际训练产物，只验证本地英语小词表与接口；例如 refund 返回简短教学输出。zip 不包含 artifacts，解压后先运行 trl_smoke 重建。

## 实现与设计

```mermaid
flowchart LR
 A[原始客服数据] --> B[规范化/脱敏/去重]
 B --> C[按问题分组切分]
 C --> D[LoRA 或 QLoRA SFT]
 D --> E[SFT适配器作为DPO参考]
 E --> F[DPO策略适配器]
 F --> G[独立评测/坏例回归]
```

- 去除空样本、超长样本和重复问题；相同问题即使回答不同也只能进入一个分区。
- 正则遮盖手机号和邮箱。它不是完整的个人信息识别器；真实数据仍需要人工审核。
- 偏好训练数据只使用 SFT 训练分区的问题，避免把测试问题泄漏进 DPO。
- SFT 显式使用 Qwen 非思考模板，仅 completion 部分计算 loss。
- DPO 加载 SFT 的 policy/reference 两份适配器，参考适配器不参与更新。
- LoRA rank=4/8/16/64、q/v 与全线性层、NF4 QLoRA 配置已提供；DPO beta=0.05/0.1/0.3 共用同一 SFT 起点；没有预先填入胜负结论。
- 评测检查预测 ID 完整性，提供关键词规则得分与坏例分类。规则通过率不等于人工偏好胜率。

## GPU 实验与复现

在 Linux NVIDIA CUDA 环境另建虚拟环境，安装适合显卡的 PyTorch 后执行：

```sh
python -m pip install -e ".[gpu,eval]" pytest
python scripts/prepare_amazonqa_alignment.py --output artifacts/amazonqa-alignment-20261012-r2
python -m ecom.train --config configs/qlora-r16-qv.json --stage sft --data-dir artifacts/amazonqa-alignment-20261012-r2 --output artifacts/amazonqa-sft-20261012-r2
# DPO 仅在冻结的质量与偏好样本门槛通过后另设唯一目录运行。
```

完整 8B 模型是否能训练取决于显存、长度、优化器及精度，不能仅凭“8B”保证某张卡够用。QLoRA 依赖 bitsandbytes；它不等于部署用的 AWQ/GPTQ。
保存 DPO 时 named adapter 位于 policy/ 子目录；部署时合并该策略适配器，不要合并 reference/。

通过项目三启动真实模型服务后：

```sh
python scripts/predict.py --output artifacts/predictions.jsonl
python -m ecom.evaluate --predictions artifacts/predictions.jsonl --output results/model-evaluation.json
```

真实消融必须保持数据切分、seed、训练步数和评测问题一致。需要另外建立独立人工偏好评测集，并记录评审标准；不能把训练集的 DPO reward accuracy 当成偏好胜率。

## 仓库内容与证据

- src/ecom/：清洗、训练、梯度 smoke、真实 TRL 集成及评测。
- configs/：四组训练配置；scripts/：消融、合并适配器、预测收集。
- data/：虚构客服政策与偏好样例；tests/：隔离切分、无泄漏、评测及梯度数值检查。
- results/：本机实测 JSON、测试报告、环境记录；experiments/：未运行实验模板。
- docs/EXPERIMENTS.md：实际结果和限制；docs/BADCASES.md：已遇到的问题及修复。

上传方式见 [docs/GITHUB.md](docs/GITHUB.md)，接口依据见 [docs/SOURCES.md](docs/SOURCES.md)。

当前原帖实验数据入口：`python scripts/build_synthetic.py`，按店铺划分 160/48/48。不要在此后运行默认 `ecom.data` 覆盖新训练集。数据为自行编写的模板化合成样例。

## 2026-10-08 真实模型验收

复用 2026-10-06 的 Qwen3-8B SFT/DPO 适配器，统一 FP16、temperature=0、非 thinking 和生成预算，48 条合成政策留出题的规则通过率为 Base 31/48、SFT 48/48、DPO 48/48。原始生成、结束原因与汇总位于 `results/policy-20261008/`；这不是实际客户满意度，也未证明 DPO 额外收益。

另用 AmazonQA 官方验证文件固定前 1 MiB 中前 20 个不同问题，向模型提供评论证据而不提供人类答案；Base/SFT/DPO 的英文 token F1 分别为 0.214826、0.220605、0.218768，exact match 均为 0。来源、范围、数据哈希与指标见 `results/amazonqa-20261008/`，复现使用 `scripts/prepare_amazonqa_legacy20.py`。该小型顺序子集非代表性抽样，F1 仅衡量词面重合，公开数据也可能出现在预训练中；新的 200 题冻结集由 `scripts/prepare_amazonqa.py` 生成，尚无模型成绩。

相同 256-token 预算下，Base 有 6 条截断，SFT/DPO 均无截断，截断计为不通过；平均输出长度分别为 134.75、13.75、13.58 token。政策规则成绩同时反映简洁输出与规则匹配，不能全部归因于事实能力差异，统计见 `results/policy-20261008/generation-audit.json`。
