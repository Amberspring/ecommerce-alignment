# 本机实际实验

环境与依赖见 results/environment.txt，测试记录见 results/tests.xml 与 tests.txt。
数据清洗：14 条输入，12 条有效，训练/验证/测试为 8/2/2；2 条拒绝；DPO 训练偏好只保留训练分区的 8 条。来源为合成商城政策。

## 最小 autograd 检查

TinyPolicy 只是一层 16 维条件字符分布，不是 Qwen3。固定 seed=42、CPU 训练：

| 指标 | 实测 |
|---|---:|
| SFT 初始 loss | 5.029090 |
| SFT 最终 loss | 3.874642 |
| DPO 初始 loss | 0.693147 |
| DPO 最终 loss | 0.000239 |
| DPO 参数差值 L2 | 10.785101 |

这只验证优化确实发生，不是泛化效果；偏好长度也会影响 summed log-probability。原始值见 results/cpu-smoke.json。

## 真实 TRL / PEFT / Qwen3 架构集成

随机初始化 2 层、hidden_size=32 的 Qwen3，LoRA rank=4；SFT 与 DPO 各 8 步。

| 指标 | 实测 |
|---|---:|
| SFT 可训练参数 L2 差值之和 | 2.406255 |
| DPO policy 参数 L2 差值之和 | 2.680711 |
| DPO reference 参数差值 | 0.000000 |

使用 2 个不同英文 prompt，各重复四次以完成集成步数；不是 8 个独立训练样本。记录见 results/trl-integration.json，包含每步 loss、gradient norm 和完整模型配置。

## Qwen3-8B GPU 主链路（2026-10-06）

硬件为 AutoDL 单张 NVIDIA GeForce RTX 4080 SUPER 32GB。数据是项目内 32 家虚构店铺政策生成的模板化样例，切分为 160/48/48；不是生产客服数据。

| 阶段 | 配置 | step | 运行时间 | PyTorch peak allocated | 参数验证 |
|---|---|---:|---:|---:|---|
| SFT | NF4 QLoRA, r=16, q/v, 2 epoch, max_len=1024 | 40 | 174.11s | 9.84GB | 144 个可训练张量变化 |
| DPO | SFT adapter 起点, beta=0.1, 2 epoch | 40 | 175.21s | 11.69GB | 144 个 policy 张量变化；reference 不变 |

SFT eval loss 从 epoch 1 的 0.2567 降到 epoch 2 的 0.001435，token accuracy 从 0.9311 到 1.0。DPO train loss 为 0.02814，训练 reward accuracy 在第 2 步后即达到 1.0。两项指标都表明模板数据过易，不能解释为真实业务泛化。可提交的原始记录位于 `results/gpu-sft-run.json` 与 `results/gpu-dpo-run.json`；本地适配器位于 gitignored 的 `artifacts/`。

## 尚未运行

LoRA rank=4/16/64、target modules、LoRA/QLoRA 全矩阵消融，独立人工偏好胜率与真实客服质量评测。配置和执行器已提供，结果表保持空值。
