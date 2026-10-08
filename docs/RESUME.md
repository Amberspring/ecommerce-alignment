# 可直接用于简历的正文

**电商客服大模型训练与偏好对齐｜Qwen3-8B、TRL、PEFT、QLoRA、DPO**

面向退换货、物流、商品参数与售后投诉场景，构建 32 家虚构店铺政策驱动的 256 条客服指令数据，完成脱敏、去重、短答过滤及按店铺隔离的 160/48/48 切分，避免同店政策跨训练与测试分区泄漏。基于 Qwen3-8B 在单张 RTX 4080 SUPER 32GB 上完成 NF4 QLoRA SFT 与 DPO 两阶段训练；LoRA rank=16、目标层 q_proj/v_proj，SFT/DPO 各运行 2 epoch、40 step，PyTorch 峰值分配显存分别为 9.84GB/11.69GB，并通过 144 个可训练张量变化和 DPO reference 零变化验证梯度链与冻结逻辑。建立独立评测、预测 ID 完整性校验及复读、答非所问、事实冲突等 badcase 回归流程。

边界：数据为模板化合成数据；SFT 验证 token accuracy 达到 1.0、DPO 训练 reward accuracy 很快饱和，只能说明数据过易和过拟合风险，不能写成真实客服准确率或人工偏好胜率。尚未完成全部 rank/target-module 消融。
