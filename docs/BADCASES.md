# 已遇到的训练问题

| 问题 | 证据与修复 | 验证 |
|---|---|---|
| named adapter 取名 train 与 PyTorch Module.train 冲突 | CPU 集成曾触发 KeyError，改名 policy；GPU 入口同步修复 | TRL CPU SFT/DPO 跑通 |
| prompt 与 completion 拼接边界不一致 | 本地词表 tokenizer 的 prompt 缺少尾空格，触发前缀不一致警告 | 增加边界空格后消失；Qwen 使用官方聊天模板 |
| 看似 DPO 训练但参考模型可能更新 | 明确复制 SFT 的 reference adapter，使用参考参数前后差值检查 | reference delta=0；policy delta>0 |
| 同一问题重复进入不同分区 | 按规范化 prompt 哈希去重并隔离 | tests/test_data.py |
| 人工规则得分被误当偏好胜率 | 结果字段使用 rubric_pass_rate，文档明确人工评审要求 | 预测 ID 与规则测试 |
| 通用 tokenizer 产生 Qwen3 不支持的 token_type_ids | CPU 服务首次请求报错，推理编码显式排除该字段 | 实际 HTTP 请求通过，增加生成回归测试 |

这些是实际实现过程中发现的问题，不是声称发生过的 8B GPU 训练故障。生产客服复读、幻觉、答非所问仍需用真实生成结果填入 badcase-template.csv。
