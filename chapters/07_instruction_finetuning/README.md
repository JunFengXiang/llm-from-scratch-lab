# 07 - 通过微调遵循人类指令

## 实现清单

- [ ] 明确的 instruction/input/output 模板
- [ ] 训练标签与 padding mask
- [ ] 自定义 collate function
- [ ] 预训练权重加载
- [ ] 指令微调训练
- [ ] 回复抽取与 JSON 保存
- [ ] 自动指标与人工误差分析

## 风险检查

- 模板是否在训练和推理时一致；
- padding 是否被 loss 忽略；
- 评估集答案是否泄漏进 prompt；
- 只看平均分是否掩盖严重失败类别。

## 门禁

追踪一条样本从原始 JSON 到 token、label、loss 和生成回复的完整生命周期。
