# 05 - 在无标签数据上预训练

## 在第 00 阶段上扩展

保留已有的 config、CSV、TensorBoard、`best.pt`、`last.pt` 和 resume，只替换数据、模型、loss 与生成监控。

## 实现清单

- [ ] token-level cross entropy
- [ ] train/validation loss 评估
- [ ] 周期性生成样本
- [ ] temperature 与 top-k
- [ ] checkpoint 与断点续训
- [ ] 加载预训练权重

## 门禁

使用一个全新 run 目录复现实验，解释 loss 曲线、生成样本变化和最终选择的 checkpoint。
