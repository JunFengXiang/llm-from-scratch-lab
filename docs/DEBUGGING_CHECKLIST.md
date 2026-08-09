# 训练调试清单

按顺序检查，不要一看到 loss 异常就先换模型。

## 1. 数据

- 随机打印一个 batch 的输入、标签、shape、dtype 和范围；
- 分类标签是否从 `0` 开始且小于类别数；
- 语言模型的 target 是否确实比 input 向后错一位；
- 训练集和验证集是否意外重叠；
- shuffle 是否只用于训练集。

## 2. 前向传播

- logits 的最后一维是否等于类别数或词表大小；
- 模型输出是否包含 `NaN`/`Inf`；
- loss 函数期待 logits 还是概率；
- 是否错误地在交叉熵前手动做了 softmax。

## 3. 梯度与更新

- `optimizer.zero_grad()` 是否在每次更新前执行；
- `loss.backward()` 后关键参数的 `.grad` 是否存在且非零；
- `optimizer.step()` 后参数是否发生改变；
- 是否在训练阶段意外使用了 `torch.no_grad()`；
- 学习率是否大到发散或小到几乎没有更新。

## 4. 最小过拟合测试

固定一个很小的 batch，关闭不必要的数据增强，重复训练。若无法把 loss 降得很低，优先怀疑实现、标签、loss 或优化器，而不是模型容量。

本仓库命令：

```bash
python -m llm_from_scratch_lab.train_toy --overfit-one-batch
```

## 5. train 与 eval 模式

- 训练前调用 `model.train()`；
- 验证前调用 `model.eval()`；
- 验证使用 `torch.inference_mode()` 或 `torch.no_grad()`；
- 验证阶段不调用 `backward()` 和 `optimizer.step()`。

这会影响 dropout、batch normalization 和显存使用。

## 6. 读曲线

| 现象 | 优先检查 |
|---|---|
| train/val loss 一起下降 | 正常学习，继续观察 |
| train 下降、val 上升 | 过拟合、数据分布或泄漏 |
| 两者都不下降 | 数据、梯度、学习率、模型实现 |
| loss 剧烈震荡 | 学习率、batch 太小、异常样本 |
| loss 变成 NaN | 数值稳定性、学习率、错误除法 |
| 指标异常地完美 | 数据泄漏、重复样本、评估代码 |

## 7. checkpoint 恢复

- 模型结构和配置是否与 checkpoint 一致；
- 模型与优化器状态是否都已加载；
- 恢复后的 epoch 和 `global_step` 是否连续；
- `map_location` 是否与当前设备匹配；
- 恢复后先跑一次验证，确认指标接近中断前。
