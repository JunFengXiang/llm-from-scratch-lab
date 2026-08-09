# 00 - 训练工程基础

这是当前优先级最高的阶段。先用两维分类 MLP 掌握训练闭环，再把同一套工程能力迁移到 GPT。

## 今日任务

1. 运行正常训练并查看 `metrics.csv`；
2. 启动 TensorBoard 对照 train/validation loss；
3. 运行 `--overfit-one-batch`；
4. 在 `loss.backward()` 和 `optimizer.step()` 处打断点；
5. 中断一次训练，并使用 `last.pt` 恢复；
6. 运行 `ruff` 和 `pytest`。

## 必须回答

- batch、step、epoch、global step 分别是什么？
- `model.train()` 与 `model.eval()` 改变了什么？
- 为什么验证阶段不更新参数？
- 为什么断点续训要保存 optimizer state？
- 最终评估通常选择 `best.pt` 还是 `last.pt`？

## 复盘

- [ ] 我能闭卷写出一次 batch 更新
- [ ] 我能找到 CSV 和 TensorBoard 日志
- [ ] 我能解释一条异常曲线
- [ ] 我能从 checkpoint 恢复训练
- [ ] 所有测试通过
