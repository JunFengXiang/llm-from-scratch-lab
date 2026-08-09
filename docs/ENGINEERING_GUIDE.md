# 工程学习规范

## 1. 一次实验必须能回答五个问题

1. 要验证的假设是什么？
2. 与上次相比只改变了什么？
3. 使用了哪份配置和随机种子？
4. 观察到了哪些训练/验证指标？
5. 结论是什么，下一次要改什么？

如果回答不了，就先不要比较两个实验的好坏。

## 2. 配置、代码、产物分离

- 配置放入 `configs/` 并提交 Git；
- 可复用实现放入 `src/`；
- 测试放入 `tests/`；
- 本地日志、权重和数据放入 `artifacts/` 或 `data/`，由 `.gitignore` 排除；
- 结论写入 `experiments/`，只记录必要的小型表格和文字。

不要用文件名 `final_final_v2.py` 管理版本，Git 已经负责版本历史。

## 3. 推荐的 Git 节奏

每项任务使用一个短分支，例如：

```text
feat/tokenizer
feat/causal-attention
experiment/learning-rate
fix/checkpoint-resume
```

一个 commit 只表达一个完整意图：

```text
feat: implement sliding-window dataset
test: verify causal mask blocks future tokens
docs: record learning-rate experiment
```

提交前运行：

```bash
python -m ruff check .
python -m pytest
git status
git diff --cached
```

## 4. 可复现性最低要求

- 固定随机种子；
- 保存完整配置；
- 日志使用 `global_step` 或 epoch 作为清晰横轴；
- checkpoint 同时保存模型、优化器、epoch 和最佳验证指标；
- 记录 Python、PyTorch、设备和 commit SHA；
- 不在同一个 run 目录混写两个不同实验。

“同一随机种子”不保证所有 GPU 算子完全确定，但它是建立可复现实验的第一步。

## 5. `last.pt` 与 `best.pt`

- `last.pt`：最近的训练状态，主要用于中断后继续；
- `best.pt`：验证指标最好的状态，主要用于最终评估或推理。

checkpoint 只从可信来源加载，因为它属于可执行的序列化数据，而不是普通文本。

## 6. 从测试开始形成编程能力

对神经网络组件至少检查：

- 输入和输出形状；
- dtype 与 device；
- 边界输入和非法参数；
- 固定小样例的数值结果；
- 梯度是否存在；
- 一次优化后参数是否真的变化。

测试的目的不是追求覆盖率数字，而是把“我认为代码对”变成可重复验证的证据。

## 7. 什么时候使用 Notebook

Notebook 适合：观察数据、手算小矩阵、画图和探索 API。

正式训练逻辑应逐步迁移到 `.py` 文件，因为脚本更容易测试、复用、比较配置和断点续训。Notebook 可以调用 `src/` 中的实现，不应维护另一份重复代码。
