# LLM From Scratch Lab

一个围绕《从零构建大模型》建立的、强调**独立实现、训练闭环、测试和实验复盘**的学习仓库。

这个仓库的目标不是把书中代码重新抄一遍，而是逐步形成三种能力：

1. 解释 GPT 各组件的数学含义与张量形状；
2. 不看答案独立写出可测试的 PyTorch 实现；
3. 像工程项目一样训练、记录、调试、保存和复现实验。

> 当前阶段：先完成 `chapters/00_training_basics`，再继续第 2 章。第 5 章的完整训练内容会在这个小模型闭环上逐步扩展。

## 第一次运行

### 1. 克隆并用 VS Code 打开

```bash
git clone https://github.com/JunFengXiang/llm-from-scratch-lab.git
cd llm-from-scratch-lab
code .
```

### 2. 创建独立环境

在 VS Code 中按 `Ctrl/Cmd + Shift + P`，选择 `Python: Create Environment` -> `Venv`；或者在终端执行：

```bash
python -m venv .venv
```

PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. 完成第一个训练闭环

```bash
python -m llm_from_scratch_lab.train_toy --config configs/toy_mlp.json --run-name first_run
```

训练完成后会生成：

```text
artifacts/
├── runs/first_run/
│   ├── config.json       # 本次超参数快照
│   ├── metrics.csv       # 可直接查看的逐轮指标
│   └── events.out...     # TensorBoard 日志
└── checkpoints/first_run/
    ├── last.pt           # 最近状态，用于断点续训
    └── best.pt           # 验证损失最低的状态
```

查看训练曲线：

```bash
tensorboard --logdir artifacts/runs
```

然后打开 `http://localhost:6006`。也可以直接在 VS Code 中打开 `metrics.csv`。

断点续训：

```bash
python -m llm_from_scratch_lab.train_toy \
  --config configs/toy_mlp.json \
  --run-name resumed_run \
  --resume artifacts/checkpoints/first_run/last.pt
```

Windows PowerShell 可把上面命令写成一行。

## 多设备维护

本仓库按多设备工作流设计。每台设备分别克隆仓库、创建自己的 `.venv` 和 `artifacts/`；GitHub 只同步源码、配置、测试、笔记和轻量实验结论。

开始工作前：

```bash
git pull --ff-only
```

离开当前设备前：

```bash
git status
git add <本次修改的文件>
git commit -m "feat(ch02): describe the change"
git push -u origin HEAD
```

上面的 `git add` 是格式说明，请替换为实际文件路径，不要原样复制。两台设备需要同时工作时，应使用不同的任务分支，不要同时修改同一分支和同一文件。完整首次设置、切换设备、冲突处理及 checkpoint 迁移方法见 [多设备协作工作流](docs/MULTI_DEVICE_WORKFLOW.md)。

## 第一个工程门禁

先尝试只拟合一个 batch：

```bash
python -m llm_from_scratch_lab.train_toy \
  --config configs/toy_mlp.json \
  --run-name overfit_test \
  --overfit-one-batch
```

通过标准：loss 明显下降，准确率接近 100%，并能说明下面六行代码的作用：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
```

然后运行质量检查：

```bash
python -m ruff check .
python -m pytest
```

## 学习路线

| 阶段 | 对应内容 | 必须产出 | 门禁 |
|---|---|---|---|
| 00 | 训练工程基础 | MLP 训练、日志、checkpoint、resume | 单 batch 过拟合 + 测试通过 |
| 01 | 理解大语言模型 | 概念图与形状流 | 能解释预训练与两类微调 |
| 02 | 文本数据 | tokenizer、滑动窗口、DataLoader | encode/decode 与 batch 测试 |
| 03 | 注意力 | 单头、因果、多头注意力 | 与参考实现数值对齐 |
| 04 | GPT 模型 | LayerNorm、GELU、Block、GPT | 前向形状与参数量测试 |
| 05 | 预训练 | loss、生成、训练与权重加载 | 可复现的小型预训练实验 |
| 06 | 分类微调 | 分类头、数据与评估 | 独立的测试集报告 |
| 07 | 指令微调 | 指令模板、collate、评估 | 保存回复并完成误差分析 |

完整验收标准见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 仓库结构

```text
chapters/       # 每阶段目标、练习、证据和复盘
configs/        # 可版本控制的实验配置
docs/           # 路线、工程规范和调试清单
experiments/    # 实验记录模板，不保存大体积产物
src/            # 可复用、可测试的实现
tests/          # 自动化测试
.vscode/        # 训练、测试和 TensorBoard 快捷任务
```

## 每个知识点的工作方式

```text
读懂概念 -> 闭卷实现 -> 写测试 -> 跑实验 -> 看日志 -> 写复盘
```

每次只改变一个主要变量；配置、指标和结论必须能够互相对应。详细规范见 [docs/ENGINEERING_GUIDE.md](docs/ENGINEERING_GUIDE.md)。训练异常时按 [docs/DEBUGGING_CHECKLIST.md](docs/DEBUGGING_CHECKLIST.md) 从数据到参数更新逐层检查。

## 版权与数据说明

本仓库只保存个人笔记、独立实现和小型教学示例，不上传原书 PDF、受版权保护的数据集、训练日志或模型权重。书籍章节只用于组织学习路线。

## License

[MIT](LICENSE)
