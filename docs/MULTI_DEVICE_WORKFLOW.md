# 多设备协作工作流

本仓库允许在 Windows、macOS 或 Linux 设备之间切换。GitHub 是源码和学习进度的唯一同步中心；每台设备保留自己的 Python 环境和训练产物。

## 同步边界

会通过 Git 同步：

- `src/`、`tests/` 和 `configs/`；
- 章节练习、Markdown 笔记和实验结论；
- `pyproject.toml`、共享 VS Code 配置和 CI 配置。

只保存在各设备本地：

- `.venv/`；
- `artifacts/`、TensorBoard 日志和 checkpoints；
- 数据集、原书 PDF、缓存和 `.env`；
- 设备相关的 VS Code 用户设置。

不要使用 OneDrive、iCloud Drive 或其他网盘直接同步整个仓库目录。Git 与网盘同时改写同一目录，容易产生锁文件、重复文件和未提交内容冲突。

## 每台设备首次设置

在每台设备分别克隆仓库，不要复制另一台设备的整个项目目录：

```bash
git clone https://github.com/JunFengXiang/llm-from-scratch-lab.git
cd llm-from-scratch-lab
git config core.autocrlf false
git config pull.ff only
git config fetch.prune true
```

两台设备尽量使用同一个 Python 次版本；本仓库以 Python 3.11 和 CI 结果为基准。

然后在每台设备独立创建环境：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell 激活命令：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活命令：

```bash
source .venv/bin/activate
```

## 日常工作：开始前

先确认当前设备没有遗漏的本地改动：

```bash
git status
```

开始一个新任务：

```bash
git switch main
git pull --ff-only
git switch -c learn/ch02-tokenizer
```

继续一个已经在另一台设备推送过的任务：

```bash
git fetch origin
git switch --track origin/learn/ch02-tokenizer
```

如果当前设备已经有这个分支：

```bash
git switch learn/ch02-tokenizer
git pull --ff-only
```

分支按任务命名，例如：

- `learn/training-basics`
- `learn/ch02-tokenizer`
- `learn/ch03-causal-attention`
- `fix/checkpoint-resume`

## 日常工作：离开设备前

不要只保存文件就换设备。必须把一个可说明的阶段提交并推送：

```bash
git status
git add -A
git status
git commit -m "feat(ch02): implement tokenizer"
git push -u origin HEAD
```

第二次 `git status` 用来确认暂存区里只有当前任务的文件。工作尚未完成时可以提交一个临时同步点：

```bash
git commit -m "wip(ch02): sync tokenizer progress"
git push -u origin HEAD
```

未提交的改动和 `git stash` 都只存在于当前设备，不会自动出现在另一台设备上。

## 两台设备同时工作

不要让两台设备同时修改同一分支、同一文件。把工作拆成两个任务分支，例如：

- 设备 A：`learn/ch02-tokenizer`，实现 tokenizer；
- 设备 B：`learn/ch02-tests`，补数据集和形状测试。

每个任务完成后通过 Pull Request 合并到 `main`。合并后，两台设备都执行：

```bash
git switch main
git pull --ff-only
```

## 冲突时的安全处理

如果 `git pull --ff-only` 失败，说明本地和远程已经分叉。不要执行 `git reset --hard`，也不要强制推送。先收集状态：

```bash
git status
git fetch origin
git log --oneline --graph --decorate --all -12
```

保存这三条命令的输出，再决定是 rebase、合并还是保留两个分支。对 Git 尚不熟悉时，先停止修改并在 Issue 或对话中说明两台设备分别做了什么。

## 训练产物与断点续训

训练日志和权重默认不进入 Git，所以代码会同步，`last.pt` 不会同步。若要把一次训练从一台设备迁移到另一台设备，应单独传输对应的 checkpoint 和配置，放回同样的 `artifacts/checkpoints/<run-name>/` 路径。

不要把大模型权重直接提交到普通 Git 历史。需要长期共享大量训练产物时，再单独引入对象存储或实验追踪平台。
