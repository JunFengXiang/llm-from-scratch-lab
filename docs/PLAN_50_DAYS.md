# 50天实践入口

周期：2026-10-27 至 2026-12-15；2026-10-26 为环境准备日。
工作日60分钟，周六日各120分钟。日历Day不代表已完成课次。

## 当前任务
[PY01-list-summary：把列表统计封装成函数](../chapters/00_training_basics/PY01-list-summary.md)。
截至2026-10-28，材料已准备，用户实现、运行结果和理解均待确认。
[进度记录](../study_progress.csv)只记录可见证据；没有推送不表示没有本地学习。

## 路线与门槛
1. Mac / CPU：Python列表、字典、函数、文件与调试；随后NumPy、pandas、Matplotlib。
   围绕小型CSV任务逐步完成读取、清洗、分组、画图和可重跑脚本。
2. 用户独立完成上述项目并解释结果后，接PyTorch张量、DataLoader、最小训练循环，再迁移5080。
3. 在5080上接续文本窗口、嵌入、注意力、Transformer Block、小型GPT及预训练、生成、保存恢复。
4. 保留最后8个日历日（12月8日至15日）用于补欠、独立重跑、短报告和诚实验收。
   CNN与进阶实验只在主线完成且时间允许时选做，不追加时长或自动延期。

## 仓库与设备
- 用户已有 [my_python_study](https://github.com/JunFengXiang/my_python_study)，Mac阶段练习可沿用其grammar目录。
- 当前自动写入授权仅覆盖本仓库的study/50days；此分支保存教学材料与进度链接，不自动修改Python仓库。
- 后续模型实现使用本仓库。两处工作属于同一条学习进度，不重复抄写代码。
- [多设备工作流](MULTI_DEVICE_WORKFLOW.md)：各机独立虚拟环境，不复制.venv；代码经GitHub同步。
- 当前纯Python练习不需要安装本仓库的训练依赖或配置GPU。
- 默认分支已有CI问题与[修复PR #2](https://github.com/JunFengXiang/llm-from-scratch-lab/pull/2)分开处理；
  2026-10-28检查时PR仍打开，其head的CI成功，本学习分支不合并该修复。

## 如何推进
一次仅一个未验收核心练习。提交代码或commit、实际输出、两三句解释/卡点。
区分材料已准备、用户已提交、程序通过检查、理解已验收。
缺少证据时继续同题；旧D01-token-windows保留待回归，不算已完成。
