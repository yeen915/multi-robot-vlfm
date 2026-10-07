# 双机器人 VLFM 项目

在原版 VLFM 上做两台机器人协作的 ObjectNav。分支 `multi-robot`，设计和进度都在 `docs/multi_robot_design.md`：第 5 节"阶段 0–6"是实现计划和进度清单（唯一的进度记录），第 6 节是已确认的决定。

## 沟通

- 用中文回复。
- 每一步先简单解释原因和打算怎么改，再动手。每个阶段完成后等我确认再进入下一阶段。
- 我正在学习这个项目。汇报结果时用通俗语言解释关键指标的含义（例如 SR、SPL、同选率、置信区间、McNemar 检验）和结论的依据（哪个数说明了什么、差别是否显著）；涉及新概念时简单说明一句。
- 结果不好、实验失败、跳过了步骤，都如实说明。

## 服务器规则（多人共用）

- 每个新 shell 先运行 `source ~/yeen/vlfm_env.sh`，Python 用 `$VLFM_PYTHON`（pyenv 会遮住 conda 的 python）。
- 不用 sudo，不动 `~/yeen` 以外的目录。例外：可以修改 conda 环境 `vlfm`。不经我同意不装新包。pytest 没装，单元测试用 `$VLFM_PYTHON scripts/dev/run_tests.py` 运行（自带 pytest 替身）。
- GPU 是共用的：模型 server 只在跑实验时开，跑完运行 `~/yeen/stop_vlm.sh` 关掉，并确认 12181–12184 端口已关闭。
- tmux 会话 `setup` 是我的，不要动。
- 超过几分钟的任务用 `setsid nohup` 脱离会话运行，再用后台循环监控；不要依赖前台命令或会话内的后台任务（有 2 小时上限，会话结束也会中断）。

## 硬件：内存有坏位

服务器内存有硬件故障（2026-10-05 确认，管理员处理后复测仍有错误，详见 `~/yeen/memcheck/`）。表现是数组下标或数值里个别高位被翻转，进程崩溃或结果被悄悄改错。因此：

- 正式实验一律通过 `scripts/eval_multi_robot.py` 运行。它按 episode 写 JSON，坏结果自动重跑，并抽查重复运行的一致性。
- 一致性抽查默认 20%；出现任何不一致，把这次运行提高到 100% 再查（流程见 `/run-exp`）。
- 汇报结果时说明完整性检查的情况：重跑了几次、抽查是否全部一致。
- **内存修好之前的所有结果都是初步结果。** 汇报、周报、设计文档里，这段时间的结果都标"初步"。论文和正式表格里的数字，要等硬件验收通过（memtest86+ 连续 2 轮 0 错误）之后重跑。验收通过后在这里写上日期，之后的结果才不用标"初步"。

## Git

- 远程：推送只用 `backup`（`git@github.com:yeen915/multi-robot-vlfm.git`，deploy key 通过仓库本地的 `core.sshCommand` 指定）。不动 `origin`，不改 `~/.ssh/config`，不在这台机器上 `gh auth login`（共用账号）。
- 本地提交：每完成一块工作就可以提交，不用问我。推送：只在 `/milestone` 时或我要求时推送。
- 提交说明用英文，结尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- 不提交：`docs/` 里的 CoRelNav PDF、`lockfiles/`、权重、视频、场景数据、日志、`outputs/`。推送前检查没有大文件（> 1 MB）或密钥进入仓库。

## 常用命令

```bash
source ~/yeen/vlfm_env.sh
$VLFM_PYTHON scripts/eval_multi_robot.py --method <method> --split <tune|dev|eval> --name <run>   # 正式实验
$VLFM_PYTHON scripts/dev/run_tests.py                                                               # 单元测试
$VLFM_PYTHON scripts/summarize_multi_robot.py a=outputs/eval/<run_a>/main b=outputs/eval/<run_b>/main  # 对比表
bash ~/yeen/stop_vlm.sh                                                                             # 关 server
```

方法名见 `scripts/eval_multi_robot.py` 里的 `METHODS`。结果在 `outputs/eval/<run>/`（不入库）。单机基准：`~/yeen/results/base_novideo`（场景 5cdEh9F2hJL 前 20 个 episode，SR 35%，SPL 0.149）。

## Skills

- `/progress`：进度检查（后台实验、完整性、server、GPU、阶段清单）。
- `/run-exp`：跑实验的标准流程。
- `/milestone`：阶段收尾（测试、回归、更新设计文档、提交、推送、打 tag）。
- `/weekly`：给导师的中文周报。
