---
name: run-exp
description: 跑实验的标准流程：冒烟测试、脱离会话运行、监控、完整性和一致性检查（20% 抽查，不一致时提高到 100%）、汇总和记录。在 GPU 上跑任何正式实验（调参、对比、回归、尺度统计）时使用。
---

# 跑实验

服务器内存有坏位（见 CLAUDE.md），所以每个正式实验都要走完下面的流程。只要有一步结果不对，就停下来告诉用户，不要继续往下跑。

## 0. 开始前

- 向用户简单说明：为什么跑、跑什么（方法、数据集划分、episode 数、关键配置），预计多久。等用户同意后再开始，除非这一步在已批准的计划里。
- 运行名统一用 `p<阶段>_<内容>`，例如 `p4lambda_1`。
- 用 `bash .claude/skills/progress/status.sh` 检查：没有我们自己的实验在跑；GPU 显存够用（需要约 10 GB，别人在用就等或者问用户）。

## 1. 冒烟测试（1 个 episode）

```bash
source ~/yeen/vlfm_env.sh
$VLFM_PYTHON scripts/eval_multi_robot.py --method <method> --split <split> --limit 1 --dup-fraction 0 \
    --name <run>_smoke --keep-servers <overrides> > outputs/eval/<run>_smoke.log 2>&1
```

检查：最后一行是 "1 of 1 episodes done"；JSON 里有这次实验要用的字段（比如新加的统计）；日志里没有异常。冒烟测试不算结果。

## 2. 脱离会话运行

1. 复制 `.claude/skills/run-exp/launch_template.sh` 到 `outputs/eval/run_<name>.sh`，填好 `RUNS`（一行一个运行：`名字 方法 划分 额外参数`）。一致性抽查默认 20%（`--dup-fraction 0.2`），不要改小。
2. 启动：
   ```bash
   setsid nohup outputs/eval/run_<name>.sh > outputs/eval/run_<name>_driver.log 2>&1 < /dev/null &
   ```
3. 几秒后确认 `eval_multi_robot.py` 进程在跑。

## 3. 监控

用后台 Bash 循环等驱动日志出现 "ALL DONE"（单次最多约 110 分钟，到时没跑完就重新挂一个）：

```bash
end=$(( $(date +%s) + 6600 )); until grep -q "ALL DONE" outputs/eval/run_<name>_driver.log || \
  ! pgrep -f "[r]un_<name>.sh" >/dev/null || [ $(date +%s) -ge $end ]; do sleep 60; done; cat outputs/eval/run_<name>_driver.log
```

用户问进度时用 `/progress`。

## 4. 完整性和一致性检查

对每个运行读 `outputs/eval/<name>/integrity.json`：

- `unfinished_main` / `unfinished_dup` 不为空：用同样的命令再跑一次（脚本会从断点继续，只补缺的）。
- `hw_corruption`、`missing_after_run`：记下次数，这些已被自动重跑。看对应日志，确认是内存坏位（下标大得离谱、`Corrupted index`），而不是代码问题。
- **抽查出现不一致**（`duplicate_mismatches` 不为空）：
  1. 保存这次的记录：`cp outputs/eval/<name>/integrity.json outputs/eval/<name>/integrity_dup20.json`。
  2. 用同一个命令、`--dup-fraction 1.0` 再跑。已有的重复运行会保留，只补跑其余 episode，最后所有 episode 都有两次运行的对比。
  3. 不一致的 episode 会自动跑第三次来裁决。如果 100% 检查后仍有无法裁决的（`unresolved`），停下来告诉用户，结果先不用。
- 确认驱动日志里 "ports after stop: 0"，server 已关闭。

## 5. 汇总

- 对比实验用 `scripts/summarize_multi_robot.py`，参数是 `名字=outputs/eval/<run>/main`，输出存到 `outputs/eval/<name>_summary.txt`。
- 非对比实验（例如尺度统计）写一段小脚本直接读 `main/*.json`。
- 只用 `main/` 里的结果，不要把 `dup/` 也算进去。

## 6. 汇报和记录

- 硬件验收通过之前（见 CLAUDE.md 硬件一节），所有结果都是**初步**结果：汇报的标题或第一句、表格标题、写进设计文档的结果都标"初步"。
- 汇报（中文）：结论在前，然后是表格；用通俗语言解释每个关键指标的含义，并说明结论的依据：差别有多大、是否显著（置信区间是否跨过 0、McNemar 的 p 值）、样本量多少。说明完整性情况：重跑了几次、抽查多少个、是否全部一致。
- 把结果（日期、数值、完整性情况）写进 `docs/multi_robot_design.md` 当前阶段的进度条目。
- 可以本地提交设计文档和相关代码（不用问）；结果文件在 `outputs/`，不入库；推送等 `/milestone`。
