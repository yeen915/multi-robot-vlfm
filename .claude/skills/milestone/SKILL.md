---
name: milestone
description: 阶段收尾：跑全部单元测试、单机回归、核对验收标准、更新设计文档的进度、检查仓库里没有大文件和密钥，然后提交、推送到 backup、打 tag。一个阶段（或阶段中一块大的工作）完成、用户要求推送时使用。
---

# 阶段收尾

每一步做完简单汇报。任何一步失败就停下来，说明原因，不要跳过。推送和打 tag 之前要用户确认。

## 1. 单元测试

```bash
source ~/yeen/vlfm_env.sh
$VLFM_PYTHON scripts/dev/run_tests.py
```

要求全部通过。报告通过数，和上一个阶段比多了哪些测试。

## 2. 单机回归（这个阶段改了策略、地图或任务代码时）

改动不能改变单机行为：在基准的 20 个 episode 上跑单机，并和基准逐个比较。

- 按 `/run-exp` 跑：方法 `single`，场景 5cdEh9F2hJL 的前 20 个 episode，不录视频。基准在 `~/yeen/results/base_novideo`。
- 比较时忽略基准之后新增的字段（目前有 `total_actions`、`team_stats`、`coverage_curve`、`utility_scales`，包括嵌在 `multi_robot` 里的），其余字段必须完全相同。
- 要求 20/20 一致。不一致时先查是不是内存坏位（重跑那个 episode）；重跑后还不一致，说明行为变了，停下来报告。

## 3. 验收标准

读 `docs/multi_robot_design.md` 第 5 节这个阶段的"验收"条目，逐条核对：已满足（附依据，比如测试名、结果数值、视频帧路径）、未满足、不适用。有未满足的就问用户：先补，还是记为遗留问题。

## 4. 更新设计文档

- 这个阶段的进度条目：已完成的打勾，写上日期和关键结果（数值、完整性情况）。
- 新发现的局限、改过的决定写进对应章节（决定写在第 6 节）。
- 下一阶段的待办列清楚。

## 5. 仓库检查（推送前必做）

```bash
git status --short
git diff --cached --stat
# 大文件：新增或修改的文件超过 1 MB
git diff --name-only backup/multi-robot..HEAD | xargs -r ls -la 2>/dev/null | awk '$5 > 1000000'
git ls-files --others --exclude-standard | xargs -r ls -la 2>/dev/null | awk '$5 > 1000000'
# 密钥和认证信息：只看新增的行（以 + 开头，去掉 +++ 文件头）。关键词要整词出现并紧跟 = 或 :，
# 所以 tokenizer、num_tokens、secret_sauce 这类正常代码不会命中
git diff backup/multi-robot..HEAD | grep -E '^\+' | grep -vE '^\+\+\+ ' | grep -niE \
  'BEGIN [A-Z ]*PRIVATE KEY|ssh-(rsa|ed25519|ecdsa)[a-z0-9-]* AAAA|tskey-[a-z]+-|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}|login\.tailscale\.com/a/|\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|password|passwd|secret)\b"?\s*[:=]\s*[^[:space:]]{8,}' | head
```

- 不能进仓库：`docs/` 里的 CoRelNav PDF、`lockfiles/`、权重、视频、场景数据、日志、`outputs/`，以及 `~/yeen/.ssh`、`~/yeen/tailscale/state` 里的任何东西。
- 有可疑项就停下来给用户看。

## 6. 提交、推送、打 tag

1. 提交剩下的改动（英文说明，结尾 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`）。
2. 给用户看 `git log --oneline backup/multi-robot..HEAD`（要推送哪些提交）。**确认后**推送：`git push backup multi-robot`。
3. 整个阶段完成时，**确认后**打 tag `phase<N>-done` 并推送：`git tag phase<N>-done && git push backup phase<N>-done`。
4. 确认 server 已关闭（`/progress`）。

## 7. 汇报

简短总结这个阶段：做了什么、关键结果（用通俗语言解释指标）、完整性情况、遗留问题、下一阶段要做什么。下一阶段等用户确认后再开始。
