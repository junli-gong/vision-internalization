# RUN-0063 报告：hpcgpu07，W1 去留判定的开发-检查集运行（D103 §三；D111 冻结的协议）

任务书 `prompts/server_hpc/0063-hpc-07-w1-devcheck.md`（origin 06d69645）。没有碰 06 的 RUN-0056。只贴数字，不解读。

## 启动与让路

- 10-22 09:28:22Z（`$VI/tmp/start0063.sh`，输出 `$VI/logs/start0063.txt`）：squeue 两条 RUNNING；RUN-0061、RUN-0062 心跳都是 done、failed steps 0；07 没有暂停文件。
- fetch：第 1–3 次 rc=128（`Connection to 20.205.243.166 port 22 timed out`），第 4 次 09:40:05Z rc=0。拿不到含 06d69645 的 origin/main 就不建工作树。
- wt0063：HEAD `95d9fc51e8eda8777f2b41560ee43118dccbbf7a`，含 06d69645，status 空，`experiments/` 只有 P0。12 个 sha256 逐字与任务书相同（devcheck.sh `739e91d0…`、w1.json `e4cc8419…` 等）；`configs/w1.json` 的 `frozen` 为 True；devcheck.sh 用本节点暂停文件。
- 09:40:15Z 启动 devcheck.sh（`EV_CAP_H=8 NODE=hpcgpu07 TQ_STATUS=train_0057-hpcgpu07.status`），step 157297.96；心跳 `09:40:46Z waiting for gpu (running: train_0057-hpcgpu07.status; pids: 1927632)`；暂停文件内容 `0063-hpc-w1-devcheck.md`。推送循环 0063：PID 234951。
- RUN-0057 在 `eval_e3b_c3_n4000_s0` 这一步之后停住：`09:42:06Z eval_e3b_c3_n4000_s0 完成`、`09:42:06Z paused (pause file: 0063-hpc-w1-devcheck.md )`。
- RUN-0063 09:42:16Z 拿到 GPU；15:16:34Z `done 0063 (failed steps: 0)`；用 20,058 s（5.57 h），上限 8 h。
- 交还：15:18Z 查，`train.pause.hpcgpu07` 不存在；RUN-0057 心跳 `2026-10-22T15:17:01Z running c3_warmup_n4000_s0 (chunk 1)`；squeue 两条 RUNNING。
- gate 一步跑完（14:16:13Z）后立即把 `gate.md` 原文发给了管理员。

## steps.log 原文

```
2026-10-22T09:42:16Z start commit 95d9fc51e8eda8777f2b41560ee43118dccbbf7a node hpcgpu07 model qwen3.5-9b w1.json e4cc841907284009b0d5ea51c075b3d52699eacb3c624207b084bc9e4952f928
2026-10-22T09:43:48Z baseline_e1_refexp_box_devcheck rc=0 wall_s=92
2026-10-22T09:43:52Z baseline_ownbox_devcheck rc=0 wall_s=4
2026-10-22T09:49:41Z baseline_poly_structured_devcheck rc=0 wall_s=349
2026-10-22T09:57:05Z w1f_devcheck rc=0 wall_s=444
2026-10-22T14:16:04Z full_refseg_devcheck rc=0 wall_s=15539
2026-10-22T14:16:13Z gate rc=0 wall_s=9
2026-10-22T15:04:13Z full_refexp_box_devtune64 rc=0 wall_s=2880
2026-10-22T15:16:34Z full_count_devtune_countbench49 rc=0 wall_s=741
2026-10-22T15:16:34Z done; failed steps: 0
```

以下由 `experiments/W1/0063_w1_devcheck_probe/p63_tables.py` 从结果文件读出。显存是 nvidia-smi 的 memory.used，按 `steps.log` 的各步起止取窗口里的最大值；只用 CPU 的 `w1f_devcheck` 窗口里读到 28,529 MiB，是窗口包含了上一步占着的显存（窗口取法如此，照录）。没有 traceback。

## 回路各次（完整工作台）

### W1-1：refseg / full / devcheck

| 项 | 值 |
|---|---|
| 句数 | 500 |
| 提交（ended_by = submission） | 442 |
| ended_by 分布 | round_cap: 58, submission: 442 |
| failure 分布 | no_answer: 58, wrong_submission: 15, None: 427 |
| 轮数分布 | 3: 1, 4: 2, 5: 5, 6: 492 |
| n_tool_calls 分布 | 2: 2, 3: 2, 4: 7, 5: 486, 6: 1, 7: 2 |
| 每轮调用个数的分布（轮数） | 0: 6, 1: 2976, 2: 6 |
| 被拒调用按原因 | {'final_round': 32, 'after_submission': 1}（n_refused 合计 33） |
| 工具调用出错（n_tool_call_errors 合计） | 6 |
| 每句生成 token 中位 / 最大 | 1694.5 / 6221 |
| 标题读数 | cIoU 0.4696 [0.4367, 0.5002] n=500; gIoU 0.4337 [0.4103, 0.4604] n=500 |
| tools_sha256 | dcc91402… |

### 定位：refexp_box / full / devtune

| 项 | 值 |
|---|---|
| 句数 | 64 |
| 提交（ended_by = submission） | 63 |
| ended_by 分布 | round_cap: 1, submission: 63 |
| failure 分布 | no_answer: 1, None: 63 |
| 轮数分布 | 1: 5, 2: 10, 3: 9, 4: 4, 5: 5, 6: 31 |
| n_tool_calls 分布 | 0: 5, 1: 10, 2: 9, 3: 4, 4: 6, 5: 30 |
| 每轮调用个数的分布（轮数） | 0: 1, 1: 278 |
| 被拒调用按原因 | 无（n_refused 合计 0） |
| 工具调用出错（n_tool_call_errors 合计） | 1 |
| 每句生成 token 中位 / 最大 | 1938.5 / 10846 |
| 标题读数 | acc@0.5 0.6250 [0.5000, 0.7344] n=64 |
| tools_sha256 | dcc91402… |

### 计数：count / full / devtune_countbench

| 项 | 值 |
|---|---|
| 句数 | 49 |
| 提交（ended_by = submission） | 49 |
| ended_by 分布 | submission: 49 |
| failure 分布 | None: 49 |
| 轮数分布 | 1: 27, 2: 10, 3: 2, 4: 2, 5: 3, 6: 5 |
| n_tool_calls 分布 | 0: 27, 1: 10, 2: 2, 3: 2, 4: 3, 5: 5 |
| 每轮调用个数的分布（轮数） | 1: 106 |
| 被拒调用按原因 | 无（n_refused 合计 0） |
| 工具调用出错（n_tool_call_errors 合计） | 0 |
| 每句生成 token 中位 / 最大 | 488.0 / 4556 |
| 标题读数 | accuracy 0.8367 [0.7347, 0.9388] n=49 |
| tools_sha256 | dcc91402… |

## 各输出的标题读数

| 输出 | 标题读数 |
|---|---|
| count/full/devtune_countbench | accuracy 0.8367 [0.7347, 0.9388] n=49 |
| refexp_box/baseline_e1/devcheck | acc@0.5 0.9060 [0.8780, 0.9320] n=500 |
| refexp_box/full/devtune | acc@0.5 0.6250 [0.5000, 0.7344] n=64 |
| refseg/baseline_ownbox/devcheck | cIoU 0.5074 [0.4858, 0.5289] n=500; gIoU 0.5229 [0.5057, 0.5409] n=500 |
| refseg/baseline_poly_structured/devcheck | cIoU 0.3463 [0.3206, 0.3706] n=500; gIoU 0.3511 [0.3297, 0.3715] n=500 |
| refseg/full/devcheck | cIoU 0.4696 [0.4367, 0.5002] n=500; gIoU 0.4337 [0.4103, 0.4604] n=500 |
| refseg/w1f/devcheck | cIoU 0.5757 [0.5435, 0.6084] n=500; gIoU 0.5512 [0.5250, 0.5791] n=500 |

## 显存（gpu_mem.csv：19929 行，全程最大 28805 MiB）

| 步 | rc | 墙钟 s | 窗口内显存峰值 MiB |
|---|---|---|---|
| baseline_e1_refexp_box_devcheck | 0 | 92 | 27287 |
| baseline_ownbox_devcheck | 0 | 4 | 4 |
| baseline_poly_structured_devcheck | 0 | 349 | 28529 |
| w1f_devcheck | 0 | 444 | 28529 |
| full_refseg_devcheck | 0 | 15539 | 28805 |
| gate | 0 | 9 | 0 |
| full_refexp_box_devtune64 | 0 | 2880 | 28005 |
| full_count_devtune_countbench49 | 0 | 741 | 28005 |

## gate.md 原文

# W1 去留判定（D103 §三，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devcheck，n = 500；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck': True, 'n_is_500': True, 'not_tiny': True, 'no_limit': True}

| 读数 | 值 |
|---|---|
| W1-1（full）cIoU | 0.4696 |
| 自己的框 cIoU | 0.5074 |
| E1 结构强制多边形 cIoU | 0.3463 |

1. W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0：差 -0.0378，95% CI [-0.0694, -0.0060] → **不满足**
2. W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）：差 0.1233（13239094038349/107393511432240）→ **不满足**

**结论：放弃**

