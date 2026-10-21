# RUN-0060 报告：hpcgpu07，W1 冒烟测试（9B，devtune）

任务书 `prompts/server_hpc/0060-hpc-07-w1-smoke.md`（origin 7235f13c；脚本 08ba07cd）。只用 devtune，没有碰 devcheck。没有碰 06 的 RUN-0056。

## 启动

- 10-21 19:07:15Z RUN-0054 写 `done`（`eval rc=0`，`done 0054-eval (failed steps: 0)`）。19:11:47Z 起经 hpcmgt 上脱离会话的 `$VI/tmp/start0060.sh`（输出 `$VI/logs/start0060.txt`）：
  - squeue 两条 RUNNING；07 上没有 0054、0057、0060 的进程，GPU 上没有计算进程，没有 07 的暂停文件。
  - `git fetch` 这一次失败（rc=128：`Connection to 20.205.243.166 port 22 timed out`）；工作树建在本地已有的 origin/main **2a7990ac4a78bede427a9f44fb6044944e44b446**。事后（19:3xZ）在开发机上 fetch，GitHub 上的 origin/main 就是 2a7990ac。
  - wt0060：`worktree add --no-checkout --detach … origin/main`、`sparse-checkout set`、`read-tree -mu HEAD` 都 rc=0；status 空；`experiments/` 只有 P0；包含 08ba07cd。
  - sha256，6 个都与任务书相同：smoke.sh `be100d15…`、w1.json `6a470459…`、w1_splits.json `a1371851…`、run_w1.py `bacb38e4…`、w1_gate.py `0826f782…`、eval_lib.sh `2380464b…`。
- 19:13:41Z 按任务书命令启动 smoke.sh（`EV_CAP_H=3`），step 157297.91；19:14:42Z squeue 两条 RUNNING。
- 推送循环 0060：19:14:42Z 启动，实际 PID 3917576。19:16Z – 20:19Z 几轮 `push failed (will retry)`（同一时段 0054、0056 的循环也是）；20:24Z 0057 的循环已能推送。
- 心跳：`2026-10-21T19:43:44Z done 0060 (failed steps: 0)`。从启动到结束 30 min，上限 3 h。

## steps.log 原文

```
2026-10-21T19:13:42Z start commit 2a7990ac4a78bede427a9f44fb6044944e44b446 node hpcgpu07 model qwen3.5-9b
2026-10-21T19:30:17Z full_refseg_devtune32 rc=0 wall_s=995
2026-10-21T19:30:40Z baseline_e1_refexp_box_devtune32 rc=0 wall_s=23
2026-10-21T19:30:44Z baseline_ownbox_devtune32 rc=0 wall_s=4
2026-10-21T19:31:09Z baseline_poly_structured_devtune32 rc=0 wall_s=25
2026-10-21T19:31:52Z w1f_devtune32 rc=0 wall_s=43
2026-10-21T19:42:05Z full_refexp_box_devtune16 rc=0 wall_s=613
2026-10-21T19:43:41Z full_count_devtune_countbench16 rc=0 wall_s=96
2026-10-21T19:43:44Z gate_rehearsal rc=0 wall_s=3
2026-10-21T19:43:44Z done; failed steps: 0
2026-10-21T19:43:44Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W1 smoke done (failed steps: 0)
```

## 各步统计（由 `experiments/W1/0060_w1_smoke_probe/p60_stats.py` 从结果文件读出）

显存：`gpu_mem.csv` 每秒一行（nvidia-smi 的 memory.used，时间是节点本地 CST），按 `steps.log` 的各步起止取窗口里的最大值。

```
gpu_mem.csv: 1801 samples; overall max 28087 MiB

### count/full/devtune_countbench: n=16
- failure: None: 16
- ended_by: submission: 16
- n_rounds: 1: 10, 2: 6
- n_tool_calls: 0: 10, 1: 6
- n_refused total 0; n_tool_call_errors total 0
- generated tokens per sentence: median 447, max 777
- thinking_end over rounds: natural: 22
- tool names called (from round text): {'submit_count': 16, 'tile_view': 3, 'zoom': 3}
- batch_rows over (sentence, round): 6: 6, 8: 16
- per-sentence wall_time_s: median 3.9, max 6.5; tool_time_s median 0.00; chunks [0]
- compute.json: n_calls 22, generated_tokens 7033, prompt_tokens 123599, call_time_s 77.4, wall_time_s 77.5

### refexp_box/baseline_e1/devtune: n=32
- failure: None: 31, multiple_boxes: 1
- ended_by: None: 32
- per-sentence wall_time_s: median 0.1, max 0.1; tool_time_s median 0.00; chunks [None]
- compute.json: n_calls 32, generated_tokens 1499, prompt_tokens 11025, call_time_s 4.6, wall_time_s 4.6

### refexp_box/full/devtune: n=16
- failure: None: 12, no_answer: 4
- ended_by: round_cap: 4, submission: 12
- n_rounds: 1: 1, 2: 3, 3: 1, 4: 1, 5: 1, 6: 9
- n_tool_calls: 0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 5, 6: 4
- n_refused total 0; n_tool_call_errors total 0
- generated tokens per sentence: median 1772, max 2899
- thinking_end over rounds: forced: 2, natural: 71
- tool names called (from round text): {'zoom': 37, 'submit_box': 12, 'read_pixels': 9, 'grabcut': 4, 'show': 4, 'tile_view': 3, 'draw_box': 2, 'watershed': 1, 'edges': 1}
- batch_rows over (sentence, round): 1: 1, 4: 4, 5: 20, 6: 18, 7: 14, 8: 16
- per-sentence wall_time_s: median 45.6, max 52.5; tool_time_s median 0.01; chunks [0]
- compute.json: n_calls 73, generated_tokens 27297, prompt_tokens 502113, call_time_s 587.6, wall_time_s 592.0

### refseg/baseline_ownbox/devtune: n=32
- failure: None: 31, multiple_boxes: 1
- ended_by: None: 32
- compute.json: n_calls 32, generated_tokens 1499, prompt_tokens 11025, call_time_s 4.6, wall_time_s 4.6

### refseg/baseline_poly_structured/devtune: n=32
- failure: None: 32
- ended_by: None: 32
- per-sentence wall_time_s: median 0.2, max 0.2; tool_time_s median 0.00; chunks [None]
- compute.json: n_calls 32, generated_tokens 1346, prompt_tokens 12881, call_time_s 6.9, wall_time_s 6.9

### refseg/full/devtune: n=32
- failure: None: 2, no_answer: 30
- ended_by: round_cap: 30, submission: 2
- n_rounds: 6: 32
- n_tool_calls: 5: 2, 6: 30
- n_refused total 0; n_tool_call_errors total 0
- generated tokens per sentence: median 1420, max 2347
- thinking_end over rounds: natural: 192
- tool names called (from round text): {'zoom': 48, 'show': 37, 'grabcut': 27, 'superpixels': 19, 'select_superpixels': 16, 'tile_view': 13, 'edges': 7, 'refine': 6, 'components': 6, 'read_pixels': 4, 'draw_box': 3, 'run_python': 2, 'submit_mask': 2, 'watershed': 1, 'combine_masks': 1}
- batch_rows over (sentence, round): 1: 1, 2: 2, 3: 3, 4: 8, 5: 50, 6: 54, 7: 42, 8: 32
- per-sentence wall_time_s: median 27.0, max 42.0; tool_time_s median 1.22; chunks [0]
- compute.json: n_calls 192, generated_tokens 47440, prompt_tokens 1334206, call_time_s 851.9, wall_time_s 895.5

### refseg/w1f/devtune: n=32
- failure: None: 31, multiple_boxes: 1
- ended_by: None: 32
- per-sentence wall_time_s: median 0.6, max 12.7; tool_time_s median 0.00; chunks [0]
- compute.json: n_calls 0, generated_tokens 0, prompt_tokens 0, call_time_s 0.0, wall_time_s 39.0

- full_refseg_devtune32: rc=0 wall_s=995; gpu_mem peak in window 28087 MiB (995 samples)
- baseline_e1_refexp_box_devtune32: rc=0 wall_s=23; gpu_mem peak in window 23123 MiB (23 samples)
- baseline_ownbox_devtune32: rc=0 wall_s=4; gpu_mem peak in window 0 MiB (4 samples)
- baseline_poly_structured_devtune32: rc=0 wall_s=25; gpu_mem peak in window 23103 MiB (25 samples)
- w1f_devtune32: rc=0 wall_s=43; gpu_mem peak in window 4 MiB (43 samples)
- full_refexp_box_devtune16: rc=0 wall_s=613; gpu_mem peak in window 27765 MiB (612 samples)
- full_count_devtune_countbench16: rc=0 wall_s=96; gpu_mem peak in window 27765 MiB (96 samples)
- gate_rehearsal: rc=0 wall_s=3; gpu_mem peak in window 0 MiB (3 samples)
```

- **full_refseg_devtune32**（第一步）：墙钟 995 s；32 句里 2 句提交（`ended_by: submission`），30 句 `failure: no_answer`（`ended_by: round_cap`，6 轮用完没有 submit_mask）；轮数 32 句都是 6；工具调用 5 次 2 句、6 次 30 句；拒收 0、工具调用出错 0；每句生成 token 中位 1420、最大 2347；每句墙钟中位 27.0 s、最大 42.0 s；子批（batch_rows）1–8；显存峰值 28,087 MiB。没有 out of memory。
- **full_refexp_box_devtune16**：16 句里 12 句提交，4 句 no_answer（round_cap）。
- **full_count_devtune_countbench16**：16 句都提交。

## devtune 读数（调参集，照贴）

| 步 | 指标 | 值 | 95% CI | n |
|---|---|---|---|---|
| refseg / full（W1-1） | cIoU | 0.1585 | [0.0, 0.3794] | 32 |
| refseg / full（W1-1） | gIoU | 0.0540 | [0.0, 0.1351] | 32 |
| refseg / baseline_ownbox | cIoU | 0.4869 | [0.3718, 0.6095] | 32 |
| refseg / baseline_ownbox | gIoU | 0.5531 | [0.4826, 0.6146] | 32 |
| refseg / baseline_poly_structured | cIoU | 0.4032 | [0.3240, 0.4773] | 32 |
| refseg / baseline_poly_structured | gIoU | 0.4236 | [0.3485, 0.4977] | 32 |
| refseg / w1f | cIoU | 0.5177 | [0.3818, 0.6281] | 32 |
| refseg / w1f | gIoU | 0.5785 | [0.4686, 0.6677] | 32 |
| refexp_box / baseline_e1 | acc@0.5 | 0.9062 | [0.7812, 1.0] | 32 |
| refexp_box / full | acc@0.5 | 0.3750 | [0.125, 0.625] | 16 |
| count / full（countbench） | accuracy | 0.875 | [0.6875, 1.0] | 16 |

## gate_rehearsal.md 原文

# W1 去留判定（D103 §三，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devtune，n = 32；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck': False, 'n_is_500': False, 'not_tiny': True, 'no_limit': False}

| 读数 | 值 |
|---|---|
| W1-1（full）cIoU | 0.1585 |
| 自己的框 cIoU | 0.4869 |
| E1 结构强制多边形 cIoU | 0.4032 |

1. W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0：差 -0.3284，95% CI [-0.5078, -0.1776] → **不满足**
2. W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）：差 -0.2447（-14081225978/57546048203）→ **不满足**

**结论：彩排，不作数（规则给出：放弃）**

## 报错

- 没有 traceback；8 步都 rc=0。

## 之后

- 07 按顺序接 RUN-0057（见 `reports/0057-hpc-07-e5-posthoc4000.md`）。
- 推送循环 0060：等最后一轮推完再按 PID 停（见下）。
