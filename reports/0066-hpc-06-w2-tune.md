# RUN-0066 报告：hpcgpu06，W2-a 调参（qwen3.5-9b，devtune800）

任务书 `prompts/server_hpc/0066-hpc-07-w2-tune.md` + 补充 1 `0066-addendum-1.md`（改在 06 上立即开始）。报告名按节点改为 hpc-06。只贴数字，不解读。

## 启动

- 10-24 04:10Z 按补充 1 先停 07 的等待脚本：确认 `eval_0066.status` 不存在、07 上无 0066 进程（07 的 GPU 只有 RUN-0057 的训练），按 PID 停 hpcmgt 上的 `start0066.sh`（291006）及其 sleep。它只打印到 `prep OK; waiting for RUN-0057`，没有启动任何东西。
- 经 hpcmgt 上脱离会话的 `$VI/tmp/start0066b.sh`：wt0066 HEAD `130d2314f11c768c686419d3e4e4d11381d706ec`，12 个 sha256 与 10-22 准备时逐字相同；06 无暂停文件、GPU 空；`train_0056-hpcgpu06.status` 末行 `2026-10-23T23:29:53Z done 0056-hpcgpu06`。
- 04:11:18Z 按补充 1 的命令启动（`srun --jobid=157296 --overlap -n1 -c 8`，`EV_CAP_H=10 NODE=hpcgpu06 TQ_STATUS=train_0056-hpcgpu06.status`）。`-c 8` 被接受，未用 CW=1 的退路；step 157296.169，cw 8。
- 同步循环 TAG 0066（PID 605932，1800 s，路径 `experiments/W2/0066_w2_tune_hpcgpu06`）。前后 squeue 两条 RUNNING。

## steps.log 原文

```
2026-10-24T04:11:19Z start commit 130d2314f11c768c686419d3e4e4d11381d706ec node hpcgpu06 model qwen3.5-9b split devtune800 cw 8
2026-10-24T04:13:28Z s0_e1_box rc=0 wall_s=129
2026-10-24T04:21:06Z s0_w1f rc=0 wall_s=458
2026-10-24T04:21:10Z s0_ownbox rc=0 wall_s=4
2026-10-24T04:21:20Z s0_random_pick rc=0 wall_s=10
2026-10-24T04:21:29Z s0_oracle_pick rc=0 wall_s=9
2026-10-24T04:21:32Z s0_rank rc=0 wall_s=3
2026-10-24T04:30:31Z s1_grid_off rc=0 wall_s=539
2026-10-24T04:48:49Z s1_sep_off rc=0 wall_s=1097
2026-10-24T06:02:45Z s1_grid_on rc=0 wall_s=4436
2026-10-24T08:02:25Z s1_sep_on rc=0 wall_s=7180
2026-10-24T08:02:26Z step1 chosen sep_off
2026-10-24T08:14:55Z s2_sep_off_k4 rc=0 wall_s=726
2026-10-24T08:30:40Z s2_sep_off_k6 rc=0 wall_s=943
2026-10-24T08:30:40Z step2 chosen sep_off_k4
2026-10-24T08:36:44Z s3_sep_off_k4_pts rc=0 wall_s=362
2026-10-24T08:36:44Z step3 chosen sep_off_k4
2026-10-24T08:36:44Z done; failed steps: 0; chosen sep_off_k4
2026-10-24T08:36:44Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）W2 tuning done (chosen sep_off_k4)
```

- 心跳：`2026-10-24T08:36:44Z done 0066 (failed steps: 0; chosen sep_off_k4)`。总墙钟 04:11:19Z – 08:36:44Z（15,925 s，4.42 h），上限 10 h。

## step1.md 原文

```markdown
# W2 调参：第 1 步 选臂

- 规则：W2 预注册 §5：取减 W1-F 的配对 cIoU 差最大的臂；与最大者相差不到 0.005 的取最简单的（给出的顺序，越前越简单）
- n = 800；参照 cIoU：w1f 0.5724，random_pick 0.5274，oracle_pick 0.6470

| 臂（简单→复杂） | cIoU | 减 W1-F | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 |
|---|---:|---:|---:|---:|---:|---:|
| grid_off | 0.5193 | -0.0530 | 0.131 | 0.095 | 0.055 | 0.004 |
| sep_off | 0.5641 | -0.0083 | 0.165 | 0.926 | 0.011 | 0.911 |
| grid_on | 0.5476 | -0.0248 | 0.169 | 0.342 | 0.061 | 0.048 |
| sep_on | 0.5610 | -0.0113 | 0.228 | 0.345 | 0.008 | 0.018 |

**差最大**：sep_off；**选中**：sep_off
```

## step2.md 原文

```markdown
# W2 调参：第 2 步 选臂

- 规则：W2 预注册 §5：取减 W1-F 的配对 cIoU 差最大的臂；与最大者相差不到 0.005 的取最简单的（给出的顺序，越前越简单）
- n = 800；参照 cIoU：w1f 0.5724，random_pick 0.5274，oracle_pick 0.6470

| 臂（简单→复杂） | cIoU | 减 W1-F | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 |
|---|---:|---:|---:|---:|---:|---:|
| sep_off_k4 | 0.5721 | -0.0002 | 0.206 | 0.958 | 0.071 | 0.852 |
| sep_off_k6 | 0.5646 | -0.0078 | 0.165 | 0.933 | 0.023 | 0.901 |
| sep_off | 0.5641 | -0.0083 | 0.165 | 0.926 | 0.011 | 0.911 |

**差最大**：sep_off_k4；**选中**：sep_off_k4
```

## step3.md 原文

```markdown
# W2 调参：第 3 步 选臂

- 规则：W2 预注册 §5：取减 W1-F 的配对 cIoU 差最大的臂；与最大者相差不到 0.005 的取最简单的（给出的顺序，越前越简单）
- n = 800；参照 cIoU：w1f 0.5724，random_pick 0.5274，oracle_pick 0.6470

| 臂（简单→复杂） | cIoU | 减 W1-F | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 |
|---|---:|---:|---:|---:|---:|---:|
| sep_off_k4 | 0.5721 | -0.0002 | 0.206 | 0.958 | 0.071 | 0.852 |
| sep_off_k4_pts | 0.0816 | -0.4908 | 0.287 | 0.957 | 0.117 | 0.766 |

**差最大**：sep_off_k4；**选中**：sep_off_k4
```

## ranking.json 原文

```json
{
 "rule": "W2 预注册 §5 补记：按 best_index（真值，并列取编号小的）计数，2..K 号从多到少，并列编号小的在前；1 号永远保留",
 "oracle": "/home/mail/chenyuma/visint/repo/experiments/W2/0066_w2_tune_hpcgpu06/0066_s0/qwen3.5-9b/refseg/oracle_pick/devtune800/predictions.jsonl",
 "n_with_candidates": 786,
 "counts": {
  "1:gc_default": 133,
  "2:gc_box_plus10": 46,
  "3:gc_soft_box_minus10": 120,
  "4:gc_default+lcc+fill": 172,
  "5:slic_s30_t50": 42,
  "6:slic_s100_t50": 24,
  "7:ws_pts": 202,
  "8:box": 47
 },
 "ranking": [
  {
   "index": 7,
   "name": "ws_pts",
   "best_count": 202
  },
  {
   "index": 4,
   "name": "gc_default+lcc+fill",
   "best_count": 172
  },
  {
   "index": 3,
   "name": "gc_soft_box_minus10",
   "best_count": 120
  },
  {
   "index": 8,
   "name": "box",
   "best_count": 47
  },
  {
   "index": 2,
   "name": "gc_box_plus10",
   "best_count": 46
  },
  {
   "index": 5,
   "name": "slic_s30_t50",
   "best_count": 42
  },
  {
   "index": 6,
   "name": "slic_s100_t50",
   "best_count": 24
  }
 ]
}

```

## final.json 原文

```json
{
 "chosen": "sep_off_k4",
 "config": "/home/mail/chenyuma/visint/repo/experiments/W2/0066_w2_tune_hpcgpu06/cfg/sep_off_k4.json",
 "sha256": "e695bbf01b61886ebed492e2d9fa563ca6605743fe415227d68fbdc671167a2f",
 "step1": "sep_off",
 "step2": "sep_off_k4"
}

```

## 标题读数与选择统计

由 `experiments/W2/0066_w2_tune_probe/p66_tables.py` 从各 `metrics.json` 读出。第 0 步四个基线为 `0066_s0` 下的 w1f、random_pick、oracle_pick、ownbox。

### 分割臂（refseg，devtune800）

| 运行目录 | 叶 | cIoU [95% CI] | gIoU | 挑中最优 | 挑 1 号 | none | 解析失败 | prompt token 中位/最大 | 生成 token 中位/最大 | 出框失败 | 模型调用均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0066_grid_off | w2a | 0.5193 [0.5000, 0.5394] | 0.5333 | 0.131（103/786） | 0.095（75/786） | 0.055（43/786） | 0.004（3/786） | 716 / 789 | 47 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |
| 0066_grid_on | w2a | 0.5476 [0.5277, 0.5690] | 0.5616 | 0.169（133/786） | 0.342（269/786） | 0.061（48/786） | 0.048（38/786） | 714 / 787 | 2095 / 2165 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |
| 0066_s0 | oracle_pick | 0.6470 [0.6240, 0.6700] | 0.6586 | 1.000（786/786） | 0.169（133/786） | 0.000（0/786） | 0.000（0/786） | 334 / 370 | 45 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.0 |
| 0066_s0 | ownbox | 0.5029 [0.4854, 0.5207] | 0.5143 | — | — | — | — | — | — | {} | — |
| 0066_s0 | random_pick | 0.5274 [0.5078, 0.5480] | 0.5299 | 0.177（139/786） | 0.128（101/786） | 0.000（0/786） | 0.000（0/786） | 334 / 370 | 45 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.0 |
| 0066_s0 | w1f | 0.5724 [0.5469, 0.5976] | 0.5499 | 0.169（133/786） | 1.000（786/786） | 0.000（0/786） | 0.000（0/786） | 334 / 370 | 45 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.0 |
| 0066_sep_off | w2a | 0.5641 [0.5404, 0.5886] | 0.5465 | 0.165（130/786） | 0.926（728/786） | 0.011（9/786） | 0.911（716/786） | 3277 / 3487 | 109 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |
| 0066_sep_off_k4 | w2a | 0.5721 [0.5470, 0.5984] | 0.5525 | 0.206（162/786） | 0.958（753/786） | 0.071（56/786） | 0.852（670/786） | 2005 / 2135 | 108 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |
| 0066_sep_off_k4_pts | w2a | 0.0816 [0.0587, 0.1056] | 0.0692 | 0.287（27/94） | 0.957（90/94） | 0.117（11/94） | 0.766（72/94） | 406 / 2199 | 214 / 257 | {'ambiguous': 33, 'malformed': 4, 'multiple_boxes': 560, 'no_box': 2, 'truncated': 107} | 1.1175 |
| 0066_sep_off_k6 | w2a | 0.5646 [0.5405, 0.5891] | 0.5460 | 0.165（130/786） | 0.933（733/786） | 0.023（18/786） | 0.901（708/786） | 2642 / 2811 | 109 / 256 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |
| 0066_sep_on | w2a | 0.5610 [0.5418, 0.5829] | 0.5750 | 0.228（179/786） | 0.345（271/786） | 0.008（6/786） | 0.018（14/786） | 3275 / 3485 | 2095 / 2162 | {'multiple_boxes': 13, 'truncated': 1} | 1.9825 |

### 选择阶段（metrics.json 的 select，只有 w2a 臂有）

- 0066_grid_off：answer {'1': 29, '2': 44, '3': 64, '4': 24, '5': 54, '6': 225, '7': 135, '8': 165, 'none': 43, 'parse_failure': 3}；parse_failure {'out_of_range': 1, 'truncated': 2}；stopped {'False': 2, 'True': 784}；thinking_end {'None': 786}；final_index {'1': 75, '2': 44, '3': 64, '4': 24, '5': 54, '6': 225, '7': 135, '8': 165}；best_index {'1': 133, '2': 46, '3': 120, '4': 172, '5': 42, '6': 24, '7': 202, '8': 47}；oracle cIoU（有候选） 0.6606；config sha256 0d021a83cd2a…
- 0066_grid_on：answer {'1': 183, '2': 47, '3': 80, '4': 19, '5': 69, '6': 98, '7': 56, '8': 148, 'none': 48, 'parse_failure': 38}；parse_failure {'no_choice': 5, 'truncated': 33}；stopped {'False': 33, 'True': 753}；thinking_end {'forced': 685, 'natural': 101}；final_index {'1': 269, '2': 47, '3': 80, '4': 19, '5': 69, '6': 98, '7': 56, '8': 148}；best_index {'1': 133, '2': 46, '3': 120, '4': 172, '5': 42, '6': 24, '7': 202, '8': 47}；oracle cIoU（有候选） 0.6606；config sha256 eacaa53688ac…
- 0066_sep_off：answer {'1': 3, '4': 1, '7': 9, '8': 48, 'none': 9, 'parse_failure': 716}；parse_failure {'no_choice': 1, 'truncated': 715}；stopped {'False': 715, 'True': 71}；thinking_end {'None': 786}；final_index {'1': 728, '4': 1, '7': 9, '8': 48}；best_index {'1': 133, '2': 46, '3': 120, '4': 172, '5': 42, '6': 24, '7': 202, '8': 47}；oracle cIoU（有候选） 0.6606；config sha256 eb711dca30c2…
- 0066_sep_off_k4：answer {'1': 27, '2': 1, '3': 7, '4': 25, 'none': 56, 'parse_failure': 670}；parse_failure {'truncated': 670}；stopped {'False': 670, 'True': 116}；thinking_end {'None': 786}；final_index {'1': 753, '2': 1, '3': 7, '4': 25}；best_index {'1': 152, '2': 167, '3': 185, '4': 282}；oracle cIoU（有候选） 0.6520；config sha256 e695bbf01b61…
- 0066_sep_off_k4_pts：answer {'1': 7, '4': 4, 'none': 11, 'parse_failure': 72}；parse_failure {'truncated': 72}；stopped {'False': 72, 'True': 22}；thinking_end {'None': 94}；final_index {'1': 90, '4': 4}；best_index {'1': 26, '2': 27, '3': 20, '4': 21}；oracle cIoU（有候选） 0.5779；config sha256 792f9b2dbfa9…
- 0066_sep_off_k6：answer {'1': 7, '3': 1, '4': 2, '5': 4, '6': 46, 'none': 18, 'parse_failure': 708}；parse_failure {'truncated': 708}；stopped {'False': 708, 'True': 78}；thinking_end {'None': 786}；final_index {'1': 733, '3': 1, '4': 2, '5': 4, '6': 46}；best_index {'1': 134, '2': 52, '3': 124, '4': 172, '5': 209, '6': 95}；oracle cIoU（有候选） 0.6585；config sha256 9c6a79cdf359…
- 0066_sep_on：answer {'1': 251, '2': 37, '3': 29, '4': 25, '5': 95, '6': 40, '7': 103, '8': 186, 'none': 6, 'parse_failure': 14}；parse_failure {'no_choice': 2, 'truncated': 12}；stopped {'False': 12, 'True': 774}；thinking_end {'forced': 689, 'natural': 97}；final_index {'1': 271, '2': 37, '3': 29, '4': 25, '5': 95, '6': 40, '7': 103, '8': 186}；best_index {'1': 133, '2': 46, '3': 120, '4': 172, '5': 42, '6': 24, '7': 202, '8': 47}；oracle cIoU（有候选） 0.6606；config sha256 d0e4a6543388…

### 候选（candidates*/metrics.json）

- 0066_grid_off/candidates：k 8，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'slic_s30_t50', 'slic_s100_t50', 'ws_pts', 'box']，duplicate_rate 0.0431（271），not_ok_by_index {'2': 1}
- 0066_grid_on/candidates：k 8，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'slic_s30_t50', 'slic_s100_t50', 'ws_pts', 'box']，duplicate_rate 0.0431（271），not_ok_by_index {'2': 1}
- 0066_s0/candidates：k 8，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'slic_s30_t50', 'slic_s100_t50', 'ws_pts', 'box']，duplicate_rate 0.0431（271），not_ok_by_index {'2': 1}
- 0066_sep_off/candidates：k 8，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'slic_s30_t50', 'slic_s100_t50', 'ws_pts', 'box']，duplicate_rate 0.0431（271），not_ok_by_index {'2': 1}
- 0066_sep_off_k4/candidates：k 4，n_with_candidates 786，names ['gc_default', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts']，duplicate_rate 0.0741（233），not_ok_by_index {}
- 0066_sep_off_k4_pts/candidates_box_points：k 4，n_with_candidates 94，names ['gc_default', 'gc_pts', 'gc_default+lcc+fill', 'ws_pts']，duplicate_rate 0.0984（37），not_ok_by_index {'2': 13}
- 0066_sep_off_k6/candidates：k 6，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'ws_pts', 'box']，duplicate_rate 0.0575（271），not_ok_by_index {'2': 1}
- 0066_sep_on/candidates：k 8，n_with_candidates 786，names ['gc_default', 'gc_box_plus10', 'gc_soft_box_minus10', 'gc_default+lcc+fill', 'slic_s30_t50', 'slic_s100_t50', 'ws_pts', 'box']，duplicate_rate 0.0431（271），not_ok_by_index {'2': 1}

### 出框（refexp_box/baseline_e1）与 box_points 出点

- 0066_grid_off：acc@0.5 0.8950 [0.8738, 0.9150]，n 800
- 0066_grid_on：acc@0.5 0.8950 [0.8738, 0.9150]，n 800
- 0066_s0：acc@0.5 0.8950 [0.8738, 0.9150]，n 800
- 0066_sep_off：acc@0.5 0.8950 [0.8738, 0.9150]，n 800
- 0066_sep_off_k4：acc@0.5 0.8950 [0.8738, 0.9150]，n 800
- 0066_sep_off_k4_pts/propose_box_points：failures {'ambiguous': 33, 'malformed': 4, 'multiple_boxes': 560, 'no_box': 2, 'ok': 94, 'truncated': 107}，fg_points {"n": 800, "min": 0, "p05": 0.0, "p50": 0.0, "p95": 3.0, "max": 3, "mean": 0.26875, "n_distinct": 4, "top": [[0, 719], [3, 61], [2, 12], [1, 8]]}，bg_points {"n": 800, "min": 0, "p05": 0.0, "p50": 0.0, "p95": 2.0, "max": 3, "mean": 0.22625, "n_distinct": 4, "top": [[0, 719], [2, 44], [3, 28], [1, 9]]}，n_points_dropped 1，complete True
- 0066_sep_off_k6：acc@0.5 0.8950 [0.8738, 0.9150]，n 800
- 0066_sep_on：acc@0.5 0.8950 [0.8738, 0.9150]，n 800

### 显存（gpu_mem.csv：15919 行，04:11:19Z – 08:36:44Z，全程最大 32069 MiB）

- s0_e1_box：wall_s 129，窗口内最大 27769 MiB（129 行）
- s0_w1f：wall_s 458，窗口内最大 4 MiB（458 行）
- s0_ownbox：wall_s 4，窗口内最大 4 MiB（4 行）
- s0_random_pick：wall_s 10，窗口内最大 4 MiB（10 行）
- s0_oracle_pick：wall_s 9，窗口内最大 4 MiB（9 行）
- s0_rank：wall_s 3，窗口内最大 4 MiB（3 行）
- s1_grid_off：wall_s 539，窗口内最大 21007 MiB（538 行）
- s1_sep_off：wall_s 1097，窗口内最大 32069 MiB（1096 行）
- s1_grid_on：wall_s 4436，窗口内最大 32069 MiB（4435 行）
- s1_sep_on：wall_s 7180，窗口内最大 29745 MiB（7178 行）
- s2_sep_off_k4：wall_s 726，窗口内最大 26567 MiB（725 行）
- s2_sep_off_k6：wall_s 943，窗口内最大 29267 MiB（942 行）
- s3_sep_off_k4_pts：wall_s 362，窗口内最大 28145 MiB（362 行）

（窗口按 steps.log 每步结束时间往前 wall_s；gpu_mem.csv 时间戳为节点本地 +08:00，已换成 UTC。相邻步的窗口在交界处各含一行。）

## traceback

- `logs/` 下 13 个步日志 grep `Traceback`、`out of memory`：都是 0。

## 收尾

- 同步循环 08:44:30Z、09:44:55Z 两轮在 done 之后推送成功（origin `87a40b9a` 起）。10:04:14Z 在 git.lock 下按 PID 停掉循环 605932（当时在 sleep）。
- 之后：06 无 tune_w2 / run_w2 进程、GPU 空、`train.pause.hpcgpu06` 不存在；squeue 157296、157297 都 RUNNING。06 空着等下一份任务书。
