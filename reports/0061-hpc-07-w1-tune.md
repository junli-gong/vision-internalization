# RUN-0061 报告：hpcgpu07，W1 开发-调参 T1–T4 —— 12 步全部 rc=2，没有跑起来

任务书 `prompts/server_hpc/0061-hpc-07-w1-tune.md`（origin 141d8f4d）。只用 devtune；没有碰 devcheck；没有碰 06 的 RUN-0056。

## 启动

- 10-21 23:17:17Z 第一次启动脚本（`$VI/tmp/start0061.sh`）：squeue 两条 RUNNING；07 没有暂停文件；`git fetch` 失败（rc=128，`Connection to 20.205.243.166 port 22 timed out`），工作树建在旧的 origin/main 1aecbebd，缺 tune.sh 与 T1–T4 配置；12 个 sha256 对不上，**脚本在启动前中止**，什么都没启动。
- 23:20:37Z 重试 fetch 成功（origin/main 含 141d8f4d）；删掉旧的（干净的）wt0061，23:20:43Z 重跑启动脚本：
  - wt0061：HEAD `141d8f4d1fa1936b9e5f6d2082dd85577c852c0b`，status 空，`experiments/` 只有 P0，含 6038d755。
  - 12 个 sha256 与任务书逐字相同；tune.sh 用 `EV_PAUSE=$LOGS/train.pause.$NODE`。
  - 23:20:56Z 按任务书命令启动 tune.sh（`EV_CAP_H=6`）；推送循环 0061 由脚本起。
- 心跳：`2026-10-21T23:20:57Z waiting for gpu (running: train_0057-hpcgpu07.status; pids: 1137806)`；暂停文件 `train.pause.hpcgpu07` 内容 `0061-hpc-07-w1-tune.md`。
- （另：为备用的 443 路线做过只读核对：`ssh-keyscan -p 443 ssh.github.com` 得 `256 SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU [ssh.github.com]:443 (ED25519)`，与 GitHub 公布的一致；没用上，没写进 known_hosts。）

## 让路与交还

- RUN-0057 在第 2 段之后停住：驱动 `=== 2026-10-22T00:11:28Z chunk rc=0 wall_s=6661`，心跳 `2026-10-22T00:11:28Z paused (pause file: 0061-hpc-07-w1-tune.md )`。
- RUN-0061 00:11:30Z 拿到 GPU（`steps.log` 的 start 行）。
- 00:12:20Z `done 0061 (failed steps: 12)`；暂停文件已删（00:13Z 查，`train.pause*` 都不存在）；RUN-0057 心跳 `2026-10-22T00:12:28Z running e3b_c3_n4000_s0 (chunk 3)`。
- 07 的 GPU 被借用约 1 分钟。`gpu_mem.csv` 49 行，最大 0 MiB（没有加载模型）。

## steps.log 原文

```
2026-10-22T00:11:30Z start commit 141d8f4d1fa1936b9e5f6d2082dd85577c852c0b node hpcgpu07 model qwen3.5-9b n 96
2026-10-22T00:11:49Z base_e1_refexp_box_devtune96 rc=2 wall_s=18
2026-10-22T00:11:52Z base_ownbox_devtune96 rc=2 wall_s=3
2026-10-22T00:11:56Z base_poly_structured_devtune96 rc=2 wall_s=4
2026-10-22T00:11:59Z base_w1f_devtune96 rc=2 wall_s=3
2026-10-22T00:12:02Z arm_T1_refseg_devtune96 rc=2 wall_s=3
2026-10-22T00:12:04Z arm_T2_refseg_devtune96 rc=2 wall_s=2
2026-10-22T00:12:07Z arm_T3_refseg_devtune96 rc=2 wall_s=3
2026-10-22T00:12:09Z arm_T4_refseg_devtune96 rc=2 wall_s=2
2026-10-22T00:12:12Z gate_tune_T1 rc=2 wall_s=3
2026-10-22T00:12:15Z gate_tune_T2 rc=2 wall_s=3
2026-10-22T00:12:17Z gate_tune_T3 rc=2 wall_s=2
2026-10-22T00:12:20Z gate_tune_T4 rc=2 wall_s=3
2026-10-22T00:12:20Z done; failed steps: 12
```

## 报错原文

- 8 个 run_w1.py 步（`logs/base_*.log`、`logs/arm_T*_refseg_devtune96.log`）都是参数检查报错，例如：

```
run_w1.py: error: --out 的目录名必须带运行编号，形如 0060_w1_devcheck，收到 '/home/mail/chenyuma/visint/repo/experiments/W1/0061_w1_tune_hpcgpu07/base'
run_w1.py: error: --out 的目录名必须带运行编号，形如 0060_w1_devcheck，收到 '/home/mail/chenyuma/visint/repo/experiments/W1/0061_w1_tune_hpcgpu07/T1'
```

- 4 个 gate 步：`不能判定：w1：/home/mail/chenyuma/visint/repo/experiments/W1/0061_w1_tune_hpcgpu07/T1/qwen3.5-9b/refseg/full/devtune 里没有 predictions.jsonl`（T2–T4 同）。
- 相关代码（只读，未改）：
  - `prompts/server_hpc/0061-hpc-07-w1-tune/tune.sh` 第 36 行 `OUT=$REPO/experiments/W1/0061_w1_tune_$NODE`，第 44 行 `--out "$OUT/$sub"`，`$sub` 是 `base`、`T1` … `T4`。
  - `scripts/run_w1.py` 第 98–99 行：`if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name): p.error(...)`。这条检查在 `--tiny` 时跳过，所以开发机的 tiny 彩排没有触发。

## 推送

- 推送循环 0061 在跑；23:53:33Z 一轮 `push failed (will retry)`（GitHub 22 端口超时，同前）。先不停，等管理员决定是否重发。

## 第 2 次（补充 1，origin 00fa15b1；tune.sh 子目录改为 0061_base、0061_T1 … 0061_T4）

### 启动与让路

- 10-22 00:17:28Z（`$VI/tmp/start0061b.sh`，输出 `$VI/logs/start0061b.txt`）：squeue 两条 RUNNING；`eval_0061.status` 为第 1 次的 `done 0061 (failed steps: 12)`；07 没有暂停文件。删掉旧的（干净的）wt0061（141d8f4d）；fetch rc=0；新 wt0061 HEAD `00fa15b1e900d3c54379a5da4f3089045402920e`，status 空，`experiments/` 只有 P0，含 00fa15b1。
- 12 个 sha256 逐字相同（tune.sh 用新值 `cfc4d740e000b21a9a74d49d4ed2f7fd6a9ebab6c67b9702bb527944502939c2`，其余 11 个同原任务书）；tune.sh 用本节点暂停文件。
- 00:18:09Z 启动 tune.sh（`EV_CAP_H=6`），step 157297.94。心跳 `00:18:40Z waiting for gpu (running: train_0057-hpcgpu07.status; pids: 1271210)`；暂停文件内容 `0061-hpc-07-w1-tune.md`。推送循环 0061 沿用（PID 1294229），没有另起。
- RUN-0057 在第 3 段之后停住：驱动 `=== 2026-10-22T02:01:29Z chunk rc=0 wall_s=6541`；心跳 `paused (pause file: 0061-hpc-07-w1-tune.md )`。
- RUN-0061 02:01:47Z 拿到 GPU；05:43:29Z `done 0061 (failed steps: 0)`；从拿到 GPU 起 13,302 s（3.70 h），上限 6 h。
- 交还：05:49Z 查，`train.pause*` 都不存在；RUN-0057 心跳 `2026-10-22T05:48:31Z running e3b_c3_n4000_s0 (5 min into chunk; mem avail 477GB)`（第 4 段）；squeue 两条 RUNNING。
- `gpu_mem.csv` 由 tune.sh 用 `>` 重新写，第 1 次的那份（49 行，全 0）被覆盖。第 1 次的 `steps.log` 行与 `logs/` 保留，第 2 次接着写。

### steps.log 原文（第 2 次）

```
2026-10-22T02:01:47Z start commit 00fa15b1e900d3c54379a5da4f3089045402920e node hpcgpu07 model qwen3.5-9b n 96
2026-10-22T02:03:13Z base_e1_refexp_box_devtune96 rc=0 wall_s=86
2026-10-22T02:03:16Z base_ownbox_devtune96 rc=0 wall_s=3
2026-10-22T02:04:57Z base_poly_structured_devtune96 rc=0 wall_s=101
2026-10-22T02:06:44Z base_w1f_devtune96 rc=0 wall_s=107
2026-10-22T02:55:34Z arm_T1_refseg_devtune96 rc=0 wall_s=2930
2026-10-22T03:44:47Z arm_T2_refseg_devtune96 rc=0 wall_s=2953
2026-10-22T04:38:09Z arm_T3_refseg_devtune96 rc=0 wall_s=3202
2026-10-22T05:43:09Z arm_T4_refseg_devtune96 rc=0 wall_s=3900
2026-10-22T05:43:19Z gate_tune_T1 rc=0 wall_s=10
2026-10-22T05:43:22Z gate_tune_T2 rc=0 wall_s=3
2026-10-22T05:43:26Z gate_tune_T3 rc=0 wall_s=4
2026-10-22T05:43:29Z gate_tune_T4 rc=0 wall_s=3
2026-10-22T05:43:29Z done; failed steps: 0
```

以下由 `experiments/W1/0061_w1_tune_probe/p61_tables.py` 从结果文件读出。每轮调用个数按各轮 `tool_calls` 的条数计；被拒调用按 `tool_calls[].refused` 的原因计——四个臂里只出现 `final_round`，`tool_cap` 与 `after_submission` 都是 0。显存是 nvidia-smi 的 memory.used。没有 traceback。

## 各臂（refseg / full / devtune，n = 96）

### T1

| 项 | 值 |
|---|---|
| 句数 | 96 |
| 提交（ended_by = submission） | 48 |
| ended_by 分布 | round_cap: 48, submission: 48 |
| failure 分布 | no_answer: 48, wrong_submission: 1, None: 47 |
| 轮数分布 | 5: 1, 6: 95 |
| n_tool_calls 分布 | 4: 2, 5: 92, 6: 1, 7: 1 |
| 计数调用（counted = true）个数分布 | 4: 2, 5: 92, 6: 1, 7: 1 |
| 每轮调用个数的分布（轮数） | 0: 2, 1: 570, 2: 3 |
| 被拒调用按原因 | {'final_round': 41}（n_refused 合计 41） |
| 工具调用出错（n_tool_call_errors 合计） | 2 |
| 每句生成 token 中位 / 最大 | 1366.5 / 4096 |
| 标题读数 | cIoU 0.3289 [0.2384, 0.4099] n=96; gIoU 0.2541 [0.1918, 0.3178] n=96 |
| tools_sha256 | dcc91402… |

### T2

| 项 | 值 |
|---|---|
| 句数 | 96 |
| 提交（ended_by = submission） | 36 |
| ended_by 分布 | round_cap: 60, submission: 36 |
| failure 分布 | no_answer: 60, None: 36 |
| 轮数分布 | 4: 2, 6: 94 |
| n_tool_calls 分布 | 3: 2, 4: 2, 5: 91, 7: 1 |
| 计数调用（counted = true）个数分布 | 3: 2, 4: 2, 5: 91, 7: 1 |
| 每轮调用个数的分布（轮数） | 0: 3, 1: 567, 2: 2 |
| 被拒调用按原因 | {'final_round': 57}（n_refused 合计 57） |
| 工具调用出错（n_tool_call_errors 合计） | 3 |
| 每句生成 token 中位 / 最大 | 1347.5 / 4093 |
| 标题读数 | cIoU 0.2406 [0.1631, 0.3265] n=96; gIoU 0.2010 [0.1458, 0.2649] n=96 |
| tools_sha256 | dcc91402… |

### T3

| 项 | 值 |
|---|---|
| 句数 | 96 |
| 提交（ended_by = submission） | 42 |
| ended_by 分布 | round_cap: 54, submission: 42 |
| failure 分布 | no_answer: 54, wrong_submission: 1, None: 41 |
| 轮数分布 | 5: 1, 6: 95 |
| n_tool_calls 分布 | 4: 1, 5: 94, 6: 1 |
| 计数调用（counted = true）个数分布 | 4: 1, 5: 94, 6: 1 |
| 每轮调用个数的分布（轮数） | 1: 574, 2: 1 |
| 被拒调用按原因 | {'final_round': 51}（n_refused 合计 51） |
| 工具调用出错（n_tool_call_errors 合计） | 0 |
| 每句生成 token 中位 / 最大 | 1543.0 / 3621 |
| 标题读数 | cIoU 0.3130 [0.2237, 0.3894] n=96; gIoU 0.1926 [0.1406, 0.2475] n=96 |
| tools_sha256 | dcc91402… |

### T4

| 项 | 值 |
|---|---|
| 句数 | 96 |
| 提交（ended_by = submission） | 39 |
| ended_by 分布 | round_cap: 57, submission: 39 |
| failure 分布 | no_answer: 57, None: 39 |
| 轮数分布 | 4: 1, 5: 1, 6: 94 |
| n_tool_calls 分布 | 3: 2, 4: 1, 5: 91, 7: 1, 8: 1 |
| 计数调用（counted = true）个数分布 | 3: 2, 4: 1, 5: 91, 7: 1, 8: 1 |
| 每轮调用个数的分布（轮数） | 0: 2, 1: 566, 2: 5 |
| 被拒调用按原因 | {'final_round': 57}（n_refused 合计 57） |
| 工具调用出错（n_tool_call_errors 合计） | 3 |
| 每句生成 token 中位 / 最大 | 1667.0 / 10612 |
| 标题读数 | cIoU 0.3048 [0.2094, 0.4062] n=96; gIoU 0.2121 [0.1514, 0.2787] n=96 |
| tools_sha256 | 0bdcaf73… |

## 基线（devtune，n = 96）

| 步 | 标题读数 |
|---|---|
| refexp_box / baseline_e1 | acc@0.5 0.8750 [0.8021, 0.9375] n=96 |
| refseg / baseline_ownbox | cIoU 0.4670 [0.4081, 0.5252] n=96; gIoU 0.5013 [0.4601, 0.5433] n=96 |
| refseg / baseline_poly_structured | cIoU 0.3487 [0.2859, 0.4177] n=96; gIoU 0.3612 [0.3129, 0.4091] n=96 |
| refseg / w1f | cIoU 0.5222 [0.4572, 0.5948] n=96; gIoU 0.5304 [0.4729, 0.5895] n=96 |

## 显存（gpu_mem.csv：13301 行，全程最大 28425 MiB）

| 步 | rc | 墙钟 s | 窗口内显存峰值 MiB |
|---|---|---|---|
| base_e1_refexp_box_devtune96 | 0 | 86 | 26987 |
| base_ownbox_devtune96 | 0 | 3 | 0 |
| base_poly_structured_devtune96 | 0 | 101 | 27689 |
| base_w1f_devtune96 | 0 | 107 | 4 |
| arm_T1_refseg_devtune96 | 0 | 2930 | 28305 |
| arm_T2_refseg_devtune96 | 0 | 2953 | 28125 |
| arm_T3_refseg_devtune96 | 0 | 3202 | 28425 |
| arm_T4_refseg_devtune96 | 0 | 3900 | 28425 |
| gate_tune_T1 | 0 | 10 | 0 |
| gate_tune_T2 | 0 | 3 | 0 |
| gate_tune_T3 | 0 | 4 | 0 |
| gate_tune_T4 | 0 | 3 | 0 |

## gate_tune_T1.md 原文

# W1 去留判定（D103 §三，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devtune，n = 96；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck': False, 'n_is_500': False, 'not_tiny': True, 'no_limit': False}

| 读数 | 值 |
|---|---|
| W1-1（full）cIoU | 0.3289 |
| 自己的框 cIoU | 0.4670 |
| E1 结构强制多边形 cIoU | 0.3487 |

1. W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0：差 -0.1381，95% CI [-0.2133, -0.0703] → **不满足**
2. W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）：差 -0.0198（-51750297677/2614320583580）→ **不满足**

**结论：彩排，不作数（规则给出：放弃）**


## gate_tune_T2.md 原文

# W1 去留判定（D103 §三，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devtune，n = 96；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck': False, 'n_is_500': False, 'not_tiny': True, 'no_limit': False}

| 读数 | 值 |
|---|---|
| W1-1（full）cIoU | 0.2406 |
| 自己的框 cIoU | 0.4670 |
| E1 结构强制多边形 cIoU | 0.3487 |

1. W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0：差 -0.2264，95% CI [-0.3132, -0.1444] → **不满足**
2. W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）：差 -0.1081（-1345230464389/12445029760610）→ **不满足**

**结论：彩排，不作数（规则给出：放弃）**


## gate_tune_T3.md 原文

# W1 去留判定（D103 §三，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devtune，n = 96；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck': False, 'n_is_500': False, 'not_tiny': True, 'no_limit': False}

| 读数 | 值 |
|---|---|
| W1-1（full）cIoU | 0.3130 |
| 自己的框 cIoU | 0.4670 |
| E1 结构强制多边形 cIoU | 0.3487 |

1. W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0：差 -0.1540，95% CI [-0.2413, -0.0883] → **不满足**
2. W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）：差 -0.0357（-460725413559/12889930633235）→ **不满足**

**结论：彩排，不作数（规则给出：放弃）**


## gate_tune_T4.md 原文

# W1 去留判定（D103 §三，规则写死）

- 模型 qwen3.5-9b@c20223623576…，划分 devtune，n = 96；判定点核对：{'model_is_qwen3.5-9b': True, 'split_is_devcheck': False, 'n_is_500': False, 'not_tiny': True, 'no_limit': False}

| 读数 | 值 |
|---|---|
| W1-1（full）cIoU | 0.3048 |
| 自己的框 cIoU | 0.4670 |
| E1 结构强制多边形 cIoU | 0.3487 |

1. W1-1 − 自己的框 的 cIoU 差，配对 bootstrap 95% CI 下界 > 0：差 -0.1621，95% CI [-0.2479, -0.0812] → **不满足**
2. W1-1 − E1 结构强制多边形 的 cIoU 差 ≥ 0.2（精确值）：差 -0.0439（-555561753302/12660656612205）→ **不满足**

**结论：彩排，不作数（规则给出：放弃）**


## 推送循环 0061

- 最后一轮 origin `8d28033a [results] 0061: sync 2026-10-22T09:07:47Z`（含 `gate_tune_T4.md` 与 `steps.log` 的 `done; failed steps: 0`），`sync_0061.status` `2026-10-22T09:07:56Z ok`。09:08:43Z 在 sleep 时、`flock -n git.lock` 下按 PID 停掉（1294229 与 sleep 4013698）。
