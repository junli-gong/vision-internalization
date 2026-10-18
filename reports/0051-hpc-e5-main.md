# RUN-0051 — E5 主比较 v3 加 E3b 起点（两台 A100 按组认领）

服务器 CC（HPC）。只贴数字与事实，不做解读，**不贴 test 的分数**。原始输出：`experiments/E5/0051_e5_main_probe/`。

## 1. 校验（2026-10-12 20:31Z 在 hpcmgt 上；`launch07_and_prep.txt`）

- origin/main 包含任务书 commit `43826228`。
- 工作树 `$VI/wt/wt0051` 先建在 `43826228`。之后按 D95，于 10-13 22:4xZ 切到 `07d677df`，同时改为稀疏签出（RUN-0053）。两次之后 status 都为空。
- **sha256**（D95 之后的值，启动前在工作树里再核对一次）：

  | 文件 | sha256 |
  |---|---|
  | `prompts/server_hpc/0051-hpc-e5-main/queue.sh` | `8f33ad28…a5ab769` |
  | `scripts/server/train_queue.sh` | `fcb10f47…7e0dbaa` |
  | `configs/e5_train.json` | `e7f28997…2b94c3` |
  | `configs/e5_prompts.json` | `c0e59d35…f80467ec4` |

  四项都与任务书相同。
- **起点检查点：** 12 个（`0040/{e3b_c3,c3_warmup}`、`0036/{warmup,e3b}` × 种子 0–2），对应 `SHA256SUMS.b200` 里的 96 行，每个 8 个文件。`sha256sum -c` 的结果：OK 96、FAILED 0、缺失 0。
- 启动前：`e5_claims_0051/` 不存在；`train.pause.hpcgpu07`、`HOLD`、`train.pause` 都不存在；07 的卡上没有进程；`run_0050.log` 的最后一行是 `2026-10-14T08:51:20Z done`。

## 2. 启动（hpcgpu07）

- **2026-10-14T09:28:12Z**，方案 B（`launch.sh`）：`env NODE=hpcgpu07 PATH=$VI/env/bin:/usr/bin:/bin QUEUE_CAP_H=170 bash $VI/wt/wt0051/…/queue.sh`，步骤 **157297.84**。
- **squeue：** 启动前 `09:27:58Z 157296 RUNNING 59-20:53:44 / 157297 RUNNING 61-18:25:53`；启动后 `09:29:12Z 157296 RUNNING 59-20:54:58 / 157297 RUNNING 61-18:27:07`。
- **驱动日志：** `队列开始：hpcgpu07，commit 07d677df0f0cd79c5f04e58b67e557e2ea18d6c1，起点根 …/b200/ckpt/0040、…/b200/ckpt/0036，每段 1000 步`，之后是 `认领组 n0_all`。
- **推送：** v3（`$VI/wt/sync3`），TAG 0051，路径 `experiments/E5/0051_e5_main_hpcgpu07`，间隔 1800 s，第一轮 `09:28:23Z ok`。
  - 循环的真实 PID 是 310182。启动时记成了 310181，那是外层子 shell 的 PID；10:46Z 已更正，见 RUN-0053 报告 §7。

## 3. 组：n0_all（hpcgpu07）

- 认领 09:28:12Z，跑完 15:05:34Z，**墙钟 5.62 h**，预估约 5.5 h。
- 每个读数的墙钟按 driver.log 里相邻两次"完成"的时刻计（`n0_all_driver_log.txt`）：

  | 种子 | e3b_c3 | c3_warmup | base_warmup | e3b |
  |---|---|---|---|---|
  | 0 | 32.8 min | 30.5 | 30.5 | 30.5 |
  | 1 | 30.0 | 30.5 | 31.0 | —（种子 1 的 E3b 起点不用，E5 预注册 §8b） |
  | 2 | 30.0 | 30.5 | 30.5 | 30.5 |

- 复核 1（任务书 §2）：比预估慢约 2%，没到 15%。已用 SendMessage 告诉管理员。
- 下面的表由 `experiments/E5/0051_e5_main_probe/p51_n0.py` 从 origin 上的结果读出，快照 `20683876`（`sync 2026-10-14T16:02:49Z`）。**不含 test 分数。**
- 要点（只陈述）：
  - 11 个起点合并核对都是 pass，都是当场跑的，没有复用；
  - restricted thinking_off 的 exact_rows 里，**E3b 种子 0 与 E3b+C3 种子 2 都是 0.95，正好等于下限**（10 / 200 行不同）；
  - 55 个任务读数都是 complete=True，n = n_expected，解析失败 0、拒答 0、截断 0；
  - 每个 metrics.json 里的 `subset` 字段都写着 `"eval"`，而 timing.jsonl 里的样本 id 是 `<任务>/test/<i>`。

- 本地快照：最后一次改动这个目录的 commit `20683876 2026-10-15T00:02:49+08:00 [results] 0051: sync 2026-10-14T16:02:49Z`。
- 结果目录：`experiments/E5/0051_e5_main_hpcgpu07`；读数 11 个（种子 0–2 × e3b_c3、c3_warmup、base_warmup；e3b 只种子 0、2）。
- `driver/done.txt` 行数 11；`driver/gpu_seconds` = 20226。
- 不含任何分数（IoU、起伏误差、门判定均不读出）。

#### 1. 起点合并核对

| 读数 | 核对文件 | pass | failures | restricted thinking_off exact_rows（n_divergent_rows / n_rows） | 判据下限 | timing_s.total_s | written_utc | run json 的 reused | 驱动日志 |
|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | `train/e3b_c3/qwen3.5-2b/start_checks/merge_check_bf73188b8c88c760.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 0.985（3 / 200） | ≥ 0.95 | 447.39 | 2026-10-14T09:36:30+00:00 | False | 当场跑了核对 |
| c3_warmup s0 | `train/c3_warmup/qwen3.5-2b/start_checks/merge_check_dca6a89d9a3e920c.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 1（0 / 200） | ≥ 0.95 | 432.71 | 2026-10-14T10:08:18+00:00 | False | 当场跑了核对 |
| base_warmup s0 | `train/base_warmup/qwen3.5-2b/start_checks/merge_check_58ba382e9cd0b5bf.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 1（0 / 200） | ≥ 0.95 | 425.06 | 2026-10-14T10:38:41+00:00 | False | 当场跑了核对 |
| e3b s0 | `train/e3b/qwen3.5-2b/start_checks/merge_check_acb7f58d1ae3aa0c.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 0.95（10 / 200） | ≥ 0.95 | 424.34 | 2026-10-14T11:09:10+00:00 | False | 当场跑了核对 |
| e3b_c3 s1 | `train/e3b_c3/qwen3.5-2b/start_checks/merge_check_7c72a5007e461cca.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 0.955（9 / 200） | ≥ 0.95 | 394.53 | 2026-10-14T11:39:11+00:00 | False | 当场跑了核对 |
| c3_warmup s1 | `train/c3_warmup/qwen3.5-2b/start_checks/merge_check_e547056affbd7f0b.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 1（0 / 200） | ≥ 0.95 | 432.15 | 2026-10-14T12:09:49+00:00 | False | 当场跑了核对 |
| base_warmup s1 | `train/base_warmup/qwen3.5-2b/start_checks/merge_check_e13b53a97ceec206.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 1（0 / 200） | ≥ 0.95 | 431.81 | 2026-10-14T12:40:18+00:00 | False | 当场跑了核对 |
| e3b_c3 s2 | `train/e3b_c3/qwen3.5-2b/start_checks/merge_check_aaacbbab3e2f2ef3.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 0.95（10 / 200） | ≥ 0.95 | 403.22 | 2026-10-14T13:10:51+00:00 | False | 当场跑了核对 |
| c3_warmup s2 | `train/c3_warmup/qwen3.5-2b/start_checks/merge_check_1ef34cdf543d80fc.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 1（0 / 200） | ≥ 0.95 | 433.27 | 2026-10-14T13:41:21+00:00 | False | 当场跑了核对 |
| base_warmup s2 | `train/base_warmup/qwen3.5-2b/start_checks/merge_check_f7367f7da15e1b80.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 1（0 / 200） | ≥ 0.95 | 426.3 | 2026-10-14T14:11:44+00:00 | False | 当场跑了核对 |
| e3b s2 | `train/e3b/qwen3.5-2b/start_checks/merge_check_a4073791af6478a6.json`（5 个任务的 run json 引用） | True | restricted: 0; merge: 0 | 0.965（7 / 200） | ≥ 0.95 | 424.99 | 2026-10-14T14:42:14+00:00 | False | 当场跑了核对 |

- `train/*/qwen3.5-2b/start_checks/` 下 merge_check_*.json 共 11 个；未被上表引用的：无。
- failures 一栏是各项核对的失败条数（`failures.<核对>` 列表长度）。

#### 2. test 评测：完整性、有效率与解析失败率

字段（metrics.json）：`subset`、`n`、`n_expected`、`complete`；`parse.{n_ok,n_failed,failure_rate}`；`refusal.n`；`truncation.{n,rate}`；`thinking_unclosed.{n,rate}`；`overall.<指标>.{n_valid,n,valid_rate}`、`headline.<指标>.{n_valid,n,valid_rate}`。pred 行 = predictions.jsonl 的行数。

##### rotation3d

| 读数 | subset | n | n_expected | complete | pred 行 | parse ok/failed | parse failure_rate | refusal n | truncation n (rate) | thinking_unclosed n (rate) | overall mask_iou_grid valid | overall mask_iou_full valid | headline mask_iou_grid valid | headline mask_iou_full valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| c3_warmup s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| base_warmup s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| e3b s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| e3b_c3 s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| c3_warmup s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| base_warmup s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| e3b_c3 s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| c3_warmup s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| base_warmup s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |
| e3b s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 300/300 (1) | 300/300 (1) |

##### viewpoint

| 读数 | subset | n | n_expected | complete | pred 行 | parse ok/failed | parse failure_rate | refusal n | truncation n (rate) | thinking_unclosed n (rate) | overall mask_iou_grid valid | overall mask_iou_full valid | headline mask_iou_grid valid | headline mask_iou_full valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| c3_warmup s0 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| base_warmup s0 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| e3b s0 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| e3b_c3 s1 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| c3_warmup s1 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| base_warmup s1 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| e3b_c3 s2 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| c3_warmup s2 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| base_warmup s2 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |
| e3b s2 | eval | 800 | 800 | True | 800 | 800/0 | 0 | 0 | 0 (0) | 0 (0) | 800/800 (1) | 800/800 (1) | 315/315 (1) | 315/315 (1) |

##### occlusion

| 读数 | subset | n | n_expected | complete | pred 行 | parse ok/failed | parse failure_rate | refusal n | truncation n (rate) | thinking_unclosed n (rate) | overall mask_iou_grid valid | overall mask_iou_full valid | headline mask_iou_grid valid | headline mask_iou_full valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| c3_warmup s0 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| base_warmup s0 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| e3b s0 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| e3b_c3 s1 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| c3_warmup s1 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| base_warmup s1 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| e3b_c3 s2 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| c3_warmup s2 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| base_warmup s2 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |
| e3b s2 | eval | 300 | 300 | True | 300 | 300/0 | 0 | 0 | 0 (0) | 0 (0) | 300/300 (1) | 300/300 (1) | 300/300 (1) | 300/300 (1) |

##### physics

| 读数 | subset | n | n_expected | complete | pred 行 | parse ok/failed | parse failure_rate | refusal n | truncation n (rate) | thinking_unclosed n (rate) | overall mask_iou_grid valid | overall mask_iou_full valid | headline mask_iou_grid valid | headline mask_iou_full valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| c3_warmup s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| base_warmup s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b_c3 s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| c3_warmup s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| base_warmup s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b_c3 s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| c3_warmup s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| base_warmup s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |

##### imagery

| 读数 | subset | n | n_expected | complete | pred 行 | parse ok/failed | parse failure_rate | refusal n | truncation n (rate) | thinking_unclosed n (rate) | overall mask_iou_grid valid | overall mask_iou_full valid | headline mask_iou_grid valid | headline mask_iou_full valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| c3_warmup s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| base_warmup s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b s0 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b_c3 s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| c3_warmup s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| base_warmup s1 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b_c3 s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| c3_warmup s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| base_warmup s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |
| e3b s2 | eval | 400 | 400 | True | 400 | 400/0 | 0 | 0 | 0 (0) | 0 (0) | 400/400 (1) | 400/400 (1) | 400/400 (1) | 400/400 (1) |

##### 缺失与异常（第 2 节）

- 无

#### 3. 每个读数的墙钟

驱动日志第一行 `=== <时刻> chunk rc=<rc> wall_s=<s>`；各任务 `logs/run_*.json` 的 `started_utc`、`finished_utc`、`wall_s`（跨度 = 最早 started 到最晚 finished）。

| 读数 | done.txt | 驱动 rc | 驱动 wall_s | 驱动结束时刻 | 任务 run json 数 | 最早 started_utc | 最晚 finished_utc | 跨度 s | 各任务 wall_s 之和 | 各任务 wall_s（rotation3d/viewpoint/occlusion/physics/imagery） |
|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 s0 | 是 | 0 | 1968 | 2026-10-14T10:01:00Z | 5 | 2026-10-14T09:36:37+00:00 | 2026-10-14T10:00:18+00:00 | 1421 | 1418 | 250/506/176/232/254 |
| c3_warmup s0 | 是 | 0 | 1830 | 2026-10-14T10:31:31Z | 5 | 2026-10-14T10:08:26+00:00 | 2026-10-14T10:31:16+00:00 | 1370 | 1369 | 248/499/173/229/219 |
| base_warmup s0 | 是 | 0 | 1830 | 2026-10-14T11:02:01Z | 5 | 2026-10-14T10:38:48+00:00 | 2026-10-14T11:01:37+00:00 | 1369 | 1369 | 248/501/173/228/219 |
| e3b s0 | 是 | 0 | 1831 | 2026-10-14T11:32:32Z | 5 | 2026-10-14T11:09:17+00:00 | 2026-10-14T11:32:03+00:00 | 1366 | 1365 | 247/498/173/228/219 |
| e3b_c3 s1 | 是 | 0 | 1800 | 2026-10-14T12:02:32Z | 5 | 2026-10-14T11:39:18+00:00 | 2026-10-14T12:02:03+00:00 | 1365 | 1365 | 247/498/173/228/218 |
| c3_warmup s1 | 是 | 0 | 1830 | 2026-10-14T12:33:02Z | 5 | 2026-10-14T12:09:57+00:00 | 2026-10-14T12:32:48+00:00 | 1371 | 1371 | 248/500/174/229/220 |
| base_warmup s1 | 是 | 0 | 1860 | 2026-10-14T13:04:02Z | 5 | 2026-10-14T12:40:25+00:00 | 2026-10-14T13:03:40+00:00 | 1395 | 1394 | 252/507/176/232/226 |
| e3b_c3 s2 | 是 | 0 | 1800 | 2026-10-14T13:34:03Z | 5 | 2026-10-14T13:10:58+00:00 | 2026-10-14T13:33:51+00:00 | 1373 | 1372 | 251/500/173/229/219 |
| c3_warmup s2 | 是 | 0 | 1830 | 2026-10-14T14:04:33Z | 5 | 2026-10-14T13:41:29+00:00 | 2026-10-14T14:04:08+00:00 | 1359 | 1359 | 246/496/172/228/218 |
| base_warmup s2 | 是 | 0 | 1831 | 2026-10-14T14:35:04Z | 5 | 2026-10-14T14:11:51+00:00 | 2026-10-14T14:34:51+00:00 | 1380 | 1380 | 248/504/175/233/220 |
| e3b s2 | 是 | 0 | 1830 | 2026-10-14T15:05:34Z | 5 | 2026-10-14T14:42:21+00:00 | 2026-10-14T15:05:08+00:00 | 1367 | 1367 | 247/499/173/229/218 |

- 没有 finished_utc 的 run json：无。
- 本地没有驱动日志 `driver/logs/n0_*.log` 的读数：无。
- 跨度只算已结束的任务；任务有缺的读数，跨度与 wall_s 之和不是整个读数的墙钟。


- 15:05:34Z 驱动认领了 `small_s0`，开始 `e3b_c3_n1000_s0`。

## 4. 被拒的评测（评测前合并核对没过；按管理员 10-14 的答复：照规则继续，记进 failed.txt，检查点留着以后补评，不改代码、配置与判据，也不手动补评）

表由 `experiments/E5/0051_e5_main_probe/p51_refusals.py` 读出。每次有新的被拒，就重跑它更新这一节。

| 运行 | 核对写于 | restricted exact_rows（不同行 / 行） | gap 中位 / 最大 | merge pass | restricted 教师强制 agreement | timing_s.total_s | failures | 核对文件 |
|---|---|---|---|---|---|---|---|---|
| e3b_c3/qwen3.5-2b/n1000/seed0 | 2026-10-14T17:01:06+00:00 | 0.825（7 / 40） | 0.1230 / 0.2458 | True | 0.9994560075685903 | 240.31790892407298 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.8250 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/e3b_c3/qwen3.5-2b/n1000/seed0/merge_check.json` |

（上表之后新增，读自 Lustre 上的文件，19:2xZ，同一个脚本）

| c3_warmup/qwen3.5-2b/n1000/seed0 | 2026-10-14T19:19:29+00:00 | 0.825（7 / 40） | 0.0846 / 0.5072 | True | 0.9993850520340587 | 227.45903131365776 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.8250 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/c3_warmup/qwen3.5-2b/n1000/seed0/merge_check.json` |

| base_warmup/qwen3.5-2b/n1000/seed0 | 2026-10-14T21:18:40+00:00 | 0.9（4 / 40） | 0.0359 / 0.1398 | False | 0.9994087038789026 | 229.70642948895693 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9000 < 0.95"], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.97025 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/base_warmup/qwen3.5-2b/n1000/seed0/merge_check.json` |

| e3b/qwen3.5-2b/n1000/seed0 | 2026-10-14T23:20:19+00:00 | 0.9（4 / 40） | 0.0626 / 0.2467 | True | 0.9997398297067172 | 231.42411695420742 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9000 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/e3b/qwen3.5-2b/n1000/seed0/merge_check.json` |

| c3_warmup/qwen3.5-2b/n4000/seed0 | 2026-10-15T03:37:29+00:00 | 0.95（2 / 40） | 0.0307 / 0.0450 | False | 0.9998817407757805 | 229.44479606673121 | {"restricted": [], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.96585 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/c3_warmup/qwen3.5-2b/n4000/seed0/merge_check.json` |

| base_warmup/qwen3.5-2b/n4000/seed0 | 2026-10-15T05:53:18+00:00 | 1.0（0 / 40） | — / — | False | 1.0 | 238.73564265295863 | {"restricted": [], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.95714 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/base_warmup/qwen3.5-2b/n4000/seed0/merge_check.json` |

| e3b_c3/qwen3.5-2b/n1000/seed1 | 2026-10-15T10:26:17+00:00 | 0.875（5 / 40） | 0.2420 / 0.3492 | True | 0.9995506149479659 | 232.28397078812122 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.8750 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/e3b_c3/qwen3.5-2b/n1000/seed1/merge_check.json` |

| c3_warmup/qwen3.5-2b/n1000/seed1 | 2026-10-15T12:31:35+00:00 | 0.95（2 / 40） | 0.2405 / 0.4376 | False | 0.9991958372753075 | 231.05827533453703 | {"restricted": [], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.97214 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/c3_warmup/qwen3.5-2b/n1000/seed1/merge_check.json` |

| base_warmup/qwen3.5-2b/n1000/seed1 | 2026-10-15T14:37:51+00:00 | 0.85（6 / 40） | 0.0986 / 0.3144 | False | 0.9993377483443708 | 227.42181426286697 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.8500 < 0.95"], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.96088 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/base_warmup/qwen3.5-2b/n1000/seed1/merge_check.json` |

| e3b_c3/qwen3.5-2b/n4000/seed1 | 2026-10-15T16:44:34+00:00 | 0.875（5 / 40） | 0.0361 / 0.0519 | True | 0.9997871333964049 | 231.4084831252694 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.8750 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/e3b_c3/qwen3.5-2b/n4000/seed1/merge_check.json` |

| c3_warmup/qwen3.5-2b/n4000/seed1 | 2026-10-15T18:50:24+00:00 | 0.95（2 / 40） | 0.0136 / 0.0149 | False | 0.9998580889309366 | 227.42871184647083 | {"restricted": [], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.97992 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/c3_warmup/qwen3.5-2b/n4000/seed1/merge_check.json` |

| base_warmup/qwen3.5-2b/n4000/seed1 | 2026-10-15T20:57:34+00:00 | 1.0（0 / 40） | — / — | False | 0.9999290444654683 | 227.2140044607222 | {"restricted": [], "merge": ["merge/thinking_off：教师强制的 argmax_agreement 0.96329 < 0.98"]} | `0051_e5_main_hpcgpu07/trained/base_warmup/qwen3.5-2b/n4000/seed1/merge_check.json` |

| e3b_c3/qwen3.5-2b/n1000/seed2 | 2026-10-15T23:00:57+00:00 | 0.925（3 / 40） | 0.0478 / 0.0911 | True | 0.9993614001892147 | 230.26868630200624 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9250 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/e3b_c3/qwen3.5-2b/n1000/seed2/merge_check.json` |

| c3_warmup/qwen3.5-2b/n1000/seed2 | 2026-10-16T01:11:08+00:00 | 0.875（5 / 40） | 0.0659 / 0.2667 | True | 0.9994796594134342 | 227.82497269287705 | {"restricted": ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.8750 < 0.95"], "merge": []} | `0051_e5_main_hpcgpu07/trained/c3_warmup/qwen3.5-2b/n1000/seed2/merge_check.json` |

- **eval_e3b_c3_n1000_s0（hpcgpu07）：**
  - 训练 16:55:07Z 完成：1000 步，6,572 s，每步中位 5.088 s，峰值 9.35 GB；检查点 `$VI/ckpt/hpcgpu07/0051/train/e3b_c3/qwen3.5-2b/n1000/seed0/ckpt/final`，131,087 KB。
  - 评测两次都是 rc=1：17:01:07Z（命令 360 s）、17:05:37Z。核对文件在第 2 次时重写，写于 17:05:07Z，失败项相同（0.8250）。上表读的是 origin 上第 1 次写的那份。
  - `failed.txt` 原文：`2026-10-14T17:05:37Z eval_e3b_c3_n1000_s0 rc=1`。驱动随后跳到 `c3_warmup_n1000_s0`。
  - 这个起点（e3b_c3 种子 0）在 n0_all 时的起点核对 exact_rows 是 0.985。
- **eval_c3_warmup_n1000_s0（hpcgpu07）：**
  - 训练 19:03:23Z 完成，检查点 `…/0051/train/c3_warmup/qwen3.5-2b/n1000/seed0/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：19:15:29Z（726 s）、19:19:59Z（270 s）。`failed.txt` 原文 `2026-10-14T19:19:59Z eval_c3_warmup_n1000_s0 rc=1`。驱动随后跳到 `base_warmup_n1000_s0`。
  - 这个起点（c3_warmup 种子 0）在 n0_all 时的起点核对 exact_rows 是 1.0。
- 截至 19:27Z：训练后的评测前核对做了 2 次，2 次没过，exact_rows 都是 0.825（7 / 40），merge 都通过。已报告管理员（只陈述）。
- **eval_base_warmup_n1000_s0（hpcgpu07）：**
  - 训练 21:09:09Z 完成，检查点 `…/0051/train/base_warmup/qwen3.5-2b/n1000/seed0/ckpt/final`，131,087 KB。
  - 评测两次 rc=1：21:14:39Z（330 s）、21:19:09Z（270 s）。`failed.txt` 原文 `2026-10-14T21:19:09Z eval_base_warmup_n1000_s0 rc=1`。驱动随后跳到 `e3b_n1000_s0`。
  - **这次 merge 也没过**：`merge/thinking_off：教师强制的 argmax_agreement 0.97025 < 0.98`。restricted 是 0.900（4 / 40）。
  - 这个起点（base_warmup 种子 0）在 n0_all 时的起点核对是 pass，exact_rows 1.0。
- **eval_e3b_n1000_s0（hpcgpu07）：**
  - 训练 23:08:16Z 完成，检查点 `…/0051/train/e3b/qwen3.5-2b/n1000/seed0/ckpt/final`，131,087 KB。
  - 评测两次 rc=1：23:16:16Z、23:20:46Z。`failed.txt` 原文 `2026-10-14T23:20:46Z eval_e3b_n1000_s0 rc=1`。驱动随后跳到 `e3b_c3_n4000_s0`。
  - restricted 是 0.900（4 / 40），merge 通过。这个起点（e3b 种子 0）在 n0_all 时的起点核对 exact_rows 是 0.95。
- 截至 23:33Z：small_s0 的 4 个 1K 运行训练都已完成，它们的 test 评测 4 次都被拒（`failed.txt` 共 4 行）。
- **eval_e3b_c3_n4000_s0（hpcgpu07）：评测前的合并核对通过，评测完成**（10-15 01:37:22Z）。
  - 训练 01:09:22Z 完成，检查点 131,091 KB。
  - 核对写于 01:14:02Z：pass True；restricted exact_rows 0.95（2 / 40），gap 中位 0.0357、最大 0.0557；merge 的教师强制 argmax_agreement 0.99506；timing_s.total_s 233.8。
  - 这是 RUN-0051 里第一条评测通过的训练运行。
- **eval_c3_warmup_n4000_s0（hpcgpu07）：**
  - 训练 03:26:58Z 完成，检查点 131,091 KB。
  - 评测两次 rc=1：03:33:28Z、03:37:58Z。`failed.txt` 原文 `2026-10-15T03:37:58Z eval_c3_warmup_n4000_s0 rc=1`。
  - **这次 restricted 通过**（0.95，2 / 40），**只有 merge 没过**：`merge/thinking_off：教师强制的 argmax_agreement 0.96585 < 0.98`。
  - 驱动随后跳到 `base_warmup_n4000_s0`。
- **eval_base_warmup_n4000_s0（hpcgpu07）：**
  - 训练 05:28:38Z 完成，检查点 131,091 KB。
  - 评测两次 rc=1：05:49:07Z（1,229 s）、05:53:37Z（270 s）。`failed.txt` 原文 `2026-10-15T05:53:37Z eval_base_warmup_n4000_s0 rc=1`。驱动随后跳到 `e3b_n4000_s0`。
  - restricted 通过（1.0，0 / 40 行不同），**只有 merge 没过**：`merge/thinking_off：教师强制的 argmax_agreement 0.95714 < 0.98`。
  - `p51_refusals.py` 的第一版在没有分歧行时（gap 的 median、max 为 None）格式化出错，漏了这一行；已修，表里的这一行是修后的脚本读出的。
- **eval_e3b_n4000_s0（hpcgpu07）：评测前的合并核对通过，评测完成**（10-15 08:24:32Z）。
  - 训练 07:47:30Z 完成，检查点 131,091 KB。
  - 核对写于 07:59:09Z：pass True；restricted exact_rows 0.975（1 / 40）；merge 的教师强制 argmax_agreement 0.99018。
- **组 small_s0 08:24:32Z 跑完**（8 个训练运行全部完成；评测通过 2 个：e3b_c3 n4000、e3b n4000；被拒 6 个，`failed.txt` 6 行）。驱动同一时刻认领组 `small_s1`。
- **eval_e3b_c3_n1000_s1（hpcgpu07）：**
  - 训练 10:13:43Z 完成，检查点 `…/0051/train/e3b_c3/qwen3.5-2b/n1000/seed1/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：10:22:13Z、10:26:43Z。`failed.txt` 原文 `2026-10-15T10:26:43Z eval_e3b_c3_n1000_s1 rc=1`。驱动随后跳到 `c3_warmup_n1000_s1`。
  - restricted 0.875（5 / 40），gap 中位 0.2420、最大 0.3492；merge 通过（merge 的教师强制 argmax_agreement 0.99489）。
- **eval_c3_warmup_n1000_s1（hpcgpu07）：**
  - 训练 12:16:55Z 完成，检查点 `…/0051/train/c3_warmup/qwen3.5-2b/n1000/seed1/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：12:27:32Z、12:32:02Z。`failed.txt` 原文 `2026-10-15T12:32:02Z eval_c3_warmup_n1000_s1 rc=1`。驱动随后跳到 `base_warmup_n1000_s1`。
  - restricted 通过（0.95，2 / 40；gap 中位 0.2405、最大 0.4376），**只有 merge 没过**：`merge/thinking_off：教师强制的 argmax_agreement 0.97214 < 0.98`。
- 截至 14:23Z：`failed.txt` 共 8 行；驱动在跑 `base_warmup_n1000_s1`。
- **eval_base_warmup_n1000_s1（hpcgpu07）：**
  - 训练 14:27:52Z 完成，检查点 `…/0051/train/base_warmup/qwen3.5-2b/n1000/seed1/ckpt/final`，131,087 KB。
  - 评测两次 rc=1：14:33:52Z、14:38:22Z。`failed.txt` 原文 `2026-10-15T14:38:22Z eval_base_warmup_n1000_s1 rc=1`。
  - restricted 0.85（6 / 40；gap 中位 0.0986、最大 0.3144）与 merge（教师强制 argmax_agreement 0.96088 < 0.98）都没过。
  - 驱动随后跳到 `e3b_c3_n4000_s1`。组 small_s1 里没有 `e3b_n1000_s1`：queue.sh 第 5 行写明 e3b 起点只用种子 0、2（种子 1 在 RUN-0048 没过合并核对），与 n0_all 里没有 `n0_e3b_s1` 一致。
- 截至 14:39Z：`failed.txt` 共 9 行。
- **eval_e3b_c3_n4000_s1（hpcgpu07）：**
  - 训练 16:29:30Z 完成，检查点 `…/0051/train/e3b_c3/qwen3.5-2b/n4000/seed1/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：16:40:30Z、16:45:01Z。`failed.txt` 原文 `2026-10-15T16:45:01Z eval_e3b_c3_n4000_s1 rc=1`。驱动随后跳到 `c3_warmup_n4000_s1`。
  - restricted 0.875（5 / 40；gap 中位 0.0361、最大 0.0519）没过；merge 通过（merge 的教师强制 argmax_agreement 0.99598）。
- 截至 16:47Z：`failed.txt` 共 10 行。
- **eval_c3_warmup_n4000_s1（hpcgpu07）：**
  - 训练 18:36:54Z 完成，检查点 `…/0051/train/c3_warmup/qwen3.5-2b/n4000/seed1/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：18:46:24Z、18:50:54Z。`failed.txt` 原文 `2026-10-15T18:50:54Z eval_c3_warmup_n4000_s1 rc=1`。驱动随后跳到 `base_warmup_n4000_s1`。
  - restricted 通过（0.95，2 / 40；gap 中位 0.0136、最大 0.0149），**只有 merge 没过**：merge 的教师强制 argmax_agreement 0.9799195837275307 < 0.98。
- 截至 19:04Z：`failed.txt` 共 11 行。
- **eval_base_warmup_n4000_s1（hpcgpu07）：**
  - 训练 20:41:35Z 完成，检查点 `…/0051/train/base_warmup/qwen3.5-2b/n4000/seed1/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：20:53:35Z、20:57:35Z。`failed.txt` 原文 `2026-10-15T20:57:35Z eval_base_warmup_n4000_s1 rc=1`。
  - restricted 通过（1.0，0 / 40 行不同），**只有 merge 没过**：merge 的教师强制 argmax_agreement 0.9632923368022706 < 0.98。
- **组 small_s1 20:57:35Z 跑完**：从 08:24:32Z 认领起 45183 s（12.55 h）；6 个训练运行（e3b_c3、c3_warmup、base_warmup × 1K、4K；没有 e3b，见上）全部完成，评测通过 0 个、被拒 6 个。驱动同一时刻认领组 `small_s2`。
- 截至 21:06Z：`failed.txt` 共 12 行；驱动在跑 `e3b_c3_n1000_s2`。
- **eval_e3b_c3_n1000_s2（hpcgpu07）：**
  - 训练 22:48:55Z 完成，检查点 `…/0051/train/e3b_c3/qwen3.5-2b/n1000/seed2/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：22:56:55Z、23:01:25Z。`failed.txt` 原文 `2026-10-15T23:01:25Z eval_e3b_c3_n1000_s2 rc=1`。驱动随后跳到 `c3_warmup_n1000_s2`。
  - restricted 0.925（3 / 40；gap 中位 0.0478、最大 0.0911）没过；merge 通过（merge 的教师强制 argmax_agreement 0.99584）。
- 截至 23:21Z：`failed.txt` 共 13 行。
- **eval_c3_warmup_n1000_s2（hpcgpu07）：**
  - 训练 10-16 00:53:08Z 完成，检查点 `…/0051/train/c3_warmup/qwen3.5-2b/n1000/seed2/ckpt/final`，131,087 KB。
  - 评测两次 rc=1：01:07:08Z、01:11:38Z。`failed.txt` 原文 `2026-10-16T01:11:38Z eval_c3_warmup_n1000_s2 rc=1`。驱动随后跳到 `base_warmup_n1000_s2`。
  - restricted 0.875（5 / 40；gap 中位 0.0659、最大 0.2667）没过；merge 通过（merge 的教师强制 argmax_agreement 0.98335）。
- 截至 10-16 01:22Z：`failed.txt` 共 14 行。
- 管理员 10-14 的答复：判据交用户与负责人定（N34），队列不调整，照原样继续、继续记录，被拒的检查点留着以后统一补评。

## addendum-1（判据 e5-v2，D96、D97；任务书 `prompts/server_hpc/0051-addendum-1.md`，f4d4f53b）

### 暂停

- squeue 10-16 02:37:32Z：157296、157297 都是 RUNNING。
- 02:37:32Z 写 `$VI/logs/train.pause.hpcgpu07`，内容 `RUN-0051-addendum-1: switch to criteria e5-v2 and re-eval (D96)`。
- 主驱动 03:01:20Z 进入 paused：停在 `base_warmup_n1000_s2 完成`（检查点 131,091 KB）之后，它的评测 `eval_base_warmup_n1000_s2` 还没跑。
- 停住时：`failed.txt` 14 行，`done.txt` 30 行。
- `$VI/logs/train.pause.hpcgpu06` 也存在：写于 10-14 13:21:32Z，内容 `0052-hpc-06-4b.md (a)`。它是 RUN-0052 段 a 的评测驱动按 `eval_lib.sh` 写的（`EV_PAUSE=$LOGS/train.pause.$NODE`）；本任务没碰它。

### 切换 wt0051

- 切换前 HEAD `07d677df0f0cd79c5f04e58b67e557e2ea18d6c1`，status 空。07 的 GPU 上没有计算进程。
- fetch rc=0；`checkout --detach f703d7dbb25817d097a574ebb0e61cd3cbc91f57` rc=0；切换后 HEAD `f703d7dbb25817d097a574ebb0e61cd3cbc91f57`，status 空；`experiments/` 只有 P0。
- sha256（切换后）：

| 文件 | sha256 | 与任务书 |
|---|---|---|
| `prompts/server_hpc/0051-hpc-e5-main/reeval.sh` | `7593ec6853ea5909a413877050827f292ebb5db6828cb6788a7977a83aa1d8ba` | 相同 |
| `src/e5/trained_eval.py` | `25a0596bb59f2e298339fdb76a249317d4a00dba15e688cc80eb81bdf5fdb2fe` | 相同 |
| `scripts/eval_e5_trained.py` | `90dcb73889cbec320d5c79770cc2b892e46a1f26f0d411fc4e4f74fb5a7b25a7` | 相同 |
| `prompts/server_hpc/0051-hpc-e5-main/queue.sh` | `8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769` | 相同，切换前也是这个值 |
| `scripts/server/train_queue.sh` | `fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa` | 相同，切换前也是这个值 |

### 测试（第 4 步，按任务书原命令）

- 07 上经 srun，03:05:21Z – 03:22:21Z（995.30 s）。`pytest_rc=1`。之后删了 `$VI/tmp/pt_0051a1`；wt0051 status 仍为空；前后 squeue 都是两条 RUNNING。
- 结果原文（尾部）：

```
FAILED tests/test_e5_merge_check_e5v2.py::test_check_only_cli_fail_exit1 - OS...
ERROR tests/test_e5_merge_check_e5v2.py::test_metric_level_exact_and_lora_in_path
ERROR tests/test_e5_merge_check_e5v2.py::test_metric_level_error_does_not_affect_pass
ERROR tests/test_e5_merge_check_e5v2.py::test_check_only_cli - OSError: We co...
ERROR tests/test_e5_merge_check_e5v2.py::test_structured_eval_identical_to_4bd7f13
1 failed, 59 passed, 4 errors in 995.30s (0:16:35)
```

- 5 条都是 `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.`，之前是对 `https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF/resolve/3f301ffcf3dcbb47893afae6650ea3e78d96fb6d/config.json` 的 HEAD 请求超时重试 5 次。
- 相关事实：
  - 任务书第 4 步的命令只设了 `PATH`，没有设 `HF_HOME`、`HF_HUB_CACHE`、`HF_HUB_OFFLINE`。
  - `$VI/hf/hub/models--OpenGVLab--InternVL3_5-2B-HF/snapshots/3f301ffcf3dcbb47893afae6650ea3e78d96fb6d/config.json` 存在。
  - 本集群上以前的测试与驱动脚本都 export 了 `HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1`（如 `driver_0044.sh` 第 22–24 行）；`reeval.sh` 第 37 行、`queue.sh` 第 40 行也自己 export 这三个。
- 按任务书：测试不过就停，暂停文件留着，主驱动保持暂停。已把输出发给管理员（10-16 03:2xZ），等指示。

### 测试重跑（管理员 03:2xZ 指示：加 HF 环境变量，其余同第 4 步）

- 先查 `~/.cache/huggingface`（只看不删）：`find ~/.cache/huggingface -newermt '2026-10-16 03:05'`（节点本地时间 CST）与 `-newermt '2026-10-16 03:05 UTC'` 都没有输出；顶层最新的是 `hub/`，2026-08-08 16:44:30 +0800。
- 命令：第 4 步的命令在 `env` 里加 `HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp`，pytest 行不变。
- 03:24:42Z – 03:30:30Z：`64 passed in 345.53s (0:05:45)`，`pytest_rc=0`。之后删了 `$VI/tmp/pt_0051a1`；wt0051 status 空；前后 squeue 两条 RUNNING。

### 启动补评（第 5 步）

- 03:31:05Z 按任务书命令启动 `reeval.sh`（`EXPECT_COMMIT=f703d7db…`，`QUEUE_CAP_H=14`），step 157297.87。启动前后 squeue 两条 RUNNING；主驱动的 step 157297.84 仍在，心跳 paused。
- 驱动日志第一行：`2026-10-16T03:31:06Z 补评开始：hpcgpu07，commit f703d7dbb25817d097a574ebb0e61cd3cbc91f57；名单 …/reeval_e5v2/plan.tsv：补评 14 个、只核对 2 个`。
- `plan.tsv`（kind、tag；checkpoint 都是 `$VI/ckpt/hpcgpu07/0051/train/<起点>/qwen3.5-2b/n<N>/seed<k>/ckpt/final`，model_dir 都是 `…/0051_e5_main_hpcgpu07/trained/<起点>/qwen3.5-2b/n<N>/seed<k>`）：

| kind | tag |
|---|---|
| reeval | e3b_c3_n1000_s0 |
| reeval | c3_warmup_n1000_s0 |
| reeval | base_warmup_n1000_s0 |
| reeval | e3b_n1000_s0 |
| reeval | c3_warmup_n4000_s0 |
| reeval | base_warmup_n4000_s0 |
| reeval | e3b_c3_n1000_s1 |
| reeval | c3_warmup_n1000_s1 |
| reeval | base_warmup_n1000_s1 |
| reeval | e3b_c3_n4000_s1 |
| reeval | c3_warmup_n4000_s1 |
| reeval | base_warmup_n4000_s1 |
| reeval | e3b_c3_n1000_s2 |
| reeval | c3_warmup_n1000_s2 |
| check | e3b_c3_n4000_s0 |
| check | e3b_n4000_s0 |

### 第一个补评（eval_e3b_c3_n1000_s0）

- 04:02:36Z `eval_e3b_c3_n1000_s0 完成`（从 03:31:06Z 起 31.5 min）。核对文件 `trained/e3b_c3/qwen3.5-2b/n1000/seed0/merge_check.json` 写于 03:39:28Z；旧文件留作同目录的 `merge_check.v1.json`。
- criteria_version `e5-v2`；pass true；failures `{"merge": [], "restricted": []}`。
- v1_report_only.pass false，failures 同 v1 原判（restricted exact_rows 0.8250 < 0.95）。
- 判定用到的读数（thinking_off）：restricted.across_merge 教师强制 argmax_agreement 0.99920；restricted 教师强制 agreement 0.99946。
- 各比较的 gap（n / 中位 / 最大）：

| 比较 | n | 中位 | 最大 |
|---|---|---|---|
| merge.generate | 21 | 0.2001 | 1.1876 |
| merge.teacher_forcing | 222 | 0.1449 | 0.5740 |
| restricted.decode_vs_generate_batch | 7 | 0.1230 | 0.2458 |
| restricted.teacher_forcing | 23 | 0.0862 | 0.2956 |
| restricted.across_merge.teacher_forcing | 34 | 0.1947 | 1.1876 |

- metric_level（val 40 行，只报告）：status done，error null；A 侧 LoRA 审计 186 / 186 层被调用；A 104.5 s、B 58.3 s。
  - 总体差（B − A）：all（n 40）iou_grid −0.00460，relief_error（16 行）+0.05956，相同 21 行，无效 0 / 0；headline（n 31）iou_grid −0.00692，relief_error（7 行）+0.13782。
  - flags：iou false，relief **true**（规则：总体起伏误差平均差的绝对值 > 0.02）。只报告，不影响判定。
- timing_s：unmerged 134.2 s，merged 125.0 s，total 422.0 s。
- 已发管理员（只核对摘要，不含 test 分数）。

### 补评跑完

- 驱动日志末尾原文：
  - `2026-10-16T10:51:45Z mcheck_e3b_n4000_s0 完成`
  - `2026-10-16T10:51:46Z 删掉主驱动的暂停文件 /home/mail/chenyuma/visint/logs/train.pause.hpcgpu07：主驱动接着跑（新判据 e5-v2）`
  - `2026-10-16T10:51:46Z 队列跑完：完成 16 条，失败 0 条`
- 心跳 `train_0051r-hpcgpu07.status`：`2026-10-16T10:51:46Z done 0051r-hpcgpu07`。
- 补评脚本墙钟：03:31:06Z – 10:51:46Z，26440 s（7.34 h）。上限 14 h。
- `reeval_e5v2/driver/failed.txt`：不存在（没有失败）。`reeval_e5v2/driver/done.txt` 16 行。
- `$VI/logs/train.pause.hpcgpu07` 已不存在（11:45Z 查）。
- `summary.tsv` 原文（制表符换成表格；`metric_error` 列都是空串，写作"（空）"；base_warmup_n4000_s0 的 `d_relief_error` 是空串）：

| kind | tag | criteria_version | pass_e5v2 | pass_v1_report_only | metric_status | metric_error | d_iou_grid_all | d_iou_grid_binary | d_relief_error | flag_iou | flag_relief | n_rows | n_identical | previous |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reeval | e3b_c3_n1000_s0 | e5-v2 | True | False | done | （空） | -0.004598861179948233 | -0.004188249974329705 | 0.059562488517258316 | False | True | 40 | 21 | merge_check.v1.json |
| reeval | c3_warmup_n1000_s0 | e5-v2 | True | False | done | （空） | 0.00218260736620407 | 0.0010769489757846695 | 0.017121349868830293 | False | False | 40 | 18 | merge_check.v1.json |
| reeval | base_warmup_n1000_s0 | e5-v2 | True | False | done | （空） | -0.013901798468439669 | -0.009202124973415035 | 0.0479762990726158 | True | True | 40 | 23 | merge_check.v1.json |
| reeval | e3b_n1000_s0 | e5-v2 | True | False | done | （空） | -0.0020743173209399335 | -0.007703318637289958 | 0.04800085736704718 | False | True | 40 | 23 | merge_check.v1.json |
| reeval | c3_warmup_n4000_s0 | e5-v2 | True | False | done | （空） | -0.002219672982324103 | -0.002026890398228065 | 0.036017467733472586 | False | True | 40 | 33 | merge_check.v1.json |
| reeval | base_warmup_n4000_s0 | e5-v2 | True | False | done | （空） | 0.0 | 0.0 | （空） | False | None | 40 | 40 | merge_check.v1.json |
| reeval | e3b_c3_n1000_s1 | e5-v2 | True | False | done | （空） | -0.00014440841919410863 | -0.0036134520858152053 | -0.05357530142646283 | False | True | 40 | 25 | merge_check.v1.json |
| reeval | c3_warmup_n1000_s1 | e5-v2 | True | False | done | （空） | -0.011498228071617433 | -0.009252063819812362 | 0.08827776170801371 | True | True | 40 | 20 | merge_check.v1.json |
| reeval | base_warmup_n1000_s1 | e5-v2 | True | False | done | （空） | 0.004059430065927155 | -0.0006779421827273618 | 0.10691560279311885 | False | True | 40 | 21 | merge_check.v1.json |
| reeval | e3b_c3_n4000_s1 | e5-v2 | True | False | done | （空） | -0.00559092663373828 | -0.011477857564953786 | 0.005909008284409767 | False | False | 40 | 30 | merge_check.v1.json |
| reeval | c3_warmup_n4000_s1 | e5-v2 | True | False | done | （空） | 0.0006085423666096768 | 0.0010142372776827668 | 0.0 | False | False | 40 | 34 | merge_check.v1.json |
| reeval | base_warmup_n4000_s1 | e5-v2 | True | False | done | （空） | -0.004444896380175568 | -0.004320987654320985 | 0.0702895363792777 | False | True | 40 | 37 | merge_check.v1.json |
| reeval | e3b_c3_n1000_s2 | e5-v2 | True | False | done | （空） | 0.0005332569817451116 | 0.0037660037011253422 | -0.0965767014422454 | False | True | 40 | 16 | merge_check.v1.json |
| reeval | c3_warmup_n1000_s2 | e5-v2 | True | False | done | （空） | -0.003017383370415877 | 0.002605989813063217 | -0.09276389220030978 | False | True | 40 | 16 | merge_check.v1.json |
| check | e3b_c3_n4000_s0 | e5-v2 | True | True | done | （空） | -0.00518220275574055 | -0.008637004592900899 | 0.0 | False | False | 40 | 34 | merge_check.v1.json |
| check | e3b_n4000_s0 | e5-v2 | True | True | done | （空） | 0.0041304117904813475 | 0.004470896821883397 | -0.03796600177884102 | False | True | 40 | 34 | merge_check.v1.json |

- 计数（由上表）：pass_e5v2 True 16 / 16；pass_v1_report_only True 2 / 16（两个 check 行）；metric_status done 16 / 16；flag_iou True 2 个；flag_relief True 11 个。

### 主驱动恢复

- 主驱动 step 157297.84 一直在（暂停期间没有杀、没有重启）。
- 恢复后第一条：`eval_base_warmup_n1000_s2`（暂停前已训练完的那个）。`2026-10-16T11:21:54Z eval_base_warmup_n1000_s2 完成`，chunk wall_s 1807。
  - 核对文件 `trained/base_warmup/qwen3.5-2b/n1000/seed2/merge_check.json` 写于 10:58:32Z：criteria_version e5-v2，pass True，failures `{"merge": [], "restricted": []}`，v1_report_only.pass False。
  - stderr 那一行原文（`driver/logs/eval_base_warmup_n1000_s2.log` 第 5 行）：`[合并核对·指标层面，只报告] 总体 iou_grid 差（B − A）+0.0007，起伏误差差 -0.0035；预测相同 23/40 行；flags.iou=False flags.relief=False`
- 主驱动 `done.txt` 45 行（暂停时 30 行 + 补评的 14 个 + eval_base_warmup_n1000_s2）；`failed.txt` 仍是 14 行（v1 时的记录，没有删改）。
- 11:41:54Z 心跳：`running e3b_n1000_s2`。

### 恢复后（e5-v2 下）small_s2 的进度（10-16 23:07Z 读）

- 驱动日志（"完成"行原文摘要）：
  - 11:21:54Z eval_base_warmup_n1000_s2 完成
  - 13:13:34Z e3b_n1000_s2 完成（检查点 131,087 KB）；13:50:23Z eval_e3b_n1000_s2 完成
  - 15:43:02Z e3b_c3_n4000_s2 完成（131,087 KB）；16:22:53Z eval_e3b_c3_n4000_s2 完成
  - 18:12:39Z c3_warmup_n4000_s2 完成（131,103 KB）；18:46:40Z eval_c3_warmup_n4000_s2 完成
  - 20:35:22Z base_warmup_n4000_s2 完成（131,091 KB）；21:07:22Z eval_base_warmup_n4000_s2 完成
  - 22:56:33Z e3b_n4000_s2 完成（131,087 KB）；23:06:33Z 心跳 `running eval_e3b_n4000_s2`
- 恢复后没有新的被拒：主驱动 `failed.txt` 仍是 14 行（v1 时的）。
- 种子 2 各运行的核对文件（e5-v2；metric_level 只报告；B − A 的总体差，val 40 行）：

| 运行 | 核对写于 | 判据 | pass | v1 判定（只报告） | metric_level | flag iou | flag relief | iou_grid 差 | 起伏误差差 | 预测相同行 |
|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 n1000 s2 | 09:42:54Z（补评） | e5-v2 | True | False | done | False | True | +0.00053 | −0.09658 | 16 |
| c3_warmup n1000 s2 | 10:14:06Z（补评） | e5-v2 | True | False | done | False | True | −0.00302 | −0.09276 | 16 |
| base_warmup n1000 s2 | 10:58:32Z | e5-v2 | True | False | done | False | False | +0.00071 | −0.00354 | 23 |
| e3b n1000 s2 | 13:24:38Z | e5-v2 | True | False | done | False | True | +0.00709 | +0.06390 | 25 |
| e3b_c3 n4000 s2 | 15:57:02Z | e5-v2 | True | True | done | False | True | +0.00204 | −0.18337 | 36 |
| c3_warmup n4000 s2 | 18:22:53Z | e5-v2 | True | True | done | False | None | −0.00272 | None | 35 |
| base_warmup n4000 s2 | 20:43:56Z | e5-v2 | True | False | done | False | None | 0.0 | None | 40 |
| e3b n4000 s2 | 23:07:20Z | e5-v2 | True | True | done | **True** | None | −0.01386 | None | 36 |

- 23:30:33Z `eval_e3b_n4000_s2 完成`。
- **组 small_s2 23:30:33Z 跑完**：从 10-15 20:57:35Z 认领起 95,578 s（26.55 h），其中含主驱动暂停的 03:01:20Z – 10:51:46Z（28,226 s，7.84 h；这段时间 GPU 跑的是补评脚本）。
  - 8 个训练运行（e3b_c3、c3_warmup、base_warmup、e3b × 1K、4K）全部完成。8 个评测都完成：e3b_c3 n1000、c3_warmup n1000 暂停前被 v1 拒绝，在补评里做完；其余 6 个在恢复后由主驱动在 e5-v2 下做完，没有被拒。
- 驱动同一时刻认领组 `mid_s0`；23:35:33Z 心跳 `running e3b_c3_n16000_s0`。
- `failed.txt` 仍是 14 行。

### mid_s0

- **e3b_c3_n16000_s0（第一个 16K 运行）：**
  - 训练 10-17 02:25:50Z 完成（从 10-16 23:30:33Z 认领组起），检查点 131,087 KB。
  - `accounting.json`：n_runs 2，n_steps_logged 2000，n_samples 32,000，wall_s 10,259，显存峰值 9.35 GB；各段 step_s 合起来中位 5.071 s。
  - `train_summary.json`：complete True，steps 2000，epochs 2，min_steps None，selected_step 2000，selection None，final_loss 0.03805，n_skipped 0。
  - 对照同起点 4K 种子 0：n_runs 1，steps 1000，epochs 2，min_steps 1000，n_samples 16,000，wall_s 6,473，step 中位 5.078 s，峰值 9.35 GB，selected_step 1000。
  - 核对（e5-v2）写于 02:37:28Z：pass True，v1_report_only.pass True；metric_level done，flags iou False、relief None；iou_grid 差 +0.00374，起伏误差差 None，预测相同 35 / 40 行。
  - 03:01:20Z `eval_e3b_c3_n16000_s0 完成`。
- 03:21:21Z 心跳：`running c3_warmup_n16000_s0`。`failed.txt` 仍是 14 行。

- **c3_warmup_n16000_s0：** 训练 05:54:32Z 完成（131,087 KB）；n_runs 2，steps 2000，epochs 2，selected_step 2000，final_loss 0.03953，wall_s 10,230，step 中位 5.073 s，峰值 9.35 GB，n_skipped 0。核对 06:04:47Z：e5-v2 pass True，v1 True，metric done，flags iou False、relief None，iou_grid 差 −0.00401，相同 35 / 40；timing total 407.3 s。06:28:03Z 评测完成。
- **base_warmup_n16000_s0：**
  - 训练 10:10:20Z 完成；n_runs 2，steps 2000，epochs 2，selected_step 2000，final_loss 0.04041，wall_s 11,557，step 中位 5.074 s，峰值 9.35 GB，n_skipped 0。
  - 驱动日志写的检查点大小是 **69,938 KB**（其余运行都是约 131,087 KB）。这个数是 `train_queue.sh` 第 178 行在复制后立即 `du -sk "$dst"` 得到的。
  - 11:5xZ 复查：`du -sk` 131,780 KB（e3b_c3、c3_warmup n16000 都是 131,776 KB）；`train_summary.json` 的 final_checkpoint_bytes 134,908,499（另两个是 134,908,489、134,908,508）；`model/` 下 7 个文件逐个与 e3b_c3 n16000 的同名文件字节数相同（`lora/adapter_model.safetensors` 都是 134,609,728）。文件 mtime 10:04:20Z – 10:04:26Z（节点本地 18:04）。
  - 核对 11:06:01Z：e5-v2 pass True，v1 True，metric done，flags iou False、relief None，iou_grid 差 +0.00377，相同 38 / 40；timing unmerged 236.1 s、merged 129.3 s、total 529.3 s（另两个 16K 的 unmerged 是 131.9、118.1 s）。
  - 11:34:31Z 评测完成：从训练完成起 84 min（另两个 16K 是 35.5、33.5 min）。
- **组 mid_s0 11:34:32Z 跑完**：从 10-16 23:30:33Z 认领起 43,439 s（12.07 h）。3 个训练运行（e3b_c3、c3_warmup、base_warmup × 16K；queue.sh 的 mid 组只用 STARTS，没有 e3b）全部完成、评测全部完成，没有被拒。
- 驱动同一时刻认领组 `mid_s1`；11:54:40Z 心跳 `running e3b_c3_n16000_s1`。`failed.txt` 仍是 14 行。

### mid_s1

| 运行 | 训练完成 | 评测完成 | n_runs | steps | epochs | selected_step | final_loss | wall_s | step 中位 s | 峰值 GB | 核对写于 | e5-v2 pass | v1（只报告） | flag iou | flag relief | iou_grid 差 | 起伏误差差 | 相同行 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e3b_c3 n16000 s1 | 16:09:35Z | 16:46:36Z | 2 | 2000 | 2 | 2000 | 0.04405 | **14,200** | 5.094 | 9.35 | 16:22:59Z | True | True | False | None | −0.00074 | None | 35 |
| c3_warmup n16000 s1 | 19:39:49Z | 20:12:50Z | 2 | 2000 | 2 | 2000 | 0.04856 | 10,230 | 5.078 | 9.35 | 19:49:23Z | True | True | False | True | +0.00209 | +0.04306 | 32 |
| base_warmup n16000 s1 | 23:08:07Z | 23:42:37Z | 2 | 2000 | 2 | 2000 | 0.05025 | 10,220 | 5.073 | 9.35 | 23:19:22Z | True | True | False | None | +0.00001 | None | 39 |

- 检查点：e3b_c3 131,091 KB，另两个 131,087 KB（驱动日志）；n_skipped 都是 0。
- e3b_c3 n16000 s1 的 wall_s 14,200，比同组另两个和 mid_s0 的三个（10,220 – 11,557）长；它的训练时段（11:34Z – 16:09Z）与 06 上 e3b_s1 第 7 段启动慢的时段（约 11:16Z – 14:18Z）重叠。每步中位 5.094 s 与其他运行相同。
- **组 mid_s1 23:42:37Z 跑完**：从 11:34:32Z 认领起 43,685 s（12.13 h）；3 个运行训练、评测全部完成，没有被拒。`failed.txt` 仍是 14 行。
- 驱动同一时刻认领组 `mid_s2`，开始 `e3b_c3_n16000_s2`（chunk 1）。

## addendum-2（D100；任务书 `prompts/server_hpc/0051-addendum-2.md`，59df4ebe）

### 第一部分：big_s0 预认领给 hpcgpu06

- 10-18 01:30:51Z（07 在跑 mid_s2 的 e3b_c3_n16000_s2）。前后 squeue 两条 RUNNING。
- 之前 `$VI/logs/e5_claims_0051/`：n0_all、small_s0–2、mid_s0–2，node 都是 hpcgpu07；mid_s2 没有 done，其余都有；没有 big_s0。
- `mkdir big_s0` 成功；`node` 内容：`hpcgpu06 2026-10-18T01:30:51Z pre-claimed by manager (D100, 0051-addendum-2)`。

### 第二部分：12 个未合并补评（D99-2）

- 06:45:24Z 写 `$VI/logs/train.pause.hpcgpu07`，内容 `RUN-0051 addendum-2: unmerged re-evals (D99-2, D100)`（之前没有任何 train.pause* 文件；07 心跳 06:40:44Z `running base_warmup_n16000_s2 (chunk 1)`）。
- 主驱动 08:07:15Z `paused (pause file: …)`：在 base_warmup_n16000_s2 第 1 段之后（驱动每段前都查暂停文件，管理员 10-18 更正）。07 GPU 上没有计算进程；`failed.txt` 14 行，`done.txt` 71 行。
  - mid_s2 中此前完成：e3b_c3_n16000_s2（02:35:57Z 训完，131,091 KB；03:08:57Z 评测完成），c3_warmup_n16000_s2（06:06:13Z 训完，131,091 KB；06:40:43Z 评测完成）。
  - `e5_claims_0051/`：big_s0 node hpcgpu06（预认领）、没有 done；mid_s2 node hpcgpu07、没有 done；n0_all、small_s0–2、mid_s0–1 node hpcgpu07、有 done；还没有 ad_s0。
- 切换 wt0051：切换前 HEAD `f703d7dbb25817d097a574ebb0e61cd3cbc91f57`，status 空；fetch、checkout rc=0；切换后 HEAD `d4c8f67a8c45e4f3ff7f77adf501a184c7c9d763`，status 空，`experiments/` 只有 P0。sha256，5 个都与任务书相同：

| 文件 | sha256 | 切换前 |
|---|---|---|
| `prompts/server_hpc/0051-hpc-e5-main/unmerged.sh` | `5905d0310ae75019ca3661b33c559a016a07be9d33c8d1de3add8dbe9ba223ff` | 不存在 |
| `prompts/server_hpc/0051-hpc-e5-main/queue.sh` | `8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769` | 相同 |
| `scripts/server/train_queue.sh` | `fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa` | 相同 |
| `src/e5/trained_eval.py` | `6695b08b98633b3f4eff76a8ddb3a17d05013ab4bf7e814d67347fde75366b05` | — |
| `scripts/eval_e5_trained.py` | `3919249c7890c65888b44b52cda3ff4ce6d69401148cc670c500030b54f80061` | — |

- 测试（07，srun，带 HF_HOME、HF_HUB_CACHE、HF_HUB_OFFLINE=1、TMPDIR）08:10:32Z – 08:19:53Z：`25 passed, 15 warnings in 551.65s (0:09:11)`，`pytest_rc=0`；删了 `$VI/tmp/pt_0051a2`；status 空；主驱动仍 paused。
- 08:21:00Z 按任务书命令启动 `unmerged.sh`，step 157297.89；前后 squeue 两条 RUNNING；主驱动 step 157297.84 仍在、paused。
  - 驱动日志第一行：`未合并补评开始：hpcgpu07，commit d4c8f67a8c45e4f3ff7f77adf501a184c7c9d763，12 个运行：e3b_c3_n1000_s0 c3_warmup_n1000_s0 e3b_c3_n4000_s0 c3_warmup_n4000_s0 e3b_c3_n1000_s1 c3_warmup_n1000_s1 e3b_c3_n4000_s1 c3_warmup_n4000_s1 e3b_c3_n1000_s2 c3_warmup_n1000_s2 e3b_c3_n4000_s2 c3_warmup_n4000_s2`。

- 第一个未合并补评：`2026-10-18T09:00:00Z unmerged_e3b_c3_n1000_s0 完成`（从 08:21:00Z 起 2,340 s）。输出 `trained/e3b_c3/qwen3.5-2b/n1000/seed0/<任务>__structured__unmerged/`。各任务 `logs/run_001.json` 的 `lora_audit`（不贴 metrics）：

| 任务 | status | lora_state | n_batches | n_called / n_modules | n_not_called | min / max calls | error | run wall_s |
|---|---|---|---|---|---|---|---|---|
| imagery | pass | injected | 7 | 186 / 186 | 0 | 1057 / 1057 | None | 372.2 |
| occlusion | pass | injected | 5 | 186 / 186 | 0 | 1057 / 1057 | None | 294.1 |
| physics | pass | injected | 7 | 186 / 186 | 0 | 1057 / 1057 | None | 387.5 |
| rotation3d | pass | injected | 7 | 186 / 186 | 0 | 1057 / 1057 | None | 424.3 |
| viewpoint | pass | injected | 13 | 186 / 186 | 0 | 1057 / 1057 | None | 809.2 |

## 5. hpcgpu06

还没有加入，等管理员另行通知。
