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
  - restricted 0.875（5 / 40），gap 中位 0.2420、最大 0.3492；merge 通过。
- **eval_c3_warmup_n1000_s1（hpcgpu07）：**
  - 训练 12:16:55Z 完成，检查点 `…/0051/train/c3_warmup/qwen3.5-2b/n1000/seed1/ckpt/final`，131,091 KB。
  - 评测两次 rc=1：12:27:32Z、12:32:02Z。`failed.txt` 原文 `2026-10-15T12:32:02Z eval_c3_warmup_n1000_s1 rc=1`。驱动随后跳到 `base_warmup_n1000_s1`。
  - restricted 通过（0.95，2 / 40；gap 中位 0.2405、最大 0.4376），**只有 merge 没过**：`merge/thinking_off：教师强制的 argmax_agreement 0.97214 < 0.98`。
- 截至 14:23Z：`failed.txt` 共 8 行；驱动在跑 `base_warmup_n1000_s1`。
- 管理员 10-14 的答复：判据交用户与负责人定（N34），队列不调整，照原样继续、继续记录，被拒的检查点留着以后统一补评。

## 5. hpcgpu06

还没有加入，等管理员另行通知。
