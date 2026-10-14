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
- 起点合并核对的 pass 与 exact_rows、test 的有效率与解析失败率：等结果目录推上来后，用脚本补进这一节。
- 15:05:34Z 驱动认领了 `small_s0`，开始 `e3b_c3_n1000_s0`。

## 4. hpcgpu06

还没有加入，等管理员另行通知。
