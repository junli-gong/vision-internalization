# RUN-0045 — HPC A100 正式训练（第一版：已启动，第一段已跑完）

服务器 CC（HPC），在用户的 Mac 上经 `ssh a100 → gpu(hpcmgt)` 操作。2026-10-11 05:12 UTC 起。只贴数字与事实，不做解读。

原始输出：`experiments/E3/0045_hpc_train_probe/raw/`；巡检与统计脚本：`experiments/E3/0045_hpc_train_probe/`。

## 0. 状态

- **两台都在训练：**
  - hpcgpu06 跑 4B，步骤 157296.151；
  - hpcgpu07 跑 0.8B，步骤 157297.79。
  - 每个种子先 C3、后 E3b+C3，种子 0 → 1 → 2。
- **第一段的每步中位**都比 RUN-0044 的实测快：4B C3 是 0.953 倍，0.8B C3 是 0.890 倍。没有触到任务书 §3 的"慢 15%"停止线。
- **推送循环 06:03:50 UTC 起停用**：它的 `git pull --rebase --autostash` 会把驱动正在写的 `.cur` 换成新文件，导致段被误判（§5）。
  - 已报告管理员，等修好后重启。
  - 结果暂时只留在 Lustre 上。
- **铁律：** 前后 squeue 两条都是 RUNNING；没有 scancel、scontrol 改动、sbatch、salloc，没有碰 slurm_script。

## 1. 校验（任务书 §2，`raw/s2_verify.txt`）

- hpcmgt 上 `flock … git pull --ff-only`，rc=0。HEAD 是 `38aa0966`，包含任务书 commit `e30f3ac`。
- 工作树 `$VI/wt/wt0045` 在 `38aa0966d03ff04e0c28d29b53d5b729917a6568`，`git status --short` 为空。
- sha256 与任务书全部相同：

  | 文件 | sha256 |
  |---|---|
  | `configs/e3b_train.json` | `b265cd87…c375` |
  | `prompts/server_hpc/0045-hpc-train/queue.sh` | `72aa7643…15f2` |
  | `scripts/server/train_queue.sh` | `0946d8f2…caa3f` |
  | `scripts/server/sync_push.sh` | `2a8a448d…fed2b` |

- `$VI/logs/HOLD`、`$VI/logs/train.pause` 都不存在。
- 份额（`lfs quota -u chenyuma`，KB）：7,101,203,060，比 RUN-0044 开始时多 41.8 GB。

## 2. 启动（任务书 §4，方案 B，`raw/s4_launch.txt`）

- **squeue（启动前）**：

  ```
  ## squeue 2026-10-11T05:13:08Z [s4 before]
  157296 RUNNING 56-16:38:54 hpcgpu06
  157297 RUNNING 58-14:11:03 hpcgpu07
  ```

- **启动方式：** 经 `launch.sh`（`experiments/E3/0045_hpc_train_probe/launch.sh`，在节点上执行 `setsid nohup srun --jobid=<任务> --overlap -n1 …`）。srun 那一行与任务书相同：
  - hpcgpu06：`env NODE=hpcgpu06 QUEUE_CAP_H=92 EST_C3_H=6.8 EST_MIX_H=21.1 PATH=$VI/env/bin:/usr/bin:/bin bash $VI/wt/wt0045/…/queue.sh`，05:13:09Z，srun 客户端 PID 313417，步骤 157296.151。
  - hpcgpu07：`env NODE=hpcgpu07 QUEUE_CAP_H=33 EST_C3_H=2.6 EST_MIX_H=7.3 PATH=…` 同上，05:13:10Z，srun 客户端 PID 665564，步骤 157297.79。
  - **与任务书不同的一处：** 加了 `PATH=$VI/env/bin:/usr/bin:/bin`。计算节点上没有系统 git，`queue.sh` 只用 `$PY`；不加的话，meta.json 里的 commit 会是 null。管理员已接受。
- **squeue（启动后）**：

  ```
  ## squeue 2026-10-11T05:14:10Z [s4 after]
  157296 RUNNING 56-16:39:56 hpcgpu06
  157297 RUNNING 58-14:12:05 hpcgpu07
  ```

- **驱动日志：**
  - `队列开始：hpcgpu06，qwen3.5-4b，commit 38aa0966…，种子 0 1 2，每段 500 步`；
  - hpcgpu07 同样，模型是 qwen3.5-0.8b。
- **进程**（06:02 的巡检，`raw/patrol_0602Z.txt`）：
  - hpcgpu06：queue.sh 313433，训练 python 318163，卡上 16,852 MiB；
  - hpcgpu07：queue.sh 665580，训练 python 688455，卡上 6,804 MiB。
  - 卡上没有别的进程；节点 MemAvailable 474 / 467 GB；训练进程 RSS 3.3–3.7 GB。

## 3. 每段的每步中位（`p45_segments.py`，从 `logs/timing_run_*.jsonl` 读）

06:02 UTC 的统计：

| 运行 | 段 | 步数 | 每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | 对 RUN-0044 中位的比 |
|---|---|---|---|---|---|---|---|
| c3/qwen3.5-4b/seed0 | 001 | 500 | 5.302 | 0.161 | 1.660 | 3.398 | 0.953 |
| c3/qwen3.5-0.8b/seed0 | 001 | 500 | 1.870 | 0.064 | 0.605 | 1.158 | 0.890 |
| c3/qwen3.5-0.8b/seed0 | 002 | 500 | 1.880 | 0.066 | 0.607 | 1.149 | 0.895 |

- 第一段墙钟（`chunk rc=0 wall_s`）：0.8B 1,051 s，含加载。

## 4. 推送循环（任务书 §5）

- 05:14:20Z 起，`sync_push.sh … 1800 0045 experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07`，PID 3576368。
- 推了两次：`db1bd069 [results] 0045: sync 2026-10-11T05:14:31Z`、`c4a92dbd … 05:44:40Z`。
- 06:03:50Z 我按 PID 停了它，原因见 §5。

## 5. 问题：推送循环的 autostash 让段被误判

- **现象：** 两台的驱动日志里各有一行：
  - hpcgpu07 `05:47:11Z c3_s0 rc=0，没有完成也没有按段停下（第 1 次）`；
  - hpcgpu06 `06:00:40Z` 同样一行。
  - 07 在 c3_s0 的 log 里，第 2 段的输出停在第 920 步，后面没有 `按 --stop-after-steps 停下`；但训练实际跑到了第 1,000 步，第 3 段从那里接着跑。
- **机制**（读代码与 git 记录得出）：
  1. 驱动在每段期间开着 `experiments/E3/0045_hpc_train_<node>/driver/logs/<run>.cur`，一直往里写（`scripts/server/train_queue.sh` 的 `chunk()`）。
  2. `sync_push.sh` 先 `git add -A` 并提交这些目录，所以 `.cur` 成了被跟踪的文件。
  3. 之后的 `git pull --rebase --autostash` 会把训练期间又改动的 `.cur` 暂存起来，再写回成一个新文件（新 inode）。
  4. 训练进程继续写旧的那个；段结束时驱动读到的是截断的内容。
  - 05:44:40Z 那一轮同步正好落在 07 的第 2 段和 06 的第 1 段里。
- **影响：**
  - `run()` 里的 `tries` 在段正常续上时不清零；同一条运行第 2 次误判就记 FAILED、跳到下一条；连续两条就 die。
  - 06 当前这一段会跨过 06:14 左右的那轮同步。
- **处理：**
  - `touch $VI/logs/git.lock.stop` 不够：循环只在一轮做完之后才检查它。
  - 我核对循环在 `sleep 1800`、没有 git 操作在进行后，按 PID 停了 3576368 和 3706012。
  - `sync_0045.status` 写了原因。
  - 没有改任何代码，没有动训练驱动。
- **核对：**
  - 训练脚本自己写的日志没有受影响：`train_log.jsonl` 的行数等于步数（06:04 时 06 是 534 行、最后一步 533；07 是 1,500 行、最后一步 1,499）。
  - 07 的第 3 段在 06:04 被正常识别，接着跑第 4 段。
- **现在：**
  - 两台的 c3_s0 都已记了 1 次误判。只要没有 git 操作碰这些文件，就不会再增加。
  - `$VI/repo` 里这些路径有未提交的改动，我不在那里做 pull 或 stash。
  - 已用 SendMessage 告诉管理员，并给了修法建议：去掉 `--autostash` 或换成单独的副本提交；`.cur` 移出仓库或加进 .gitignore；段续上后清零 `tries`。等管理员给出新的推送方式再重启。

## 6. 之后

- 每条运行跑完时更新本报告：墙钟、每步中位、显存峰值、最终 loss、`CK_FINAL` 下的检查点目录与 `.sha256.txt`。
- 本会话里挂了每小时第 17、52 分钟的巡检，7 天后过期。
