[RUN-TASK 0045] HPC A100 正式训练：hpcgpu06 跑 4B、hpcgpu07 跑 0.8B，每个种子先 C3 再 E3b+C3（D83、D85、D89）

## 铁律（先于一切）

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297，跑完也不关，让它空转占着卡。**

- 不用 `scancel`（带 step 号也不行），不用 `scontrol requeue/hold/suspend/update`，不用 `sbatch`、`salloc`，不碰 `slurm_script` 进程。
- 每次在集群上动手之前和之后，都跑一次 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，两条都必须是 RUNNING。
- 要停训练，只停我们自己的驱动：写暂停文件，或按 PID 杀驱动与它的子进程。觉得非取消任务不可时，停下来问用户。

## 为什么做

- 这是 H1 修订版的主判据：E3b+C3 对 C3，0.8B、2B、4B 各 3 个种子（E3 预注册 §12，D83-2）。2B 在 B200 上已经训完。
- 同一尺寸放同一台机器（D72-4）：4B 在 hpcgpu06（任务 157296），0.8B 在 hpcgpu07（任务 157297）（D89-5）。
- 每个种子先训 C3、再训 E3b+C3，种子 0 → 1 → 2，这样最早拿到成对的比较。E3b、E3a 单独训练的消融另发任务。
- RUN-0044 的一致性检查 6 组比较全部"一致"，可以开训（D88）。

## 0. 先读

- `CLAUDE.md`；`~/a100-hpc-guide.md`（第 0 节，以及 srun、路径两节）
- `DECISIONS.md` 的 D83、D85、D88、D89
- `prompts/server_hpc/0045-hpc-train/queue.sh`（本任务的全部训练命令）
- `scripts/server/train_queue.sh`（闸门、分段、上限、收尾）与 `scripts/server/sync_push.sh`（hpcmgt 上的推送循环）

本任务只运行，不改任何仓库代码与配置。

## 1. 先收完 RUN-0044

两台节点上的 RUN-0044 吞吐驱动还在做 A/B：先把训练数据拷到 `/dev/shm/visint-chenyuma/train_data`，再用内存盘里的数据跑 20 步 E3b+C3。

1. 等两台的 `driver_0044.sh throughput` 都退出，确认 `/dev/shm/visint-chenyuma` 已删、我们在卡上没有残留进程。哪台先退出，哪台就可以先做第 3 节。
2. 补完 `reports/0044-hpc-bootstrap.md` 的 §7、§8 后推送：
   - 一致性 6 组比较的原文；
   - 吞吐表：用 `p44_tput_table.py` 生成，包括 A/B 那一行和拷贝耗时；
   - `test_grid_checkpoint_split_across_merge_fails_restricted` 失败的完整输出放进 `experiments/E3/0044_hpc_bootstrap/raw/`，管理员据此派开发任务；
   - `probe_scripts/` 下还没入库的脚本一并提交。
3. **训练数据就放在 Lustre 上**，不用内存盘副本：
   - 理由：两台节点的输入构建中位数都是 0.19–0.21 秒，与 B200 本地盘的 0.22 秒相当；
   - 内存盘副本拷一次要一个多小时，节点重启就丢；
   - A/B 的结果只记进报告。如果内存盘那一次每步快了 5% 以上，在报告里写明，由管理员决定后面的任务要不要换。

## 2. 拉取与校验（在 hpcmgt 上）

- `flock $VI/logs/git.lock -c 'git -C $VI/repo pull --ff-only'`，HEAD 必须包含本任务书所在的 commit。
- 建固定工作树：`git -C $VI/repo worktree add --detach $VI/wt/wt0045 HEAD`。确认 `git -C $VI/wt/wt0045 status --short` 为空。
- 核对 sha256（在工作树里）：

  | 文件 | sha256 |
  |---|---|
  | `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
  | `prompts/server_hpc/0045-hpc-train/queue.sh` | `72aa764334f3bea043d9fc65401f460cae62c24e9bb28bb034741a32898a15f2` |
  | `scripts/server/train_queue.sh` | `0946d8f2ad05a20d93f49021601f0cd5bc1b7ca1dd18dd2264e1a952690caa3f` |
  | `scripts/server/sync_push.sh` | `2a8a448d0c755d57f148d4194e73471eaf8217a72381022d4471aa0e820fed2b` |

- `$VI/logs/HOLD` 不存在；`$VI/logs/train.pause` 不存在。

## 3. 预估与上限（D81-2）

预估按 RUN-0044 的实测：20 步里每步的中位数，乘以配置步数。C3 是 `steps` = 4000，E3b+C3 是 `mixed.steps` = 8000。另加 10%，覆盖加载、检查点和每 500 步一次的分段重启。

| 节点 | 模型 | 运行 | 每步中位 s | 单条 h | 单条 h（+10%） | 3 个种子 h（+10%） |
|---|---|---|---:|---:|---:|---:|
| hpcgpu06 | qwen3.5-4b | C3 | 5.562 | 6.18 | 6.8 | 20.4 |
| hpcgpu06 | qwen3.5-4b | E3b+C3 | 8.636 | 19.19 | 21.1 | 63.3 |
| | | **合计** | | | | **约 84 小时（3.5 天）** |
| hpcgpu07 | qwen3.5-0.8b | C3 | 2.102 | 2.34 | 2.6 | 7.7 |
| hpcgpu07 | qwen3.5-0.8b | E3b+C3 | 2.986 | 6.64 | 7.3 | 21.9 |
| | | **合计** | | | | **约 30 小时** |

- **上限：hpcgpu06 `QUEUE_CAP_H=92`，hpcgpu07 `QUEUE_CAP_H=33`。** 驱动只计跑命令的时间。每条运行开始前，它检查"已用 + 这一条的预估"，超过上限就停下，写 `blocked`。
- 预估用环境变量传给驱动：
  - hpcgpu06：`EST_C3_H=6.8 EST_MIX_H=21.1`
  - hpcgpu07：`EST_C3_H=2.6 EST_MIX_H=7.3`
- 显存峰值（RUN-0044）：4B 的 C3 16.5 GB、E3b+C3 19.2 GB；0.8B 两者都是 6.4 GB。40 GB 的卡够用。
- 开跑前按你手上的实测复核一遍。第一段（500 步）跑完时，每步中位比上表慢 15% 以上，就停下报告。

## 4. 启动（方案 B：在节点上发起 srun，hpcmgt 重启不会断训练）

每台节点分别做：

1. 在 hpcmgt 上跑 `squeue`，确认两条都是 RUNNING。
2. 启动：

   ```bash
   # hpcgpu06（4B）
   ssh -n hpcgpu06 'mkdir -p /home/mail/chenyuma/visint/logs && setsid nohup srun --jobid=157296 --overlap -n1 \
     env NODE=hpcgpu06 QUEUE_CAP_H=92 EST_C3_H=6.8 EST_MIX_H=21.1 \
     bash /home/mail/chenyuma/visint/wt/wt0045/prompts/server_hpc/0045-hpc-train/queue.sh \
     > /home/mail/chenyuma/visint/logs/train_0045-hpcgpu06.driver.log 2>&1 < /dev/null &'

   # hpcgpu07（0.8B）
   ssh -n hpcgpu07 'mkdir -p /home/mail/chenyuma/visint/logs && setsid nohup srun --jobid=157297 --overlap -n1 \
     env NODE=hpcgpu07 QUEUE_CAP_H=33 EST_C3_H=2.6 EST_MIX_H=7.3 \
     bash /home/mail/chenyuma/visint/wt/wt0045/prompts/server_hpc/0045-hpc-train/queue.sh \
     > /home/mail/chenyuma/visint/logs/train_0045-hpcgpu07.driver.log 2>&1 < /dev/null &'
   ```

3. 再跑一次 `squeue`，两条仍是 RUNNING。
4. 驱动日志里应出现"队列开始：…"，commit 是工作树的 HEAD。`$VI/logs/train_0045-<节点>.status` 在更新，`c3_s0` 开始训练。
   - 驱动开头会检查 `CUDA_VISIBLE_DEVICES=0`。不在 srun 的资源范围里，它会直接退出，不会训练。

## 5. 推送循环（在 hpcmgt 上）

计算节点不碰 git（`TQ_NO_GIT=1`）。结果由 hpcmgt 上的循环每 30 分钟提交并推送一次：

```bash
setsid nohup bash /home/mail/chenyuma/visint/wt/wt0045/scripts/server/sync_push.sh \
  /home/mail/chenyuma/visint/repo /home/mail/chenyuma/visint/logs/git.lock 1800 0045 \
  experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07 \
  > /home/mail/chenyuma/visint/logs/sync_0045.log 2>&1 < /dev/null &
```

- 心跳：`$VI/logs/sync_0045.status`。
- hpcmgt 重启后，这个循环会断，训练不受影响。巡检时发现它不在，就照上面的命令重启。
- 要停循环：`touch $VI/logs/git.lock.stop`，它跑完当前一轮就退出。

## 6. 巡检

每 30–60 分钟巡检一次。你的会话里有定时工具的话，用它挂一个。

- **squeue**：两条都是 RUNNING。
- **两台的驱动**：status 文件在更新，驱动进程还在。
- **推送**：`sync_0045.status` 是 `ok`，origin 上能看到 `[results] 0045: sync …` 提交。
- **资源**：
  - 训练进程的 RSS 不超过 150 GB（D89-8）。这一条由你在巡检时查，驱动管的是节点 MemAvailable 低于 100 GB 就停；
  - Lustre 份额不超过 80 GB（驱动自己查）。
- **卡上有别人的进程**：驱动会自动暂停，等它走。持续 2 小时以上，报告用户与管理员。
- 驱动因为停止线退出（`blocked` 或 `die`）：
  - 不改代码、不放宽上限，把 status 原文与驱动日志的最后 50 行写进报告，推送，告诉管理员；
  - 段边界上的检查点还在，修好后用同一条命令重启，驱动会跳过已完成的运行，从最后一段接着训。

有事要找管理员时，用 SendMessage 发给 `cvpr2027-7c`，同时写进报告推送。用户在睡觉，不要等用户确认。铁律相关的事例外：先停下来问用户。

## 7. 报告

`reports/0045-hpc-train.md`，只贴数字与事实：

- **启动后先推一版**：
  - 校验结果、工作树 commit、两台的启动时刻；
  - 驱动的 PID 与 srun 的 step 号；
  - 前后两次 squeue 的原文；
  - 第一段的每步中位。
- **每跑完一条运行更新一次**：运行名、墙钟、每步中位、显存峰值、最终 loss，以及 `CK_FINAL` 下检查点目录和 `.sha256.txt` 的位置。
- **两台都跑完**：
  - 两台的 status 原文；
  - 累计 GPU 小时与上限；
  - Lustre 份额；
  - 最后一次 squeue。

一台节点的队列跑完后，不要在那张卡上接着启动别的东西。任务照铁律空转，0.8B 的消融与评测另发任务。

两台都跑完、报告推送后，按 `~/a100-hpc-guide.md` 收尾：停推送循环，删工作树 `$VI/wt/wt0045`。检查点留在 `$VI/ckpt/<节点>/0045`。
