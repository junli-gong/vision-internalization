[RUN-TASK 0055] hpcgpu06：9B 在 A100-40GB 上的实测（N38 答复前的默认动作）。第一部分现在做：下载并核对 9B 权重，只用 CPU 与网络。第二部分等 06 跑完 E5 的 big_s0 再做：训练 20 步 × 3 臂，评测各一小段，约 2.5–3.5 小时，上限 5 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 每一部分前后各跑一次 `squeue`。
- 不碰 06 上正在跑的 RUN-0051（big_s0），不碰 07 上的任何东西。
- 第二部分只用 06 自己的暂停文件 `train.pause.hpcgpu06`：由 eval_lib 的 `claim_gpu` 写，退出时自动删。
- 不对 Lustre 上的大目录 `du`。看下载进度用 `ls -l` 或 `stat`。

## 为什么

- 06 约 10-20 07:00 UTC 跑完 E5 的 big_s0，比原排期早约 5 天空出来。9B 原定 10-25 之后（D93）。N38 问负责人 9B 能否提前；答复前先做实测。
- 9B 在 40 GB 的 A100 上能不能训，还没有实测：
  - B200 上 9B 只加载权重 17.5 GB；
  - LoRA r32、批 8、序列 1060、梯度检查点，显存峰值 47.8 GB（`docs/env_report.md` §7）；
  - E3 的配置是微批 4。
- 结果只作计时与显存用，不进任何结论。

## 文件（commit `e496ea0b9bd368b29240cdb153b1312cfbf984ac`）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0055-hpc-06-9b-measure/fetch_9b.sh` | `8ef9a46511010e4989f1b6c61200c5606f963a9c802bad5834d23df472ed303b` |
| `prompts/server_hpc/0055-hpc-06-9b-measure/measure_9b.sh` | `d680172ee26d606b472fe88b810da430c753652b5e9c7bf7997cde58868a0fc1` |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |
| `configs/e3b_train.json` | `b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375` |
| `src/data/train_loaders.py` | `76535736def11a13a19a94a930ead215684ee4a697db3c05f6fd8c35d54d10cb`（DEV-0047） |
| `experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py` | `0fcd224eb3b0025e7d34045a5243af8acd8eafc14daef98aa7007f3452504550` |
| `experiments/P0/0009_ifeval_rescore/weights_manifest.json` | `8535b8cb89e6def1f8a79feff669b315ca0578c75d4b7ce310d4ff8aa86efa46` |

开发机上用 tiny 彩排过两遍：一遍正常走，一遍强制走"微批 2"的分支。各步都 rc=0；网格与 T 文本评测能直接读 20 步停下时写的 `latest/model`。

## 第一部分：现在做——下载并核对 9B 权重（约 1.5–2 小时）

1. hpcmgt 上 `squeue`，两条 RUNNING。
2. **建工作树 wt0055**（稀疏，同 RUN-0054）：

   ```bash
   VI=/home/mail/chenyuma/visint
   flock $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"
   git -C $VI/repo worktree add --no-checkout --detach $VI/wt/wt0055 e496ea0b9bd368b29240cdb153b1312cfbf984ac
   git -C $VI/wt/wt0055 sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'
   git -C $VI/wt/wt0055 read-tree -mu HEAD
   ls $VI/wt/wt0055/experiments/            # 应只有 P0
   git -C $VI/wt/wt0055 status --short      # 应为空
   ```

   核对上表 7 个文件的 sha256。
3. **启动下载**（经 srun 在 06 上；只用 CPU 与网络，与 big_s0 的训练并行）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     bash /home/mail/chenyuma/visint/wt/wt0055/prompts/server_hpc/0055-hpc-06-9b-measure/fetch_9b.sh \
     > /home/mail/chenyuma/visint/logs/fetch_0055.driver.log 2>&1 < /dev/null &'
   ```

   - 再跑一次 `squeue`。
   - 心跳 `$VI/logs/fetch_0055.status`。
   - 脚本已设 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`，并取消了 `HF_HUB_OFFLINE`（RUN-0044 的教训）。
4. **看速度**：每 10 分钟用 `ls -l $VI/hf/hub/models--Qwen--Qwen3.5-9B/blobs/` 看 `.incomplete` 文件的字节数。
   - RUN-0044 的经验：一条长连接可能降到约 100 KB/s。连续 10 分钟低于 500 KB/s，就按 PID 停掉 fetch_9b.sh 起的 python 进程，然后照第 3 步重启。下载会从 `.incomplete` 接着来。
   - 只停我们自己的 python。**不用 scancel，不碰 slurm_script。**
   - 同一时刻只跑一份下载。
5. **完成**：心跳是 `done fetch_0055 (attempt k)`，日志里有 `qwen3.5-9b @ c202236235762e1c871ad0ccb60c8ee5ba337b9a: 16/16 files verified`。共约 19.3 GB。
   - 3 次都失败（`failed fetch_0055`）就停下报告。
   - 别人的残留不要删；只删我们停掉的那次留下、确认是我们的 `.incomplete`。

## 第二部分：06 跑完 E5 之后——实测（约 2.5–3.5 小时，上限 5 小时）

**以下都满足才开始**：
1. `$VI/logs/train_0051-hpcgpu06.done` 存在，心跳是 `done 0051-hpcgpu06`；
2. `e5_claims_0051/big_s0/done` 存在；
3. 06 上没有 0051 的 queue.sh 进程，GPU 上没有进程；
4. 第一部分已完成（16/16）。

预计 10-20 07:00–08:00 UTC。big_s0 里另两个 64K 起点的跳过核对照常单独报，不挡本部分。

1. hpcmgt 上 `squeue`；确认上面四条。
2. **启动**（方案 B）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin EV_CAP_H=5 \
     bash /home/mail/chenyuma/visint/wt/wt0055/prompts/server_hpc/0055-hpc-06-9b-measure/measure_9b.sh \
     > /home/mail/chenyuma/visint/logs/eval_0055.driver.log 2>&1 < /dev/null &'
   ```

   - 再跑一次 `squeue`。
   - 心跳 `$VI/logs/eval_0055.status`，最后一行是 `done 0055 (failed steps: N)`。
   - 脚本开头会再核一次 9B 的本地权重（只读）。没过就写 `blocked` 退出，不用卡。
3. **内容**（全部命令都在 measure_9b.sh 里）：

   | 项 | 内容 | 预估 |
   |---|---|---:|
   | 训练 20 步 × 3 臂 | E3b+C3、E3b、C3，种子 0，正式配置（微批 4 × 累积 4）；每臂含加载 | 约 45–70 分钟 |
   | 微批 2 的补测 | 只在某臂显存不够时，同一臂改用微批 2 × 累积 8 再跑 20 步 | 每臂约 15–25 分钟 |
   | 网格评测 | E3b+C3 的 20 步检查点（没有就用 E3b 的）：refseg 前 100 条、depth 前 50 条，结构强制；各含加载与合并核对 | 约 45–75 分钟 |
   | T 文本 | 9B 基座 dev 前 20 条、free 前 10 条；E3b+C3 的检查点 dev 前 20 条（没有就用 C3 的） | 约 40–60 分钟 |

   - 20 步的检查点没训好，合并核对被拒是可能的。照实记，不绕过；这一步 rc ≠ 0 也算读数。
   - 任何一步显存不够都照实记，不改批配置重试（只有训练有微批 2 的补测）。
4. **推送**：另起一个 v3 循环，TAG 0055，路径两个：

   ```bash
   setsid nohup bash /home/mail/chenyuma/visint/wt/sync3/scripts/server/sync_push.sh \
     /home/mail/chenyuma/visint/repo /home/mail/chenyuma/visint/logs/git.lock 1800 0055 \
     experiments/E3/0055_hpc_measure_9b_hpcgpu06 experiments/E3/0055_hpc_measure_9b_e1_hpcgpu06 \
     > /home/mail/chenyuma/visint/logs/sync_0055.log 2>&1 < /dev/null &
   ```

   - 跑完、最后一轮推送确认后，在它 sleep 时按 PID 停掉这个循环。
   - 不要用 `git.lock.stop`。
5. 跑完之后，06 空着等管理员的下一份任务书（N38 的答复）。`$VI/ckpt/hpcgpu06/tmp0055` 由脚本删掉，确认已删。

## 报告

推送 `reports/0055-hpc-06-9b-measure.md`，只贴耗时、显存与事实，**不贴任何指标读数**（20 步的检查点，读数没有意义）：
- **第一部分**：开始与完成时刻、平均速度、有没有重启；核对那一行原文；工作树 commit 与 sha256 核对。
- **第二部分**：
  - `steps.log` 原文；
  - **训练**每臂：从启动到第 1 步的时间、20 步的每步中位、输入构建中位、显存峰值（allocated 与 reserved）。微批 4 显存不够的，贴报错的最后几行；跑了微批 2 的，同样报这几项。
  - **网格评测**每条：合并核对的耗时与结果（pass 与 failures 原文）；解码的每批耗时与合计墙钟；显存峰值；批大小。
  - **T 文本**每条：加载与合并核对、每条样本的耗时（按任务）、合计墙钟；显存峰值；批大小。日志里取不到的写"日志里没有"。
  - 用量；`tmp0055` 已删。

有事用 SendMessage 发给管理员会话。
