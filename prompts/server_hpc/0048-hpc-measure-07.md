[RUN-TASK 0048] hpcgpu07 训完 0.8B 后的实测：E5、2B 的 E3a+C3、0.8B 的评测、基座 E1 各跑一小段，给迁移后的长任务定预估与上限（D81-2、D91）

## 铁律（先于一切）

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297，跑完也不关。**

- 前后各跑一次 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，两条都必须是 RUNNING。
- 本任务只用 hpcgpu07（任务 157297）。**不要写 `$VI/logs/train.pause`**：RUN-0045 两台的驱动都看这个文件，写了会停住 hpcgpu06 的 4B 训练。本脚本用本节点自己的暂停文件 `$VI/logs/train.pause.hpcgpu07`。

## 为什么做

- B200 退役后，E5 主比较、2B 的 E3a+C3、0.8B 与 4B 的评测、基座 E1 的重跑都改在 A100 上做（D91）。
- 这几类命令在 A100 上都没有实测：
  - 训练的倍数（A100 一张卡约是 B200 的 1/3）不能直接套到评测上，评测的生成受显存带宽限制；
  - D81-2 要求每份批了上限的任务书先按实测逐项预估。
- 所以先每类跑一小段，量墙钟与单位耗时，约 1.5–2 小时。结果只作计时用，不进任何结论。
- 顺带核对：E5 的三个 2B 起点（从 B200 传来的检查点）在 A100 上过得了合并核对。训练开头会自动做；不过就拒绝训练，那就是要报告的异常。

## 何时启动

**等 RUN-0045 的 hpcgpu07 队列跑完**，即 `$VI/logs/train_0045-hpcgpu07.done` 出现、status 是 `done 0045-hpcgpu07`，预计 10-12 中午（UTC）前后。还要满足：

- RUN-0047 已完成，`$VI/b200/` 下逐文件核对通过。

脚本开头会检查这两样，以及 `CUDA_VISIBLE_DEVICES=0`。不满足就写 `blocked` 退出，不会用卡。

## 0. 先读

- `CLAUDE.md`；`~/a100-hpc-guide.md`
- `DECISIONS.md` 的 D81、D89、D91
- `prompts/server_hpc/0048-hpc-measure-07/measure.sh`（本任务的全部命令）
- `scripts/server/eval_lib.sh`：新加了 HPC 模式，变量是 `EV_PAUSE`、`EV_STATUS_GLOB`、`EV_NO_GIT`、`EV_QUOTA_CMD`、`EV_DF_PATH`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验（在 hpcmgt 上）

- **不要在 `$VI/repo` 里 pull**（0045-addendum-1）：`flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'`，origin/main 必须包含本任务书所在的 commit。
- `git -C $VI/repo worktree add --detach $VI/wt/wt0048 origin/main`，确认 `status --short` 为空。
- 核对 sha256（在工作树里）：

  | 文件 | sha256 |
  |---|---|
  | `prompts/server_hpc/0048-hpc-measure-07/measure.sh` | `c3f6395e89871353ebf49f820b0838500ae58fc2bf0b1723111921b6cc0ff782` |
  | `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` |

## 2. 预估与上限

| 项 | 内容 | 预估 |
|---|---|---:|
| E5 训练 20 步 × 3 个起点 | 每个起点：加载 2B 与起点、合并核对（E1 grid16 dev）、20 步 | 约 30–45 分钟 |
| E5 val 一次 | e3b_c3 起点，5 类 × 64 条，结构强制解码（B200 约 3.3 分钟） | 约 10–20 分钟 |
| E3a+C3 2B 训练 20 步 | | 约 5–10 分钟 |
| 0.8B 评测 | T 文本 dev 前 20 条、free 前 10 条；网格 refseg 前 100 条、depth 前 50 条；各含加载与合并核对 | 约 30–45 分钟 |
| 0.8B 基座 E1 | dev 前 20 条 | 约 10–15 分钟 |
| **合计** | | **约 1.5–2 小时** |

- **上限 4 小时**，从拿到 GPU 算起。脚本在每条命令之前判断，超过就写 `blocked` 退出。

## 3. 启动（方案 B）

1. hpcmgt 上 `squeue`，两条 RUNNING。
2. 启动：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     bash /home/mail/chenyuma/visint/wt/wt0048/prompts/server_hpc/0048-hpc-measure-07/measure.sh \
     > /home/mail/chenyuma/visint/logs/eval_0048.driver.log 2>&1 < /dev/null &'
   ```

3. 再跑一次 `squeue`。
4. 心跳：`$VI/logs/eval_0048.status`；最后一行是 `done 0048 (failed steps: N)`。
5. 结果在 `experiments/E3/0048_hpc_measure_hpcgpu07/` 与 `experiments/E5/0048_hpc_measure_hpcgpu07/`。推送用 hpcmgt 上的 `sync_push.sh` v2（0045-addendum-1）：
   - 在 RUN-0045 的循环之外另起一个：`SYNC_WT=$VI/wt/push`，脚本用 `$VI/wt/sync` 里的，TAG `0048`，路径是这两个目录，间隔 600 秒；
   - 两个循环共用 `$VI/wt/push` 与 `git.lock`，不冲突；
   - 跑完、推送确认后按 PID 停掉这个循环。不要用 `git.lock.stop`，它会让 RUN-0045 的循环也退出。

## 4. 报告

推送 `reports/0048-hpc-measure-07.md`，只贴数字与事实：

- 校验结果、工作树 commit、拿到 GPU 的时刻；`steps.log` 原文。
- **E5 训练**：每个起点的
  - 合并核对结果（pass 与各项，原样取自训练开头写的核对 JSON）；
  - 从启动到第 1 步的时间；
  - 20 步的每步中位（`timing_run_*.jsonl` 或训练日志）；
  - 显存峰值。
- **E5 val**：
  - 每类 64 条的墙钟（不含加载）、每批耗时；
  - 起点合并核对的耗时。
- **E3a+C3**：每步中位、输入构建中位、显存峰值。
- **0.8B 评测、基座 E1**：每条命令拆成三段——加载与合并核对、每条样本的耗时（按任务）、合计墙钟。数据从各自的 `metrics.json` 与日志里取，取不到的写"日志里没有"。
- 用量；`$VI/ckpt/hpcgpu07/tmp0048` 已删。

跑完删工作树 `$VI/wt/wt0048`，回到 RUN-0045 的巡检。有事用 SendMessage 发给 `cvpr2027-7c`。
