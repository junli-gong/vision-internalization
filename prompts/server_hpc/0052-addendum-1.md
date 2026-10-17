[RUN-TASK 0052-addendum-1] hpcgpu06：E3b 队列在 e3b_s2 前因用量上限停下、run.sh 做完种子 0、1 的 eval b 之后，换到含 DEV-0047 的 commit，另起一步训 e3b_s2 并只评种子 2（D98）

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 本任务不杀、不停任何进程。07 上的 RUN-0051 不碰。

## 什么时候做

**等这三件事都发生了再开始**，按顺序：
1. E3b 队列（`train_0052-hpcgpu06.status`）在 e3b_s1 完成之后，因用量上限停在 e3b_s2 之前：`blocked cap: used … + estimate 13.6 h for e3b_s2 > cap 46 h`。驱动日志与 `run_0052.log` 有 `queue rc=1`。
2. `run.sh` 接着跑 `eval.sh b`：种子 0、1 的 refseg 与 depth 结构强制网格；种子 2 的检查点不在，`steps.log` 记"检查点不在"。
3. `run_0052.log` 出现 `eval b rc=…` 与 `done`，并且 `pgrep -af 0052-hpc-06-4b` 没有任何 run.sh、queue.sh、eval.sh 进程。

**在 `run.sh` 结束之前，绝不切换 wt0052**：bash 边读边执行脚本，换文件会打乱正在跑的脚本。

- 如果 e3b_s1 之后队列**没有**停下（e3b_s2 照常开始），说明用量没到上限：不用做本任务，告诉管理员。
- 任何一步与预期不同，就停下来报告。

预计：e3b_s1 约 10-17 18:00 UTC 训完；种子 0、1 的 eval b 约 6.6 小时，到 10-18 00:40 前后。

## 为什么（D98）

- 10-17 Lustre 元数据一度很慢。训练加载器逐个 `stat` 约 17 万个文件，e3b_s1 第 7 段在 GPU 空闲时启动了约 3 小时，E3b 队列的用量因此超过原上限 46 小时。
- DEV-0047（706d3a0a）把加载器改成按目录列一次清单：
  - 训练池逐条相同，续跑身份不变；
  - 约 17 万次 stat 降到约 280 次目录列举。
- e3b_s2 全程用新 commit，不在种子中途换代码。
- 用量上限放宽到 52 小时。超出原上限约 6 小时，全是 Lustre 慢时的空等。

## 新 commit 与文件

切到 `b62986d5f0fa476950a0717081d887c1142ce725`（含 DEV-0047 与本补充的脚本）。

| 文件 | sha256 | 说明 |
|---|---|---|
| `prompts/server_hpc/0052-hpc-06-4b/run_s2.sh` | `c870fb2dd6ab2c32b87d91efc8eb68a4888bf9713bed103cc8965a248c829b54` | 新：queue.sh，然后只评种子 2 的 eval b |
| `prompts/server_hpc/0052-hpc-06-4b/eval.sh` | `494ef3ad067434674d7cc68e8d96ec5abfd57a3914a2869d8a4bf737805fa91d` | 只加 `EVAL_SEEDS`（默认仍是 0 1 2） |
| `prompts/server_hpc/0052-hpc-06-4b/queue.sh` | `5daa6013b7b97022e8fcb144a2d68a02a6deb1fe2936f7aaf91a0f62a7e3e8b4` | 不变 |
| `prompts/server_hpc/0052-hpc-06-4b/run.sh` | `a4ae09a0117790047babd15ba5fd21e6a6368b58c4f678d4452c12a661919ffc` | 不变（本任务不用） |
| `scripts/server/train_queue.sh` | `fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa` | 不变 |
| `scripts/server/eval_lib.sh` | `2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59` | 不变 |
| `src/data/train_loaders.py` | `76535736def11a13a19a94a930ead215684ee4a697db3c05f6fd8c35d54d10cb` | DEV-0047 |

- `queue.sh` 再跑一次时，会跳过驱动 `done.txt` 里已完成的 e3b_s0、e3b_s1，只训 e3b_s2。
- 驱动的 `gpu_seconds` 接着累计，所以上限要写 52，不是 13。

## 做法

1. hpcmgt 上 `squeue`，两条 RUNNING。确认上面"什么时候做"的三件事，记下：
   - `run_0052.log` 的最后几行；
   - 驱动的 `gpu_seconds`；
   - `steps.log` 里种子 0、1 的四个 grid 步的 rc 与 wall_s。
2. **切换 wt0052**：

   ```bash
   flock /home/mail/chenyuma/visint/logs/git.lock -c 'git -C /home/mail/chenyuma/visint/wt/wt0052 fetch -q origin main'
   git -C /home/mail/chenyuma/visint/wt/wt0052 checkout -q --detach b62986d5f0fa476950a0717081d887c1142ce725
   git -C /home/mail/chenyuma/visint/wt/wt0052 rev-parse HEAD      # 应为 b62986d5…
   git -C /home/mail/chenyuma/visint/wt/wt0052 status --short       # 应为空
   ls /home/mail/chenyuma/visint/wt/wt0052/experiments/             # 应只有 P0
   ```

   核对上表 7 个文件的 sha256。
3. **在 06 上跑 DEV-0047 的测试**（CPU；环境变量与驱动相同；约 1 分钟）：

   ```bash
   ssh -n hpcgpu06 'srun --jobid=157296 --overlap -n1 env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     HF_HOME=/home/mail/chenyuma/visint/hf HF_HUB_CACHE=/home/mail/chenyuma/visint/hf/hub HF_HUB_OFFLINE=1 TMPDIR=/home/mail/chenyuma/visint/tmp \
     bash -c "cd /home/mail/chenyuma/visint/wt/wt0052 && CUDA_VISIBLE_DEVICES= python -m pytest tests/test_train_loaders_listing.py tests/test_e3_data.py -q -p no:cacheprovider --basetemp=/home/mail/chenyuma/visint/tmp/pt_0052a1"' < /dev/null
   ```

   - 全部通过才往下。区分大小写那一条在 Linux 上应当照跑并通过。
   - 跑完删掉 `$VI/tmp/pt_0052a1`。
   - 测试不过就停下，把输出发给管理员。
4. **启动**（方案 B）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin QUEUE_CAP_H=52 EST_E3B_H=13.6 EV_CAP_B_H=6 \
     bash /home/mail/chenyuma/visint/wt/wt0052/prompts/server_hpc/0052-hpc-06-4b/run_s2.sh \
     > /home/mail/chenyuma/visint/logs/run_0052_s2.driver.log 2>&1 < /dev/null &'
   ```

   - 再跑一次 `squeue`。
   - 心跳仍是 `train_0052-hpcgpu06.status`（队列）与 `eval_0052-b.status`（评测）。
5. **核对**：
   - 驱动日志先出现"e3b_s0 已完成，跳过""e3b_s1 已完成，跳过"，再开始 e3b_s2 第 1 段。
   - 第 1 段从开始到 `run_001` 的 started_utc，记下启动用时。新加载器下应是几分钟。
   - e3b_s2 第 1 段跑完时报一次：段长、每步中位。
6. **全部跑完**（e3b_s2 训练约 13–14 小时，加 eval b 约 3.5 小时）：贴
   - `run_0052.log` 的 s2 几行；
   - `steps.log` 种子 2 的两个 grid 步；
   - 驱动 `gpu_seconds`。

## 报告

追加进 `reports/0052-hpc-06-4b.md`，新开一节"addendum-1"，只贴事实：
- 队列停下与 run.sh 结束的时刻；
- 切换前后的 commit 与 sha256 核对；
- 测试结果原文；
- e3b_s2 各段的启动用时与段长；
- 两个 grid 步的 rc 与 wall_s。
