[RUN-TASK 0051-addendum-2] E5 剩余部分的排程（D100）：big_s0 预认领给 hpcgpu06；07 跑完 mid_s2 后做 D99-2 的 12 个未合并补评；06 在 RUN-0052 补充 1 做完后加入 E5，只跑 big_s0

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 每一部分前后各跑一次 `squeue`。
- 不杀、不停任何驱动进程，主驱动只靠暂停文件在两步之间等着。
- 不要用 `git.lock.stop`，它会停掉所有推送循环。

本任务切到的 commit：`d4c8f67a8c45e4f3ff7f77adf501a184c7c9d763`（含 DEV-0047、DEV-0048 与 `unmerged.sh`）。

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0051-hpc-e5-main/unmerged.sh` | `5905d0310ae75019ca3661b33c559a016a07be9d33c8d1de3add8dbe9ba223ff` |
| `prompts/server_hpc/0051-hpc-e5-main/queue.sh`（不变） | `8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769` |
| `scripts/server/train_queue.sh`（不变） | `fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa` |
| `src/e5/trained_eval.py` | `6695b08b98633b3f4eff76a8ddb3a17d05013ab4bf7e814d67347fde75366b05` |
| `scripts/eval_e5_trained.py` | `3919249c7890c65888b44b52cda3ff4ce6d69401148cc670c500030b54f80061` |

不加 `--unmerged` 时，正式评测的路径与输出和 2db1055（07 现在用的 f703d7db 的评测代码）逐字节相同，DEV-0048 用快照测试钉住。

## 第一部分：现在就做——big_s0 预认领给 06

**必须在 07 跑完 mid_s2 之前做完**（mid_s2 预计 10-18 12:00 UTC 前后跑完）。

```bash
C=/home/mail/chenyuma/visint/logs/e5_claims_0051
ls $C                                  # 应有 n0_all small_s0–2 mid_s0–2，没有 big_s0
mkdir $C/big_s0 && echo "hpcgpu06 $(date -u +%Y-%m-%dT%H:%M:%SZ) pre-claimed by manager (D100, 0051-addendum-2)" > $C/big_s0/node
cat $C/big_s0/node
```

- `mkdir` 失败，说明 big_s0 已存在：不要动它，停下报告。
- 依据：
  - queue.sh 的 `group()` 遇到已有目录、`node` 不是本节点的组，记"已被 … 认领，跳过"；所以 07 跑完 mid_s2 会跳过 big_s0，直接认领 ad_s0。
  - 06 加入后，遇到 `node` 是本节点、没有 `done` 的组，走"本节点认领过、没跑完的，接着跑"。

## 第二部分：07 跑完 mid_s2 后——12 个未合并补评（D99-2）

1. **在 07 跑 mid_s2 的最后一条（base_warmup_n16000_s2）期间**，写主驱动的暂停文件：

   ```bash
   echo "RUN-0051 addendum-2: unmerged re-evals (D99-2, D100)" > /home/mail/chenyuma/visint/logs/train.pause.hpcgpu07
   ```

   - mid_s2 跑完后，驱动会认领 ad_s0，然后停在它第一条运行之前，心跳是 `paused`。
   - 记下停住的时刻，以及 `e5_claims_0051/` 的列表（应有 ad_s0，node 是 hpcgpu07；big_s0 的 node 是 hpcgpu06）。
2. **切换 wt0051**（主驱动已暂停）：

   ```bash
   flock /home/mail/chenyuma/visint/logs/git.lock -c 'git -C /home/mail/chenyuma/visint/wt/wt0051 fetch -q origin main'
   git -C /home/mail/chenyuma/visint/wt/wt0051 checkout -q --detach d4c8f67a8c45e4f3ff7f77adf501a184c7c9d763
   git -C /home/mail/chenyuma/visint/wt/wt0051 rev-parse HEAD; git -C /home/mail/chenyuma/visint/wt/wt0051 status --short
   ```

   核对上表 5 个文件的 sha256。
3. **在 07 上跑 DEV-0048 的测试**（CPU，环境变量照抄，约 5–10 分钟）：

   ```bash
   ssh -n hpcgpu07 'srun --jobid=157297 --overlap -n1 env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     HF_HOME=/home/mail/chenyuma/visint/hf HF_HUB_CACHE=/home/mail/chenyuma/visint/hf/hub HF_HUB_OFFLINE=1 TMPDIR=/home/mail/chenyuma/visint/tmp \
     bash -c "cd /home/mail/chenyuma/visint/wt/wt0051 && CUDA_VISIBLE_DEVICES= python -m pytest tests/test_e5_unmerged.py tests/test_train_loaders_listing.py -q -p no:cacheprovider --basetemp=/home/mail/chenyuma/visint/tmp/pt_0051a2"' < /dev/null
   ```

   - 全部通过才往下，跑完删掉 basetemp。
   - 不过就停下报告，暂停文件留着。
4. **启动补评**（方案 B）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env NODE=hpcgpu07 EXPECT_COMMIT=d4c8f67a8c45e4f3ff7f77adf501a184c7c9d763 QUEUE_CAP_H=11 PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     bash /home/mail/chenyuma/visint/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/unmerged.sh \
     > /home/mail/chenyuma/visint/logs/train_0051u-hpcgpu07.driver.log 2>&1 < /dev/null &'
   ```

   - 开始前它核对：commit；主驱动暂停中；12 个运行的正式评测都已完成、检查点都在。任何一条不满足，它就停，暂停文件留着，这时报告。
   - 名单固定为 E3b+C3 与 C3+预热 × 1K、4K × 种子 0、1、2。
   - 每个约 0.65–0.7 小时，合计约 8–9 小时。
   - 跑完写 `experiments/E5/0051_e5_main_hpcgpu07/unmerged_d99/summary.tsv`（只有叶目录数与审计状态，没有分数），然后删掉主驱动的暂停文件，主驱动接着跑 ad_s0。
5. **报告**：
   - 第一个补评跑完时：墙钟，以及运行记录 `lora_audit` 的 status 与层数。**不要贴任何 metrics 读数。**
   - 全部跑完：`summary.tsv` 原文；主驱动恢复的时刻；恢复后第一条运行的名字。

## 第三部分：RUN-0052 补充 1 做完后——06 加入 E5，只跑 big_s0

**前提**：
- `run_0052.log` 有 `s2 done`，没有任何 0052 的 run_s2.sh、queue.sh、eval.sh 进程；
- 第二部分第 2 步的切换已做完（wt0051 在 d4c8f67a）。

如果到 06 空出来时第二部分还没切换，先报告，等管理员。

1. `squeue`；`cat $VI/logs/e5_claims_0051/big_s0/node`，应是 hpcgpu06，且没有 `done`。
2. **启动**（方案 B，同 RUN-0051 任务书 §3，只换节点与上限）：

   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 \
     env NODE=hpcgpu06 PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin QUEUE_CAP_H=46 \
     bash /home/mail/chenyuma/visint/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/queue.sh \
     > /home/mail/chenyuma/visint/logs/train_0051-hpcgpu06.driver.log 2>&1 < /dev/null &'
   ```

   - 预期驱动日志：n0_all、small_s*、mid_s* 都"已被 hpcgpu07 认领，跳过"或"已跑完，跳过"；big_s0"是本节点认领过、没跑完的，接着跑"；ad_s* 已被 hpcgpu07 认领，跳过；最后 finish。
   - big_s0 = 3 个 64K 运行，每个约 12.6 小时训练加 0.5 小时评测。
3. **推送**：另起一个 v3 循环，不碰现有的 0051 循环：

   ```bash
   setsid nohup bash /home/mail/chenyuma/visint/wt/sync3/scripts/server/sync_push.sh \
     /home/mail/chenyuma/visint/repo /home/mail/chenyuma/visint/logs/git.lock 1800 0051h06 experiments/E5/0051_e5_main_hpcgpu06 \
     > /home/mail/chenyuma/visint/logs/sync_0051h06.log 2>&1 < /dev/null &
   ```

4. **报告**：
   - 驱动开头那几行"跳过 / 接着跑"的原文；
   - 第一个 64K 运行第 1 段的墙钟与每步中位，应接近 5.1 s/步。

## 报告

追加进 `reports/0051-hpc-e5-main.md`，新开一节"addendum-2"，三部分分别记：
- 时刻、commit、sha256 核对、测试原文；
- `summary.tsv`；
- 06 驱动的跳过与开始的原文。
- 不贴任何 test 分数。
