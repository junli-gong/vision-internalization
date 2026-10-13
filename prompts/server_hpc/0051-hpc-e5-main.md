[RUN-TASK 0051] E5 主比较 v3 加 E3b 起点，在两台 A100 上按组分摊（E5 预注册 §8a、§8b；D83、D91、D93）

## 铁律（先于一切）

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297，跑完也不关。**

- 每次在集群上动手之前和之后，都跑一次 `squeue -h -j 157296,157297 -o "%i %T %M %N"`，两条都必须是 RUNNING。
- 每台节点只用自己的暂停文件 `$VI/logs/train.pause.<节点>`。**不要写 `$VI/logs/train.pause`**：RUN-0045 的驱动在看它。

## 为什么做

- E5 是"心智"这一问的主实验。H5 的主判据是 E3b+C3 对 C3+预热（§8a）。
- E3b 起点是 D93 加的，探索性，不进 H5 的判定（§8b）。
- B200 退役，E5 改到两台 A100 上（D91-1）：
  - 同一个数据量档、同一个种子的所有起点放在同一台节点上，比较都在机内完成；
  - B200 上已做的 4 个 N = 0 读数在这里重做，那份不进结论。

## 何时启动

- **hpcgpu07**：等 RUN-0050 跑完，即 `$VI/logs/run_0050.log` 最后一行是 `done`，预计 10-14 晚上（UTC）。之后马上启动。
- **hpcgpu06**：4B 的工作做完以后，管理员另发一条消息，叫你用同一个脚本、`NODE=hpcgpu06` 启动。不要自己提前启动。

## 0. 先读

- `CLAUDE.md`；`~/a100-hpc-guide.md`
- `DECISIONS.md` 的 D83、D91、D93；`PREREGISTRATION.md` 的 E5 §8、§8a、§8b
- `prompts/server_hpc/0051-hpc-e5-main/queue.sh`（本任务的全部命令）
- `scripts/server/train_queue.sh`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验（在 hpcmgt 上）

- 不在 `$VI/repo` 里 pull：`flock $VI/logs/git.lock -c 'git -C $VI/repo fetch -q origin main'`，origin/main 必须包含本任务书所在的 commit。
- `git -C $VI/repo worktree add --detach $VI/wt/wt0051 origin/main`，确认 `status --short` 为空。两台节点共用这一个工作树。06 加入时不要重建；如果已经删了，就重建在同一个 commit 上。
- 核对 sha256（在工作树里）：

  | 文件 | sha256 |
  |---|---|
  | `prompts/server_hpc/0051-hpc-e5-main/queue.sh` | `8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769`（D95 改过：不查份额） |
  | `scripts/server/train_queue.sh` | `fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa`（D95：TQ_DF_PATH） |
  | `configs/e5_train.json` | `e7f28997935f7281a6d0f4fe1d29132cc2d14e4b500000f00882dd7b2b2b94c3`（e5-train-v3） |
  | `configs/e5_prompts.json` | `c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4` |

- 起点检查点都在，逐文件核对 `$VI/b200/SHA256SUMS.b200` 里对应的行：
  - `$VI/b200/ckpt/0040/{e3b_c3,c3_warmup}/qwen3.5-2b/seed{0,1,2}`
  - `$VI/b200/ckpt/0036/{warmup,e3b}/qwen3.5-2b/seed{0,1,2}`
- `$VI/logs/e5_claims_0051/` 不存在（第一次启动时由脚本建）。
- E5 生成器跨平台：d 类的数据只在 x86_64 Linux 上生成（D24）。RUN-0044 的 pip freeze 显示 HPC 与 B200 的 80 个共有包版本全部相同，`test_synth_repro.py` 的钉住摘要也都通过了。这一条不用另查。

## 2. 组与顺序（queue.sh；两台按组认领）

两台节点跑同一个脚本，在 `$VI/logs/e5_claims_0051/<组>` 上用 mkdir 原子认领，按下表顺序。哪台先空，就认领下一个还没被认领的组。

| 组 | 内容 | 运行数 | 预估（加 10%） |
|---|---|---:|---:|
| n0_all | N = 0：三个 v3 起点 × 种子 0–2，加 E3b 种子 0、2 | 11 个读数 | 约 5.5 小时 |
| small_s0 | 1K、4K：三个 v3 起点加 E3b，种子 0 | 8 | 约 20 小时 |
| small_s1 | 1K、4K：三个 v3 起点，种子 1 | 6 | 约 15 小时 |
| small_s2 | 1K、4K：三个 v3 起点加 E3b，种子 2 | 8 | 约 20 小时 |
| mid_s0、mid_s1、mid_s2 | 16K：三个 v3 起点 | 各 3 | 各约 11 小时 |
| big_s0 | 64K：三个 v3 起点，种子 0 | 3 | 约 39 小时 |
| ad_s0、ad_s1、ad_s2 | a–d 迁移，4K，test 只评 e | 各 3 | 各约 7 小时 |
| **合计** | | 51 个运行加 11 个 N = 0 | **约 155 小时** |

- **预估的依据**（RUN-0048 实测，A100）：
  - 每步 5.05 s，val 一次约 2.7 分钟，起点合并核对约 7 分钟；
  - test 一次 2,300 条，约 0.4 小时。
  - 每条运行的预估写在 queue.sh 的 `est_train` 里（1K、4K 是 2.05 小时，16K 是 3.25 小时，64K 是 12.6 小时），评测是 0.5 小时，只评 e 时是 0.2 小时。
- **上限**：hpcgpu07 `QUEUE_CAP_H=170`。06 加入时，管理员按剩下的组另给上限。
- 每条运行开始前，驱动按"已用 + 这一条的预估"判断，超过上限就写 `blocked` 停下。
- **第一组跑完时复核**：n0_all 跑完、small_s0 跑完第一个 1K 运行时，各把实测墙钟发给管理员。比预估慢 15% 以上，就停下报告。

## 3. 启动（方案 B，以 07 为例）

1. hpcmgt 上 `squeue`，两条 RUNNING。
2. 启动：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env NODE=hpcgpu07 PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin QUEUE_CAP_H=170 \
     bash /home/mail/chenyuma/visint/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/queue.sh \
     > /home/mail/chenyuma/visint/logs/train_0051-hpcgpu07.driver.log 2>&1 < /dev/null &'
   ```

3. 再跑一次 `squeue`。
4. 心跳：`$VI/logs/train_0051-hpcgpu07.status`；驱动日志在 `experiments/E5/0051_e5_main_hpcgpu07/driver/driver.log`。
5. 推送：另起一个 v3 循环（0053-addendum-1；脚本用 `$VI/wt/sync3/scripts/server/sync_push.sh`，不要 `SYNC_WT`），TAG `0051`，间隔 1800 秒。路径：`experiments/E5/0051_e5_main_hpcgpu07`，06 加入后再加 `experiments/E5/0051_e5_main_hpcgpu06`。

## 4. 巡检

- 同 RUN-0045：squeue、心跳、推送状态、训练进程的 RSS 不超过 150 GB。**不设份额上限**（D95）；驱动在 Lustre 剩余空间低于 3 GB 时停。
- **用量**：E5 的最终检查点每个约 0.1 GB，全部约 5 GB。巡检时照旧记一次 lfs quota 的差值，只作记录。
- **认领目录**：如果某个组只有目录、没有 `node` 文件（认领刚做完驱动就断了），不要手动删，告诉管理员。
- 某条运行失败，驱动会从检查点重试一次，还失败就记进 `failed.txt`，跳到下一条。把 `failed.txt` 的原文和那条的日志尾部发给管理员。
- **起点合并核对不过**：脚本会拒绝训练，这是规则，不要绕过。报告里照实写，包括 exact_rows。

## 5. 报告

推送 `reports/0051-hpc-e5-main.md`，只贴数字与事实：

- **启动后先推一版**：校验结果、工作树 commit、启动时刻、step 号、前后 squeue。
- **每跑完一组更新一次**：
  - 组名、节点、墙钟；
  - 每条运行的步数、每步中位、val 选中的步（1K、4K）、显存峰值；
  - test 评测的有效率与解析失败率；
  - 起点合并核对的 pass 与 exact_rows。
  - **不贴 test 的分数**，分数由管理员用分析脚本读。
- **全部跑完**：两台的 status 原文、累计 GPU 小时、份额、`e5_claims_0051/` 的清单。
