[RUN-TASK 0051-addendum-1] hpcgpu07 的 E5 换成合并核对判据 e5-v2（D96、D97）：主驱动暂停，wt0051 切到新 commit，补评被 v1 拒绝的检查点，已评过的只补核对，然后主驱动接着跑

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 本任务**不停、不杀任何驱动进程**。主驱动只靠暂停文件在两步之间等着。
- 06 上的 RUN-0052 不碰。

## 为什么

- **D96（用户，10-15）**：N34 选 (b′)。E5 训练后评测前的合并核对改用判据 e5-v2：
  - 判允许集合上的合并前后一致率、受限教师强制，以及所有比较里分歧处的 gap；
  - 全词表的合并一致率和 exact_rows 改为只报告。
  - 所有 E5 运行统一适用，被 v1 拒绝的检查点统一补评。
  - 另加一项指标层面的合并前后比较（val 上的结构强制解码，比较 mask IoU 与起伏误差），只报告。
- **DEV-0046 已合入**（commit `2db10553`；本任务切到的 commit 是 `f703d7dbb25817d097a574ebb0e61cd3cbc91f57`，含 DEV-0046 与 `reeval.sh`）：
  - `src/e5/trained_eval.py`：判据 e5-v2、v1 只报告、指标层面的比较。
  - `scripts/eval_e5_trained.py`：加 `--check-only`，只做核对、不评 test。
  - 旧的 v1 核对文件会自动改名留存为 `merge_check.v1.json`。
- 到 10-16 01:15 UTC，07 上被 v1 拒绝的有 14 个（`failed.txt`）；v1 下通过并评过 test 的有 2 个（e3b_c3 4K 种子 0、e3b 4K 种子 0）。
- **补评都在 07 上做**：同一组的评测在训练它的节点上（D93-1）。
- **主驱动的脚本一个字节都不改**：
  - `queue.sh`（sha256 `8f33ad28…`）与 `scripts/server/train_queue.sh`（`fcb10f47…`）在新 commit 里不变；
  - 正在跑的 bash 读的那两个文件，切换后照样是原样；
  - 只有 python 代码换了，之后每一条评测命令都用新判据。

## 新文件

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0051-hpc-e5-main/reeval.sh` | `7593ec6853ea5909a413877050827f292ebb5db6828cb6788a7977a83aa1d8ba` |
| `src/e5/trained_eval.py` | `25a0596bb59f2e298339fdb76a249317d4a00dba15e688cc80eb81bdf5fdb2fe` |
| `scripts/eval_e5_trained.py` | `90dcb73889cbec320d5c79770cc2b892e46a1f26f0d411fc4e4f74fb5a7b25a7` |
| `prompts/server_hpc/0051-hpc-e5-main/queue.sh`（不变） | `8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769` |
| `scripts/server/train_queue.sh`（不变） | `fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa` |

`reeval.sh` 做什么：
- **开始前核对**：工作树的 commit 等于 `EXPECT_COMMIT`；主驱动的暂停文件存在；主驱动心跳是 paused。任何一条不满足就停，不用卡。
- **名单**：开始时定一次，写进 `experiments/E5/0051_e5_main_hpcgpu07/reeval_e5v2/plan.tsv`。
  - 补评：主驱动 `failed.txt` 里 `eval_*` 的名字，不在主驱动 `done.txt` 里的。命令与 queue.sh 的 e5() 相同，`--out` 是主驱动的结果目录。
  - 只核对：主驱动 `done.txt` 里 `eval_*` 的名字，核对文件还不是 e5-v2 的，跑 `--check-only`。
- **运行**：
  - 用 train_queue.sh 的 `run_cmd` 逐条跑，失败重试一次；连续两条失败就停。
  - 驱动记录放在 `reeval_e5v2/driver/`，不占主驱动的用量上限。
  - 补评成功的名字也追加进主驱动的 `done.txt`。
- **最后**：
  - 写 `reeval_e5v2/summary.tsv`：每个运行的 e5-v2 判定、v1 判定、指标层面的差与标记。不含 test 分数。
  - 删掉主驱动的暂停文件 `$VI/logs/train.pause.hpcgpu07`，主驱动接着跑。
- 心跳：`$VI/logs/train_0051r-hpcgpu07.status`。

开发机上的 tiny 彩排（10-16 02:36 UTC，CPU）：
- 补评 3 个、只核对 1 个，名单与设定相同。
- 两个正常的补评通过。a–d 迁移的那个只评 e 类，同 queue.sh。
- 一个病态检查点在 e5-v2 下判不过，没写任务目录，记进补评的 failed.txt，脚本照样往下跑。
- 只核对的那个，已有的任务输出 sha256 不变。
- 跑完删掉了主驱动的暂停文件，补评成功的名字追加进了主驱动的 done.txt。

## 做法

1. hpcmgt 上 `squeue`，两条 RUNNING。
2. **暂停 07 的主驱动**（在两步之间等，不杀进程）：

   ```bash
   echo "RUN-0051-addendum-1: switch to criteria e5-v2 and re-eval (D96)" > /home/mail/chenyuma/visint/logs/train.pause.hpcgpu07
   ```

   - 等 `train_0051-hpcgpu07.status` 变成 `paused (pause file: …)`。当前这一段或这条评测跑完才会停，最多约 1.5 小时。
   - 停住时记下：主驱动停在哪一条之前，`failed.txt` 与 `done.txt` 的行数。
3. **切换 wt0051**（主驱动已暂停，GPU 上没有我们的进程）：

   ```bash
   flock /home/mail/chenyuma/visint/logs/git.lock -c 'git -C /home/mail/chenyuma/visint/wt/wt0051 fetch -q origin main'
   git -C /home/mail/chenyuma/visint/wt/wt0051 checkout -q --detach f703d7dbb25817d097a574ebb0e61cd3cbc91f57
   git -C /home/mail/chenyuma/visint/wt/wt0051 rev-parse HEAD     # 应为 f703d7dbb25817d097a574ebb0e61cd3cbc91f57
   git -C /home/mail/chenyuma/visint/wt/wt0051 status --short      # 应为空
   ls /home/mail/chenyuma/visint/wt/wt0051/experiments/            # 稀疏规则不变，应只有 P0
   ```

   核对上表 5 个文件的 sha256。`queue.sh` 与 `train_queue.sh` 必须与原值相同。
4. **在 07 上跑新增的测试**（CPU，GPU 空着；约 10–20 分钟）：

   ```bash
   ssh -n hpcgpu07 'srun --jobid=157297 --overlap -n1 env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     HF_HOME=/home/mail/chenyuma/visint/hf HF_HUB_CACHE=/home/mail/chenyuma/visint/hf/hub HF_HUB_OFFLINE=1 TMPDIR=/home/mail/chenyuma/visint/tmp \
     bash -c "cd /home/mail/chenyuma/visint/wt/wt0051 && CUDA_VISIBLE_DEVICES= python -m pytest tests/test_e5_merge_check_e5v2.py -q -p no:cacheprovider --basetemp=/home/mail/chenyuma/visint/tmp/pt_0051a1"' < /dev/null
   ```

   - **更正（10-16 03:30 UTC）**：第一版的命令漏了 `HF_HOME`、`HF_HUB_CACHE`、`HF_HUB_OFFLINE`。测试去连 huggingface.co，5 条用到真实模型配置的测试失败（a100-69 03:22 报告）。上面已补上，与 reeval.sh、queue.sh 自己 export 的相同。

   全部通过才往下。不过就停下，把输出发给管理员，**暂停文件留着**，主驱动保持暂停。跑完删掉 `$VI/tmp/pt_0051a1`。
5. **启动补评**（方案 B）：

   ```bash
   ssh -n hpcgpu07 'setsid nohup srun --jobid=157297 --overlap -n1 \
     env NODE=hpcgpu07 EXPECT_COMMIT=f703d7dbb25817d097a574ebb0e61cd3cbc91f57 QUEUE_CAP_H=14 PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin \
     bash /home/mail/chenyuma/visint/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/reeval.sh \
     > /home/mail/chenyuma/visint/logs/train_0051r-hpcgpu07.driver.log 2>&1 < /dev/null &'
   ```

   - 再跑一次 `squeue`。
   - 预计：14 个补评，每个约 0.6 小时；2 个只核对，每个约 0.15 小时。合计约 8.5–9 小时。上限 14 小时。
6. **第一个补评跑完时**：
   - 把它的 `merge_check.json` 摘要发给管理员：判据、pass、v1_report_only.pass、`metric_level` 的 status、总体差、两个标记、`timing_s`；同时附 `plan.tsv`。
   - **不要发 test 分数**：metrics.json 里的读数一律不贴。
7. **全部跑完**：
   - 主驱动的暂停文件应已被删掉，主驱动心跳从 paused 变回 running；
   - 贴 `summary.tsv` 原文，`reeval_e5v2/driver/failed.txt`（如果有）原文。
   - 主驱动恢复后跑的第一条评测，把它 stderr 里那一行 `[合并核对·指标层面…]` 发给管理员，确认主驱动用的是新判据。
8. 推送：`reeval_e5v2/` 在 `experiments/E5/0051_e5_main_hpcgpu07` 下，0051 的 v3 推送循环会带上，不用另起。

## 异常

- 补评脚本停了（blocked）：主驱动的暂停文件留着，主驱动保持暂停。把 `train_0051r-hpcgpu07.status`、`reeval_e5v2/driver/driver.log` 的尾部发给管理员，等指示。不要手动删暂停文件。
- 某个补评在 e5-v2 下也没过：照规则不评 test，记进 `reeval_e5v2/driver/failed.txt`，把那份 `merge_check.json` 的 failures 原文发给管理员。
- `metric_level.error` 不空，或 stderr 出现 `警告：[合并核对·指标层面] 出错`：不影响判定，照样往下跑，在报告里列出。

## 报告

追加进 `reports/0051-hpc-e5-main.md`，新开一节"addendum-1"，只贴事实：
- 暂停与恢复的时刻；
- 切换前后的 commit 与 sha256 核对；
- 测试结果原文；
- `plan.tsv`、`summary.tsv` 原文；
- 补评脚本的墙钟。
