[RUN-TASK 0037] B200 后台队列：E5 主比较（2B）。先跑 N = 0 的读数，再训 E5，每个检查点训完在 test 上评一次

## 为什么做

- E5 预注册 §8 已批准 v2（D81），依据是 D77-3、D78、D81。尺寸 2B；起点用 RUN-0036 训的 2B：基座加预热、E3b、E3a（full）。
- 队列的内容与顺序：
  0. **N = 0**：基座加预热与 E3b 的起点，种子 0–2，不经 E5 训练直接评 test（DEV-0040，`eval_e5_trained --start-only`）。用来区分截距与斜率，最先跑，顺带检验评测入口。
  1. 1K、4K × 三个起点 × 种子 0–2，H5 的主信号；
  2. 16K × 三个起点 × 种子 0–2；
  3. 64K × 三个起点 × 种子 0；
  4. a–d 迁移：4K，起点 E3b、E3a、基座加预热，种子 0–2；test 只评 e；
  5. 诊断起点"基座、不预热"：1K、4K，种子 0。
- 每个 E5 运行训完，紧接着用 `eval_e5_trained --checkpoint` 在 test 上评一次。1K、4K 与 a–d 的检查点是 val 上选中的那一步；test 只评这一次。
- 队列会等 RUN-0036 跑完（`train_0036.done`）才开始。可以提前启动，它会先等。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（"后台训练队列"与"批了上限的运行"两节）、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D75、D77-3、D78、D79、D81
- `scripts/server/train_queue.sh` 与 `prompts/server/0037-b200-e5-main/queue.sh`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

- `flock $PROJ/logs/git.lock -c 'git -C $REPO pull --ff-only'`，HEAD 必须包含本任务书所在的 commit。
- 建固定工作树：`git -C $REPO worktree add --detach /dev/shm/visint/tmp/wt0037 HEAD`。
- 核对 sha256：
  - `configs/e5_train.json`：`5aa1339b36c6b9643ae072aa9983f2be0db37fd20a96186af93458941cf692c1`
  - `configs/e5_prompts.json`：`c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4`
  - `configs/e3b_train.json`：`35e69b56324281356aba22419391fe8d092e3c0f0413ee1af4ed8ea74b60f457`
- 起点检查点：RUN-0036 跑完之后，`$PROJ/ckpt/0036` 下 9 个起点都在，`<检查点>.sha256.txt` 逐文件核对通过：
  - `warmup/qwen3.5-2b/seed{0,1,2}/ckpt/final/model`
  - `e3b/qwen3.5-2b/seed{0,1,2}/ckpt/final/model`
  - `e3a/full/qwen3.5-2b/seed{0,1,2}/ckpt/final/model`
  - 如果启动时 RUN-0036 还没跑完，这一项等它跑完再核对，写进第一份进度报告。
- `pip install --dry-run -r requirements.txt` 没有任何 "Would install"。
- E5 的 d 类数据只在规范平台上生成（`src/e5/runner.check_platform`）。B200 就是 RUN-0032 用的平台，照常。

## 2. 用量预估与上限（D81-2）

按 RUN-0036 实测的 2B 每步约 2.0 s（E5 与 E3b 同为 32×32 网格输出）；评测按 RUN-0032 的 2B 每条约 1.6 s 推算：

| 项 | 运行数 | 每条预估（小时） | 合计 |
|---|---:|---:|---:|
| N = 0 评测（含起点的合并核对） | 6 | 0.5 | 3 |
| 训练 1K、4K（各 1,000 步，含 val 选择） | 18 | 0.8 | 14.4 |
| 训练 16K（2,000 步） | 9 | 1.1 | 9.9 |
| 训练 64K（8,000 步） | 3 | 4.5 | 13.5 |
| 训练 a–d 迁移（4K，1,000 步，含 val） | 9 | 0.8 | 7.2 |
| 训练诊断起点 base（1K、4K） | 2 | 0.8 | 1.6 |
| test 评测，五类（含合并核对） | 32 | 0.5 | 16 |
| test 评测，只评 e | 9 | 0.15 | 1.4 |
| **合计** | | | **约 67** |

- **上限 80 GPU 小时**，按跑命令的墙钟累计，不算暂停与等待。队列在每条运行开始前判断：已用加这一条的预估超过上限，就停下、写 `blocked`、推送，不等跑完再停。
- 开跑前，请按你手上的实测（RUN-0036 的每步耗时、RUN-0038 的评测墙钟）复核这张表。预计合计超过 80 小时，就先不启动，推送一份短报告，写清逐项预估。

## 3. 启动

```bash
setsid nohup bash /dev/shm/visint/tmp/wt0037/prompts/server/0037-b200-e5-main/queue.sh \
  > $PROJ/logs/train_0037.driver.log 2>&1 < /dev/null &
```

- 心跳：`$PROJ/logs/train_0037.status`；没等到 RUN-0036 跑完时是 `waiting for queue 0036`。
- 结果：`experiments/E5/0037_e5_main/`，每条完成后驱动自己推送。训练在 `train/…`，评测在 `trained/…`，N = 0 在 `trained/<起点>/<模型>/n0/…`。最终检查点放 `$PROJ/ckpt/0037`，账在 `ckpt_ledger.tsv`。
- 评测要插队时照 STANDBY 暂停（例如 E3 的评测）。

## 4. 启动报告与回到值守

队列进入第一条运行（`n0_base_warmup_s0`），或者在等 RUN-0036 时：

- 推送 `reports/0037-b200-e5-main-start.md`，内容：
  - 第 1 节的校验；
  - 第 2 节你复核后的预估；
  - 驱动的进程号与会话号；
  - `train_0037.status` 原文。
- 把 `0037-b200-e5-main.md` 追加进 `$DONE`，回到值守。

## 5. 收尾

队列跑完（`done 0037`、推送 `queue complete`）后，管理员发 addendum 写最终报告。最终报告里要有一项核对（DEV-0040 的建议，D81-1"顺带检验评测入口"）：同一起点、同一种子，N = 0 的 `config.yaml` 里 `checkpoint.fingerprint.value` 与 E5 训练的 `e5_stack.json` 记下的 fingerprint 应相同。

## 6. 停止线（驱动自己执行）

- 同 RUN-0036：内存可用低于 60 GB 就停，并删掉内存盘上的训练数据；根盘剩余低于 3 GB 停；本项目根盘占用超过 15 GB 停；连续两条失败停。
- 加上第 2 节的用量上限。
- **根盘份额**：E5 的最终检查点每个约 130 MB，41 个约 5.3 GB。加上现有的约 9 GB，接近 15 GB。驱动触线会停；真到那时，管理员把已评测过的检查点挪到开发机归档。
