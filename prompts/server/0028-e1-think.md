[RUN-TASK 0028] E1 思考子集（预算强制收尾：思考 4,096 + 作答 1,024；8 个模型；低优先级，不挡 G1）

## 为什么做

- E1 预注册已批准 v2（D59-1）。思考子集改了口径：
  - 预算强制收尾：思考写到 4,096 token 还没收尾，补上本家族的结束标记，再给 1,024 token 作答；
  - 自然收尾的条数照报；
  - 去掉 8192 敏感性检查。
- DEV-0027（D60）实现了它。思考子集用的配置 sha256 是 `f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452`。
- 这是正式实验，规则同 RUN-0027：不改参数、不挑结果、异常照实记下。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（第 3 步：固定 commit 的工作树）、`.claude/skills/exp-triad-flow/references/server.md`
- `PREREGISTRATION.md` 的 E1（§5 思考子集，§7 预测里的"思考的效果"）
- `DECISIONS.md` 的 D59、D60
- `python scripts/run_e1.py --help`

本任务只运行，不改任何仓库代码与配置。

## 1. 拉取与校验

- `git pull --ff-only`，HEAD 必须包含本任务书所在的 commit。
- `configs/e1_prompts.json` 的 sha256 必须是 `f3353de0…`，不是就停下报告。
- 建固定工作树：`git worktree add --detach /dev/shm/visint/tmp/wt0028 HEAD`。

## 2. 环境

同 RUN-0027（`VISINT_DATA_ROOT`、`NLTK_DATA`、`TMPDIR`、`PYTORCH_CUDA_ALLOC_CONF`），批配置用冻结值 64。

## 3. 运行

模型顺序同 RUN-0027。每个模型一条命令：

```bash
python scripts/run_e1.py --model <键> --task all --subset think --data-root $PROJ/data \
  --out $REPO/experiments/E1/0028_e1_think
```

- 中断了加 `--resume` 续跑。
- 退出码为 2 时贴出报错，继续下一个模型。

## 4. 时间与停止

- 估算约 12.5 GPU 小时（E1 预注册 §9）。
- 累计超过 15 小时（估算的 120%）：跑完当前模型就停，推回已完成的部分，写 `blocked e1-think-over-budget`。
- 每跑完一个模型推一次，心跳文件 `standby.status` 也更新一次：写 `running 0028 <已完成的模型数>/8`。
- 根盘剩余低于 3 GB 立即停（D6）。

## 5. 产出与回传

- `experiments/E1/0028_e1_think/`：run_e1 的全部输出、`disk_ledger.tsv`、`README.md`、`logs/`。大于 20 MB 的文件按 RUN-0027 的办法压缩。
- `reports/0028-e1-think.md`，按 report 模板，第 3 节原样贴：
  - **每个模型 × 任务**：
    - 结束方式四类的条数：自然收尾、强制收尾、没开思考段、思考段里就停；
    - 每一类的严格读数与有效率，以及全体的严格读数；
    - 思考 token 与作答 token 的中位数。
  - **与关思考的对照**：同一批 id 上，关思考读数取 RUN-0027 主评测的预测；字母读数的四个任务，另取 RUN-0027 的 free 自由作答。给出差值与配对 bootstrap CI，按全体与按结束方式分开。
  - **用量**：每个模型的 GPU 墙钟、合计，以及与估算的对照。
- 只贴数字与事实，不做解读。

完成后按 STANDBY.md，把 `0028-e1-think.md` 追加进完成清单，重新进入值守。
