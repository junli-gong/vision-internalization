[RUN-TASK 0023] 导出 B200 上评测数据的感知哈希清单（跨机器去重用，硬约束 7；不占 GPU）

## 为什么做

- DEV-0022（D49）写好了 E2/E3 训练数据的落地与跨机器去重。
  - 训练数据落在 A800；评测数据在 B200 的 `$PROJ/data`。
  - 两边各导出"图像键 + 感知哈希"清单，再在一台机器上比较。
- 本任务导出 B200 这一侧的清单并入库。清单只有键与哈希，几 MB，不含图像。
- 下一个 A800 脚本包会带上它，在 A800 上直接出去重报告。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`（被唤醒后第 3 步：从固定 commit 的工作树运行）、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D11（第 7 条）、D49
- `python scripts/data/dedupe_report.py --help`

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `fba37dd`（DEV-0022 的合入）与本任务书所在的 commit。不包含就停下报告。

## 2. 运行

```bash
python scripts/data/dedupe_report.py --export-hashes $PROJ/data \
  --out $REPO/experiments/P0/0023_eval_hashes/eval_hashes_b200.json
```

- 不用 GPU。预计几分钟到半小时，取决于图像张数。
- 输出超过 20 MB 就停下报告，不入库。
- 根盘剩余低于 3 GB 立即停（D6）。

## 3. 产出与回传

- `experiments/P0/0023_eval_hashes/`：`eval_hashes_b200.json`、`README.md`、`logs/`。
- `reports/0023-eval-hash-export.md`，按 report 模板，第 3 节原样贴：
  - 每个数据源的图像张数；
  - 清单的字节数与 sha256；
  - 用时。

  只贴数字与事实，不做解读。
- 一次推回：

```bash
git add <逐个路径> && git commit -m "[results] 0023: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0023-eval-hash-export.md` 追加进完成清单，重新进入值守。
