# experiments/E1/0028_e1_think — RUN-0028：E1 思考子集（预算强制收尾：思考 4,096 + 作答 1,024；8 个模型）

服务器 CC，b200，2026-10-03 19:23 – 2026-10-04 06:26 UTC。运行报告：`reports/0028-e1-think.md`。

## 运行用的代码

- 驱动 `probe_scripts/run_e1_think.sh` 在一个固定于任务开始时 HEAD `819149c` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0028`）。
- `configs/e1_prompts.json` 的 sha256 是 `f3353de0…`（D60），见 `logs/worktree.txt`。
- 汇总脚本 `probe_scripts/p29_think_summary.py` 也在这个 worktree 里运行。

## 每个模型执行的命令

模型顺序同 RUN-0027。每个模型一条命令：

```bash
python scripts/run_e1.py --model <键> --task all --subset think --data-root $PROJ/data --out <本目录>
```

- 批配置用冻结值 64，环境同 RUN-0027。
- 每个模型跑完推一次（`probe_scripts/push_model.sh`）。超过 20 MB 的文件按 RUN-0027 的办法 gzip；本运行没有超过 20 MB 的文件。

## 文件

| 路径 | 内容 |
|---|---|
| `<模型>/<任务>__think/` | 思考子集的输出：`predictions.jsonl`（含 thinking_end、n_thinking_tokens、n_answer_tokens、inserted_* 等）、`metrics.json`（含 thinking_end 的四类分解）、`compute.json`、`config.yaml`、`meta.json`、`README.md`、`logs/` |
| `think_summary.md`、`think_summary.json` | 由 `probe_scripts/p29_think_summary.py` 生成：结束方式四类、每类读数、token 中位数、与关思考的配对对照、用量 |
| `logs/` | `<模型>__think.log`、`driver.log`、`worktree.txt`；`think_gpu.csv`、`think_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 磁盘收支；dry-run 的输出（没有 "Would install"） |
| `probe_scripts/` | `run_e1_think.sh`、`push_model.sh`、`prep_push.py`、`p29_think_summary.py`、`gpu_sampler.sh`、`diskcheck28.sh` |

## 与关思考对照的做法（`p29_think_summary.py`）

- 取 `src.e1.tasks.select(task, "think")` 的样本。
- 开思考一侧用本运行的记录；关思考一侧用 RUN-0027 主评测（eval）同 id 的记录。字母读数的四个任务另取 RUN-0027 的 free 同 id 的记录。定位的 eval 预测原文件留在服务器上，读不到时读 `.gz`。
- 逐样本取值用 `src.e1.scoring.per_sample_metric`；检测用 `ovd_per_image_counts` 加 `src.stats.f1_statistic`。
- 差值与 95% CI 用 `src.stats.paired_bootstrap_ci`，按开思考 − 关思考，全体与按结束方式各算一次。
