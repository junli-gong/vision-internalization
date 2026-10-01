# experiments/E1/0013_acceptance — RUN-0013 §3：批处理验收（新的生成上限，三种一致口径）

服务器 CC，b200，2026-10-01 14:51–21:14 UTC。运行报告：`reports/0013-e1-accept-estimate-v2.md`。计数任务用 CountBenchQA 人工问句跑的那一遍，在旁边的 `experiments/E1/0013_acceptance_countbenchqa/`。

## 做了什么

8 个模型依次运行，驱动脚本是 `probe_scripts/run_accept13.sh`，由 `probe_scripts/chain13.sh` 启动。每个模型执行以下命令：

```bash
python scripts/run_e1.py --acceptance --model <键> --task all --batch-size 16 --limit 100 --agreement bitwise --out experiments/E1/0013_acceptance
python scripts/run_e1.py --acceptance --model <键> --task all --batch-size 16 --limit 100 --resume --agreement iou90   --out experiments/E1/0013_acceptance
python scripts/run_e1.py --acceptance --model <键> --task all --batch-size 16 --limit 100 --resume --agreement outcome --out experiments/E1/0013_acceptance
python scripts/run_e1.py --acceptance --model <键> --task count --count-questions countbenchqa --batch-size 16 --limit 100 --agreement bitwise --out experiments/E1/0013_acceptance_countbenchqa
#   随后同样用 --resume --agreement iou90 / outcome 各重算一遍汇总
```

- **环境**：先 `source $PROJ/scripts/env.sh`，再 export `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`。
- **`--resume` 那几遍**不重新生成。所有 id 都已在 `predictions.jsonl` 里，日志里没有生成进度行。这几遍和 CountBenchQA 的计数那遍，每遍 6–37 s，含加载模型。
- **汇总文件会被覆盖**：每次运行都改写 `acceptance.json` 和 `acceptance.md`。所以驱动脚本在每一遍之后，把它们复制成 `acceptance__<口径>.json` 和 `acceptance__<口径>.md`。
  - 三种口径每一遍都会算，所以三份里每行的 `agreement_by_mode` 和顶层的 `all_pass_by_agreement` 相同。
  - 不同的只有与"本次选用的口径"有关的字段：
    - 顶层：`header.agreement`、`criteria.agreement`、`agreement`、`all_pass`、`not_evaluated`、`written_utc`；
    - 每行：`agreement`、`answer_agreement`、`n_disagree`、`disagree_ids`、`pass`。
- **CountBenchQA 那一遍为什么另放目录**：`src/e1/runner.leaf_dir` 的目录名不含问句来源。如果写进同一个 `--out`，会和启发式问句的 `count__dev` 撞名：不加 `--resume` 会因目录非空报错，加了又会因配置不同被拒绝。所以另放到 `experiments/E1/0013_acceptance_countbenchqa/`，那里的结构与本目录相同。

## 文件

| 路径 | 内容 |
|---|---|
| `<键>/acceptance/acceptance__{bitwise,iou90,outcome}.{json,md}` | 三种口径的汇总。`acceptance.json`、`acceptance.md` 与 `__outcome` 版相同，是最后写的那一遍 |
| `<键>/acceptance/bs1/…`、`bs16/…` | 两遍的全部输出：每个任务一个目录，有 `predictions.jsonl`、`metrics.json`、`compute.json` 等 |
| `accept_summary.md`、`accept_summary.json` | 由 `probe_scripts/p17_accept_summary.py` 从上面的文件和 `0013_acceptance_countbenchqa/` 生成，含六张表（见下） |
| `countbenchqa_questions_MANIFEST.json`、`land_countbenchqa.log` | 落地 CountBench 人工问句的清单副本和日志：`python scripts/data/land_e1.py countbenchqa_questions --root $PROJ/data` |
| `logs/` | `accept_<键>[__<口径>].log`、`acceptc_<键>[__<口径>].log`（CountBenchQA）、`driver.log`；`chain_gpu.csv`、`chain_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出，没有 "Would install" |
| `disk_ledger.tsv` | 磁盘收支，本运行的验收与估算共用，在第二批推送 |
| `probe_scripts/` | 见下 |

`accept_summary.md` 的六张表：

1. 三种口径的验收汇总；
2. 每种口径的总判定；
3. CountBenchQA 问句；
4. 截断率与解析失败；
5. 诊断：failure_detail、自发思考标记、思考未闭合；
6. 每遍的墙钟与生成 token。

`probe_scripts/` 里的脚本：

- `run_accept13.sh`：验收驱动；
- `chain13.sh`：先跑验收，再跑估算；
- `gpu_sampler.sh`：GPU 采样；
- `p17_accept_summary.py`：生成汇总表；
- `diskcheck13.sh`：记磁盘收支。

## 口径

一致与通过的判据、诊断的规则都原样取自 `src/e1/acceptance.py` 与 `src/e1/scoring.py`（D33）。汇总脚本只读文件，不重算。
