# experiments/E1/0020_acceptance_v3 — RUN-0020：E1 批处理验收 v3（冻结批配置 64，B = 48；先跑 4 个模型）

服务器 CC，b200，2026-10-02 08:59–10:53 UTC。运行报告：`reports/0020-e1-acceptance-v3.md`。

## 运行用的代码

- 驱动 `probe_scripts/run_accept_v3.sh` 在一个固定于任务开始时 HEAD `90ab6b8` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0020`）。
- `configs/e1_prompts.json` 的 sha256 是 `5de1aec7…`（`logs/worktree.txt`）。
- 输出用绝对路径写进本目录。跑完 worktree 已删除；运行中它的 `git status` 为空。

## 做了什么

4 个模型依次运行：qwen3.5-0.8b、gemma-4-e2b-it、internvl3.5-2b-hf、qwen3.5-9b。每个模型执行：

```bash
python scripts/run_e1.py --acceptance --model <键> --task all --data-root $PROJ/data --out <本目录>
```

- 没有加 `--batch-size`、`--alt-batch-size`、`--allow-unfrozen-batch`，用的是冻结值：A = 64，B = 48。
- 环境：先 `source $PROJ/scripts/env.sh`，再 export `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 文件

| 路径 | 内容 |
|---|---|
| `<模型>/acceptance/acceptance.json`、`acceptance.md` | 验收 v3 的结果：总判定、(i) 逐位重跑、(ii) 按组的噪声底、v2 三种一致口径的参考列 |
| `<模型>/acceptance/A1/…`、`A2/…`、`B/…` | 三遍的全部输出。每个任务一个目录，含 `predictions.jsonl`、`metrics.json`（三列、有效率、按 person 的拒答）、`compute.json` 等 |
| `accept_v3_summary.md`、`accept_v3_summary.json` | 由 `probe_scripts/p25_accept_v3_summary.py` 生成，表见下 |
| `logs/` | `accept_<模型>.log`、`driver.log`、`worktree.txt`；`acc_gpu.csv`、`acc_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `disk_ledger.tsv` | 磁盘收支 |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出，没有 "Would install" |
| `probe_scripts/` | `run_accept_v3.sh`、`p25_accept_v3_summary.py`、`gpu_sampler.sh`、`diskcheck20.sh` |

`accept_v3_summary.md` 的表：

1. 总判定；
2. (i) 逐位重跑；
3. (ii) 噪声底；
4. 三列；
5. 拒答诊断；
6. v2 三种一致口径；
7. 计算量与显存。

## 口径

- 判据与读数都原样取自 run_e1 写出的文件（`src/e1/acceptance.py`，D38-3），汇总脚本不重算。
- run_e1 的验收不记录 torch 峰值显存；nvidia-smi 峰值按 `logs/driver.log` 里每个模型的起止时间归属。
