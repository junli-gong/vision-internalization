# experiments/E1/0027_e1_main — RUN-0027：E1 主评测（eval 全部任务 + free 自由作答诊断 + extra 附加对照，8 个模型）

服务器 CC，b200，2026-10-03 05:14–19:15 UTC。运行报告：`reports/0027-e1-main.md`。E1 预注册已批准 v2（D59-1）。

## 运行用的代码

- 驱动 `probe_scripts/run_e1_main.sh` 在一个固定于任务开始时 HEAD `a3153d2` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0027`）。
- `configs/e1_prompts.json` 的 sha256 是 `5de1aec7…`，见 `logs/worktree.txt`。
- 输出用绝对路径写进本目录。
- 汇总脚本 `probe_scripts/p28_e1_summary.py` 也是在这个 worktree 里运行的，评分代码与运行时相同。

## 每个模型执行的命令

模型顺序：qwen3.5-0.8b、2b、4b、9b、internvl3.5-2b-hf、4b-hf、gemma-4-e2b-it、e4b-it。

```bash
python scripts/run_e1.py --model <键> --task all --subset eval --data-root $PROJ/data --out <本目录>
python scripts/run_e1.py --model <键> --task rel_depth_nyu rel_depth_blink spatial_cvbench spatial_embspatial --subset free --data-root $PROJ/data --out <本目录>
# qwen3.5-0.8b、qwen3.5-9b：--policy qwen:1024；internvl3.5-2b-hf、internvl3.5-4b-hf：--policy internvl:native
python scripts/run_e1.py --model <键> --task refexp_box ovd --subset extra --policy <口径> --data-root $PROJ/data --out <本目录>
```

- **批配置：** 用冻结值，64、左填充，没有加 `--batch-size`、`--allow-unfrozen-batch`。
- **环境：** 先 `source $PROJ/scripts/env.sh`，再 export `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 文件

| 路径 | 内容 |
|---|---|
| `<模型>/<任务>/` | eval 子集的输出：`predictions.jsonl`、`metrics.json`（三列、breakdown、parse、truncation、diagnostics、refusal_by_person）、`compute.json`、`config.yaml`、`meta.json`、`README.md`、`logs/` |
| `<模型>/<任务>__free/` | free 子集，自由作答诊断，只有 4 个字母读数任务 |
| `<模型>/<任务>__extra__<口径>/` | extra 子集，附加对照：refexp_box、ovd |
| `<模型>/refexp_box/predictions.jsonl.gz`、`LARGE_FILES.sha256` | 定位的预测超过 20 MB，入库的是 gzip（mtime=0、不存文件名）。`LARGE_FILES.sha256` 记原文件的 sha256 与字节数、gz 的 sha256 与字节数；原文件只留在服务器上，不入库。8 个原文件合计 380.4 MB，gz 合计 31.0 MB |
| `e1_summary.md`、`e1_summary.json` | 由 `probe_scripts/p28_e1_summary.py` 生成的报告第 3 节的表 |
| `logs/` | 每条命令的日志 `<模型>__{eval,free,extra}.log`、`driver.log`、`worktree.txt`；`main_gpu.csv`、`main_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 磁盘收支；dry-run 的输出（没有 "Would install"） |
| `probe_scripts/` | `run_e1_main.sh`（驱动）、`prep_push.py` 与 `push_model.sh`（每个模型的入库，含 gzip）、`p28_e1_summary.py`（汇总）、`gpu_sampler.sh`、`diskcheck27.sh` |

## 汇总脚本里两处"同样本对照"的做法

- **free 对字母读数：** 取 `src.e1.tasks.select(task, "free")` 的样本，把 eval 的 predictions 里同 id 的记录按 free 的顺序排好，交给 `src.e1.scoring.aggregate` 算三列；另报两边逐条对错的配对计数。
- **extra 对主口径：** 取 `select(task, "extra")` 的样本，把 eval 里同 id 的记录交给同一个 `aggregate`。
- 两处都只调用仓库现有的评分代码，没有另写指标。
