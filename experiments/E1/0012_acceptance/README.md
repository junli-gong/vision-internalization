# experiments/E1/0012_acceptance — RUN-0012 §3 证据：批处理验收

服务器 CC，b200，2026-10-01 08:34–12:23 UTC。运行报告：`reports/0012-e1-accept-estimate.md`。

**做了什么**：对 8 个模型（按任务书 §3，先跑 4 个，有不达标后补跑其余 4 个）执行

```bash
python scripts/run_e1.py --acceptance --model <键> --task all --batch-size 16 --limit 100 --out experiments/E1/0012_acceptance
```

**环境**（`probe_scripts/run_accept.sh`）：
- 先 `source $PROJ/scripts/env.sh`；
- 再 export `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`。

**开发子集**：只用开发子集。不足 100 条的任务全取，即 count 49、rel_depth_blink 12、spatial_cvbench 50、spatial_embspatial 50。

## 文件

| 路径 | 内容 |
|---|---|
| `<键>/acceptance/acceptance.json`、`acceptance.md` | run_e1 写出的验收结果：每个任务一行，含一致率、指标差与配对 CI、raw_text 相同比例、是否通过 |
| `<键>/acceptance/bs1/<键>/<任务>__dev/` | 逐条那一遍（batch=1，走 generate）。每个任务一个目录，里面有：`predictions.jsonl`、`metrics.json`、`compute.json`、`meta.json`、`config.yaml`、`README.md`、`logs/` |
| `<键>/acceptance/bs16/<键>/<任务>__dev/` | 批量那一遍（batch=16，左填充），结构同上 |
| `accept_summary.md`、`accept_summary.json` | 由 `probe_scripts/p15_accept_summary.py` 从上面的文件生成，含四张表（见下） |
| `logs/accept_<键>.log` | 每个模型 run_e1 的 stdout/stderr，含 `time` 的输出 |
| `logs/driver.log` | 驱动日志（验收与估算共用）：权重核对、退出码、删除与释放量、累计耗时 |
| `logs/*_gpu.csv`、`logs/*_apps.csv` | 每 5 s 一次的 nvidia-smi 采样。`accept_*` 跟随第一个驱动（前 4 个模型）；`chain_*` 跟随链式脚本，从第一个驱动开始一直到估算结束 |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出（没有 "Would install"） |
| `probe_scripts/` | 见下 |

`accept_summary.md` 的四张表：

1. 验收汇总；
2. 截断率与解析失败分布；
3. 两遍不一致的细目；
4. 每遍的墙钟与生成 token。

`probe_scripts/` 里的脚本：

- `run_accept.sh`：验收驱动；
- `chain.sh`：等第一个驱动结束后，补跑 4 个模型的验收，再跑估算；每一步开始前检查累计 6 小时；
- `gpu_sampler.sh`：GPU 采样；
- `p15_accept_summary.py`：生成汇总表；
- `diskcheck12.sh`：记磁盘收支。

## `p15_accept_summary.py` 的口径

- **判据不变。** 一致、通过与否都原样取自 `acceptance.json`，即 `src/e1/acceptance.py` 的判据（D32-2）。
- **"不一致的细目"只描述两遍 parsed 的差别，不参与判定：**
  - refexp_box：两遍的框在原图像素上的 IoU，以及四个坐标的最大绝对差；
  - ovd：标签序列是否相同、框数是否相同；两者都相同时，逐框 IoU 的最小值；
  - 其余任务：一遍失败一遍成功、两遍都失败、两遍都成功但 parsed 不同的条数。
- **截断率**：取自各遍的 `metrics.json`（`truncation`，即 `stopped=False`，D32-2）。
- **墙钟与生成 token**：取自 `compute.json`。墙钟只含 generate。
