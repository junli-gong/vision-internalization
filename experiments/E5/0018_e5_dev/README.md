# experiments/E5/0018_e5_dev — RUN-0018：E5 开发集核对（零样本、文本思维链、平凡基线；val 划分，每档 4 条）

服务器 CC，b200，2026-10-02 07:18–08:49 UTC。运行报告：`reports/0018-e5-dev-check.md`。

## 运行用的代码

驱动 `probe_scripts/run_e5_18.sh` 在一个固定于任务开始时 HEAD `a0766eb` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0018`）。

- 这样运行期间主 checkout 的 pull 不会改变正在运行的代码。
- 输出用绝对路径写进本目录。
- 跑完 worktree 已删除。运行中 worktree 的 `git status` 为空（`logs/worktree.txt`、`logs/driver.log` 第一行）。

## 做了什么

1. **摘要核对**：在 worktree 里运行 `python scripts/synth_pin_digests.py check`，300/300 相同，规范平台（`logs/synth_check.log`）。
2. **平凡基线**（CPU，不加载模型）：

   ```bash
   python scripts/run_e5.py --system baseline --task all --subset dev --n-per-bin 4 --out <本目录>
   ```

3. **8 个模型**，顺序同 RUN-0017，每个一条命令；批大小取提示配置的默认值 64：

   ```bash
   python scripts/run_e5.py --system zeroshot cot --model <键> --task all --subset dev --n-per-bin 4 --out <本目录>
   ```

**环境**：先 `source $PROJ/scripts/env.sh`，再 export `TMPDIR=/dev/shm/visint/tmp`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 文件

| 路径 | 内容 |
|---|---|
| `zeroshot/<模型>/<任务>__dev/`、`cot/<模型>/<任务>__dev/` | run_e5 的输出：`predictions.jsonl`、`metrics.json`、`compute.json`、`config.yaml`、`meta.json`、`README.md`、`logs/` |
| `baseline/<基线>/<任务>__dev/` | 平凡基线与参照线的输出，结构同上 |
| `e5_summary.md`、`e5_summary.json` | 由 `probe_scripts/p22_e5_summary.py` 从上面的文件生成，表见下 |
| `logs/` | `baseline.log`、`e5_<模型>.log`、`driver.log`、`synth_check.log`、`worktree.txt`；`e5_gpu.csv`、`e5_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `disk_ledger.tsv` | 磁盘收支，单位字节 |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出，没有 "Would install" |
| `probe_scripts/` | `run_e5_18.sh`、`p22_e5_summary.py`、`gpu_sampler.sh`、`diskcheck18.sh` |

`e5_summary.md` 里的表：

1. 逐模型 × 系统 × 任务：条数、有效率、解析失败、截断、思考未闭合、生成量、批大小、吞吐；
2. 峰值显存；
3. 严格读数与条件读数；
4. 平凡基线；
5. 原始输出举例；
6. 墙钟与用量合计。

## 口径

- 指标、有效率、失败原因都原样取自 `metrics.json`，即 `src/e5` 的口径（D39）。汇总脚本不重算。
- 样本/秒 = n ÷ `compute.json` 的 `wall_time_s`，后者只含 generate。
- run_e5 不记录 torch 的峰值显存，输出里没有这个字段。nvidia-smi 峰值按 `logs/driver.log` 里每个模型的起止时间归属。
