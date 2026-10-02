# experiments/E5/0019_lisa_e5_dev — RUN-0019：LISA-7B-v1 在 E5 开发集上的首次运行（外部对照，隔离；val 划分，每档 4 条，a–d）

服务器 CC，b200，2026-10-02 08:50–09:00 UTC。运行报告：`reports/0019-lisa-e5-dev.md`。

## 隔离（硬约束 1，D36-5、D40-3）

- LISA 的输出只在本目录的两个外部预测目录里：`lisa_predictions/` 和 `lisa_predictions_limit5/`。
- 评分由 `scripts/run_e5.py --system external` 完成。
- 没有写 `$PROJ/data`，也没有写任何训练或数据目录。
- 导出的输入（`$PROJ/tmp/e5_inputs_dev_0019`，372,418 B，64 个文件）用完已删除。
- LISA 的 venv 与代码沿用 RUN-0015 的 `$PROJ/ext/lisa/`。LISA clone 的 HEAD 是 `3cb2d43…`，没有改动。

## 运行用的代码

全部步骤都在固定于任务开始时 HEAD `60fba2b` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0019`），跑完已删除。

| 步骤 | 用哪个 venv | 命令 |
|---|---|---|
| 1. 权重 | 项目 venv | `experiments/P0/0015_lisa_smoke/probe_scripts/p18_download.py` → `weights_download.json`。两份权重逐文件核对，哈希与 RUN-0015 相同 |
| 2. 导出 | 项目 venv | `run_e5.py --export-inputs $PROJ/tmp/e5_inputs_dev_0019 --task a b c d --subset dev --n-per-bin 4` |
| 3. 彩排 | LISA venv | `scripts/external/lisa_e5_infer.py … --limit 5` → `lisa_predictions_limit5/`（`probe_scripts/run_lisa_19.sh`） |
| 4. 全量 | LISA venv | 同上，不加 `--limit` → `lisa_predictions/` |
| 5. 评分 | 项目 venv | `run_e5.py --system external --model lisa-7b-v1 --external-dir lisa_predictions …` 写出 `external/`；`run_e5.py --system baseline --task a b c d …` 写出 `baseline/` |
| 6. 收尾 | — | 从内存盘删掉 LISA 与 CLIP；删掉导出目录 |

## 文件

| 路径 | 内容 |
|---|---|
| `lisa_predictions/`、`lisa_predictions_limit5/` | 外部预测目录（format_version 1）：`EXTERNAL_PREDICTIONS.json`、`run_config.json`、`<任务>.jsonl`、`logs/run_001.json` |
| `external/lisa-7b-v1/<任务>__dev/` | LISA 的评分输出 |
| `baseline/<基线>/<任务>__dev/` | 同一批样本上的平凡基线 |
| `overlays/` | 彩排输出里每个任务前 2 条成功样本的"原图 \| mask 叠加图"与 text，由 `probe_scripts/p23_overlays.py` 生成；`index.json` 是清单 |
| `lisa_summary.md`、`lisa_summary.json` | 由 `probe_scripts/p24_lisa_summary.py` 生成：加载、逐任务、评分、平凡基线、文本举例 |
| `weights_download.json` | 两份权重的下载与逐文件核对 |
| `logs/` | 下载、导出、彩排、全量、两次评分的日志，`driver.log`，`worktree.txt`；`lisa5_*.csv`、`lisafull_*.csv` 是 nvidia-smi 采样 |
| `disk_ledger.tsv` | 磁盘收支 |
| `probe_scripts/` | `run_lisa_19.sh`、`p23_overlays.py`、`p24_lisa_summary.py`、`diskcheck19.sh` |
