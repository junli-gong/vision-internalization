# experiments/P0/0004_g0_smoke — RUN-0004 证据（G0 冒烟，三家族 + Qwen3.5-4B）

服务器 CC，2026-09-30。运行报告：`reports/0004-g0-smoke-all.md`。

| 文件 | 内容 |
|---|---|
| `pytest.txt` | `python -m pytest tests/ -q` 在服务器 venv 下的完整原始输出（558 passed, 1 skipped） |
| `pytest_skip_reason.txt` | `tests/test_metrics_seg.py -rs`：唯一的跳过项是 cv2 未装 |
| `pip_dryrun_requirements.txt` | `pip install --dry-run -r requirements.txt` 原始输出（不改动任何包） |
| `smoke_*.json` | 四份真实权重冒烟 JSON（第 2 批加入） |
| `smoke_*.log`、`disk_ledger.tsv` | 冒烟的 stdout/stderr；磁盘与内存盘收支（第 2 批加入） |
