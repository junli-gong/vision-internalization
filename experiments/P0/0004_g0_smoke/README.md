# experiments/P0/0004_g0_smoke — RUN-0004 证据（G0 冒烟，三家族 + Qwen3.5-4B）

服务器 CC，2026-09-30。运行报告：`reports/0004-g0-smoke-all.md`。

| 文件 | 内容 |
|---|---|
| `pytest.txt` | `python -m pytest tests/ -q` 在服务器 venv 下的完整原始输出（558 passed, 1 skipped） |
| `pytest_skip_reason.txt` | `tests/test_metrics_seg.py -rs`：唯一的跳过项是 cv2 未装 |
| `pip_dryrun_requirements.txt` | `pip install --dry-run -r requirements.txt` 原始输出（不改动任何包） |
| `smoke_*.json` | 四份真实权重冒烟 JSON（第 2 批加入） |
| `smoke_*.log` | 四次冒烟的 stdout/stderr（原样） |
| `smoke_summary.md` | 由 `probe_scripts/p8_smoke_summary.py` 从四份 JSON 生成的关键数值表（report §3 即此表） |
| `shm_delete_gemma.txt`、`shm_delete_4b.txt` | 用 `scan_cache_dir().delete_revisions()` 删除内存盘上 Gemma 与 4B 的输出（预计释放字节数，以及删除后的仓库清单） |
| `test_image_sha256.txt` | 测试图的 sha256（本次下载的与 RUN-0001 的副本相同） |
| `disk_ledger.tsv` | 磁盘与内存盘收支 |
| `probe_scripts/` | `p8_smoke_summary.py`（渲染摘要）、`diskcheck4.sh`（记账） |
