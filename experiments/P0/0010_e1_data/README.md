# experiments/P0/0010_e1_data — RUN-0010 证据

服务器 CC，2026-10-01。内容：E1 数据落地到 `$PROJ/data`（不入库）；核对开发子集；在完整数据上生成 NYU 点对与评测子集两份协议文件，由管理员入库。运行报告：`reports/0010-e1-data.md`。

| 文件 | 内容 |
|---|---|
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt`：只新装 pyarrow 25.0.1（h5py 3.16.0 已在 base env） |
| `pytest.txt` | `python -m pytest tests/ -q -rs`（1278 passed, 12 skipped） |
| `manifests/<源>.MANIFEST.json` ×8 | 各源 `$PROJ/data/<源>/MANIFEST.json` 的副本：来源 URL 与 revision、每个文件的 sha256 与字节数、许可原文、估算、每一步的 df、核对项、不可达 URL |
| `manifest_summary.md` | 由 `probe_scripts/p13_manifest_summary.py` 从清单与驱动日志生成的逐源汇总 |
| `e1_nyu_pairs.json` | `build_e1_configs.py nyu-pairs --root $PROJ/data --out …` 的输出（待管理员入库到 `configs/`） |
| `e1_eval_subsets.json` | `build_e1_configs.py eval-subsets --root $PROJ/data --pairs-config <上一行> --out …` 的输出（待入库） |
| `counts.json` | 由 `probe_scripts/p13_counts.py` 用加载器读出的每个评测集的 n（eval/dev，含每个加载器的剔除统计），以及点对配置的统计与约束 |
| `logs/` | 各源落地、配置生成与计数的 stdout/stderr |
| `land_driver.log`、`disk_ledger.tsv` | 逐源落地的驱动日志（退出码、墙钟、前后 df）；磁盘与内存盘收支 |
| `probe_scripts/` | `run_land.sh`（逐源落地驱动）、`p13_counts.py`、`p13_manifest_summary.py`、`diskcheck10.sh` |

开发子集 `dev-subsets --check` 与入库的 `configs/e1_dev_subsets.json` 一致，所以没有写新版。
