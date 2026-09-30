# experiments/P0/0008_synth3d_check — RUN-0008 证据

服务器 CC，2026-09-30。E5 生成器第二部分（a 出平面三维旋转、b 视角变化）的跨平台逐位复现核对。运行报告：`reports/0008-synth3d-check.md`。

| 文件 | 内容 |
|---|---|
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt`：没有任何 "Would install" |
| `pytest.txt` | `python -m pytest tests/ -q -rsx` 的完整输出（1102 passed, 11 skipped） |
| `synth_check.txt` | `python scripts/synth_pin_digests.py check` 的原样输出（五个任务各 60/60 相同） |
| `disk_ledger.tsv` | 磁盘收支 |

没有出现不同，所以按任务书没有写 `synth_digests_server.json` 与 `component_digests.json`。
