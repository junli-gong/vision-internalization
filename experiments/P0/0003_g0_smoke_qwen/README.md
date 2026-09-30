# experiments/P0/0003_g0_smoke_qwen — RUN-0003 证据

服务器 CC，2026-09-30。**RUN-0003 停在任务书 §3**：服务器全量测试 16 failed，按规定没有运行 §4 冒烟，所以这里没有冒烟 JSON。详见 `reports/0003-g0-smoke-qwen.md`。

| 文件 | 内容 |
|---|---|
| `pytest.txt` | `python -m pytest tests/ -q` 在服务器 venv 下的完整原始输出（16 failed, 295 passed, 1 skipped；含全部 traceback） |
| `pytest_diag_fla_blocked.txt` | 诊断运行：同一命令，另加 `PYTHONPATH` 指向只含 `diag_nofla_sitecustomize.py` 的目录（该进程树里 `import fla` 失败）；结果 311 passed, 1 skipped |
| `pytest_diag_skip_reason.txt` | `-rs` 输出：唯一的跳过项是 cv2 未装 |
| `diag_nofla_sitecustomize.py` | 上述诊断用的 3 行文件原样（不是修复，没有放进仓库代码路径） |
| `server_env.txt` | 服务器 torch、CUDA、torchvision、fla、triton、opencv 等版本 |
| `pip_dryrun_requirements.txt` | `pip install --dry-run -r requirements.txt` 原始输出（没有任何包会被改动） |
| `disk_ledger.tsv` | 磁盘与内存盘收支 |
