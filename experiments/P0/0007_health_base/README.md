# experiments/P0/0007_health_base — RUN-0007 证据

服务器 CC，2026-09-30。内容：环境更新（langdetect、nltk、immutabledict）；全量测试；E5 生成器 v2 摘要核对，并在规范平台上写出 d 的摘要；8 个基座模型的健康检查（D26 协议；IFEval 只生成、不评分，见 N12）。运行报告：`reports/0007-health-base.md`。

| 文件 | 内容 |
|---|---|
| `pip_dryrun.txt`、`pip_install.log` | 依赖 dry-run 与实际安装的原始输出 |
| `pytest.txt` | 任务书命令 `python -m pytest tests/ -q -rs -rx` 的完整输出。后一个 `-r` 盖过前一个，所以没有打印跳过原因 |
| `pytest_rsx.txt` | 用 `-rsx` 重跑一次的完整输出（列出 11 个跳过项的原因） |
| `synth_check.txt` | `python scripts/synth_pin_digests.py check` 的原样输出 |
| `synth_digests_v2.json` | `python scripts/synth_pin_digests.py write --only physics --out …`：以仓库文件为底，只换 d 的清单（本平台上的规范值，状态 canonical），由管理员复制回 `tests/data/` |
| `health_<键>.json` | 8 个模型的健康检查输出（第 2、3 批加入） |
| `health_<键>.log` | 对应的 stdout/stderr（第 2、3 批加入） |
| `disk_ledger.tsv` | 磁盘与内存盘收支 |
