# experiments/P0/0006_env_synth — RUN-0006 证据

服务器 CC，2026-09-30。服务器 venv 装了 pymunk；跑了全量测试；做了 E5 生成器（`src.synth`）的跨机器摘要核对与生成速度测量。运行报告：`reports/0006-env-synth.md`。

| 文件 | 内容 |
|---|---|
| `pip_dryrun.txt`、`pip_install.log` | `pip install --dry-run -r requirements.txt` 与实际安装的原始输出（只新装 cffi 2.1.1、pycparser 3.0、pymunk 7.3.0） |
| `pytest.txt` | `python -m pytest tests/ -q` 的完整输出（2 failed, 849 passed, 1 skipped；失败的两条是钉住摘要的核对） |
| `synth_digests.json` | `PINNED_CE`、`PINNED_D` 每个 case 在服务器上的摘要、钉住的摘要、是否相同、scene_seed、各分量的子摘要（sha256 前 16 位），以及不一致的 d case 在 k=5/10/20/40 各档下的服务器侧位姿与 mask 数据 |
| `diag_npy_disable_cpu_features.txt` | 诊断：设 `NPY_DISABLE_CPU_FEATURES` 关掉 numpy 的 AVX2/AVX-512 分派后重算三个 test-2 case 的摘要；另附 `np.show_runtime()` 的 simd_extensions，确认关闭生效 |
| `speed.json` | c、d、e 各 200 个样本（train，序号 0–199，档位按默认轮转）的墙钟、CPU 时间与样本/秒 |
| `disk_ledger.tsv` | 磁盘收支 |
| `probe_scripts/` | `p10_synth_probe.py`（只调用 src.synth 的公开接口；钉住的摘要从 tests/test_synth_repro.py 导入，不重抄）、`diskcheck6.sh` |
