# experiments/E3/0034_b200_prep — RUN-0034：B200 训练前的准备（训练数据落地到内存盘、去重、跨机器一致性 2B × 50 步、吞吐 2B/9B × 6 个变体 × 20 步）

服务器 CC，b200，2026-10-07 03:44–04:41 UTC。运行报告：`reports/0034-b200-e3-prep.md`。

## 运行用的代码

- 驱动 `probe_scripts/run_0034.sh` 在一个固定于任务开始时 HEAD `6e77ab0` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0034`），跑完已删除。
- 两个训练配置的 sha256：
  - `configs/e3b_train.json`：`35e69b56…`
  - `configs/e3a_train.json`：`28be66e5…`
  - 都与任务书相同，见 `logs/worktree.txt`。
- 驱动用 `setsid nohup` 启动，与 CC 的进程脱钩。它自己推送每个阶段的结果，每一步与每 30 分钟更新一次心跳。
- 每 30 秒看一次内存：可用内存低于 60 GB 就停下，并删掉内存盘上的训练数据。本次没有触发。
- 训练命令的环境照 A800 的 `prompts/server_a100/0035-a800-e3/common.sh`：
  - `HF_HUB_OFFLINE=1`（权重先下载并逐文件核对）；
  - `OMP_NUM_THREADS=MKL_NUM_THREADS=16`；
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
  - `--data-root /dev/shm/visint/train_data --resume`。

## 做了什么

1. **内存检查**（落地前、后与结束时）：`free -g`、`df -h /dev/shm`，见 `logs/mem_*.txt`。
2. **落地：** 在 worktree 里运行 `scripts/data/land_train.py all --root /dev/shm/visint/train_data --tmpdir /dev/shm/visint/tmp --workers 8`。
   - 落地清单的副本在 `land_manifests/`（gzip）。
   - 与 A800 的清单逐文件对照，由 `probe_scripts/p33_manifest_compare.py` 生成 `land_compare.md/json`。
3. **链接 E1 数据：** `train_data/refcoco → $PROJ/data/refcoco`，`train_data/nyuv2 → $PROJ/data/nyuv2`。
4. **去重：** `scripts/data/dedupe_report.py --train-root … --eval-manifest experiments/P0/0023_eval_hashes/eval_hashes_b200.json --out train_data/dedupe`。
   - 输出的副本在 `dedupe/`。
   - 与 A800 剔除清单的对照由 `probe_scripts/p34_exclusions_compare.py` 生成 `dedupe_compare.md/json`。
5. **一致性：** qwen3.5-2b，`train_e3b.py --supervision grid` 与 `train_e3a.py --variant full`，种子 0，各 50 步。结果在 `0034_e3_consistency/`。
6. **吞吐：** qwen3.5-2b 与 qwen3.5-9b × 6 个变体（e3a full/c1/c2、e3b、c1prime、c3），种子 0，各 20 步。结果在 `0034_e3_throughput/`。
   - 汇总由 `probe_scripts/p35_tput_table.py` 生成 `tput_table.md/json`。
7. 检查点写在 `/dev/shm/visint/tmp/e3_ckpt_0034`，跑完已删除（5,547,240,516 B，103 个文件）。**训练数据留在内存盘上。**

- `--out` 的目录名：任务书写的是 `$OUT/consistency` 与 `$OUT/throughput`。训练脚本要求目录名带运行编号，第一遍因此 rc=2，见报告 §4。实际用的是 `0034_e3_consistency` 与 `0034_e3_throughput`，与 A800 的 `0035_e3_consistency`、`0035_e3_throughput` 同一写法。

## 文件

| 路径 | 内容 |
|---|---|
| `0034_e3_consistency/<变体>/qwen3.5-2b/seed0/` | `train_log.jsonl`（50 步）、`accounting.json`、`compute.json`、`config.yaml`、`meta.json`、`tokens_init.json`、`logs/`（`run_001.json`、`timing_run_001.jsonl`）；不含检查点 |
| `0034_e3_throughput/<变体>/<模型>/seed0/` | 同上，20 步 |
| `tput_table.md/json` | 每条训练命令的 rc、墙钟、每步中位耗时（取自 `timing_run_001.jsonl`）与显存峰值（`accounting.json`） |
| `land_compare.md/json`、`land_manifests/` | 落地清单与 A800 的对照、四个源的 MANIFEST（gzip） |
| `dedupe/`、`dedupe_compare.md/json` | 去重报告与 `train_exclusions.json` 的副本、与 A800 的对照 |
| `logs/` | 每步的日志、`driver.log`、`mem_*.txt`、`worktree.txt`、`push_*.txt`；`prep*_gpu.csv`、`prep*_apps.csv` 是 nvidia-smi 采样；`*__pass1_rc2.log` 是第一遍 rc=2 的日志 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 收支：根盘、`/dev/shm/visint`、`$PROJ`、训练数据、MemAvailable；dry-run 输出（没有 "Would install"） |
| `probe_scripts/` | `run_0034.sh`、`push_phase.sh`、`prep_push.py`、`p33_manifest_compare.py`、`p34_exclusions_compare.py`、`p35_tput_table.py`、`gpu_sampler.sh`、`diskcheck34.sh` |
