# experiments/E1/0025_determinism_probe — RUN-0025：E1 验收 (i) 的诊断（qwen3.5-2b 定位同配置重跑那 1 条不同）

服务器 CC，b200，2026-10-02 13:34–13:41 UTC。运行报告：`reports/0025-determinism-probe.md`。

## 运行用的代码

- 驱动 `probe_scripts/run_probe_25.sh` 在一个固定于任务开始时 HEAD `317a09e` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0025`），跑完已删除。
- 探针 `rerun_batch.py` 与 `run_e1_det.py` 复制到 worktree 的同一相对路径后运行。删除前核对过，两份副本与本目录的版本逐字节相同。
- 输出用绝对路径写进本目录。

## 做了什么

只用 qwen3.5-2b。每种设置都在新开的进程里跑。

| 设置 | 内容 | 遍数 |
|---|---|---|
| S1 | RUN-0024 的环境（`VISINT_DATA_ROOT`、`TMPDIR=/dev/shm/visint/tmp`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`） | 1 个进程 × 20 遍 |
| S2a/b/c | 同 S1 | 3 个新进程 × 各 3 遍 |
| S3 | S1，加 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；探针在加载模型之前 `torch.use_deterministic_algorithms(True, warn_only=True)` | 1 × 20 |
| S4 | S3，加 `FLA_CACHE_MODE=default`（fla 0.5.2 唯一与固定内核配置有关的变量；本机没有它的配置文件，见报告 §3） | 1 × 20 |
| full_S1 | `run_e1.py --acceptance --model qwen3.5-2b --task refexp_box`，S1 环境（A1、A2、B 三遍） | 1 |
| full_S3 | 同上，S3 环境：`probe_scripts/run_e1_det.py` 先打开确定性算法，再调 `run_e1.main` | 1 |

**探针 `rerun_batch.py` 的做法：**

- 用现有接口取 refexp_box 开发子集（600 条），批划分与 RUN-0024 的 A1 相同（冻结批大小 64）。取第 4 批（下标 3，第 192–255 条，含 `refcoco+_unc/val/7909`）。
- 用 `generate_rows` → `adapter.generate_batch` 重复生成，记录每遍每条的 raw_text 与 token id。
- 与 run_e1 的唯一不同：多传一个只读的 logits 处理器，记录每一步的前两名，原样返回 scores。

## 文件

| 路径 | 内容 |
|---|---|
| `probe_out/S1.json` … `S4.json` | 每种设置的探针输出：环境、版本、批范围、每遍与第 1 遍的差别、每遍墙钟、全部 warnings（去重）、每遍每条的 raw_text 与 token id |
| `probe_out/full_S3_warnings.json` | full_S3 全程的 warnings（去重） |
| `0025_full_S1/`、`0025_full_S3/` | 两次完整开发子集验收的 run_e1 输出，目录名按 run_e1 的运行编号规则命名 |
| `det_summary.md`、`det_summary.json` | 由 `probe_scripts/p27_det_summary.py` 生成：各设置、跨进程、与 RUN-0024 的逐条对照、完整开发子集、warnings |
| `logs/` | 各步的日志、`driver.log`、`worktree.txt`；`det_gpu.csv`、`det_apps.csv` 是 nvidia-smi 采样 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 磁盘收支；dry-run 输出（没有 "Would install"） |
| `probe_scripts/` | `run_probe_25.sh`、`rerun_batch.py`、`run_e1_det.py`、`p27_det_summary.py`、`gpu_sampler.sh`、`diskcheck25.sh` |
