# experiments/P0/0011_health_v2_base — RUN-0011 证据

服务器 CC，b200，2026-10-01。用健康检查协议 v2（D28-7、D31）在 8 个基座上重跑，并与 v1 比较。v2 的规则：MMLU-Pro 生成到 256 token；32 token 的读数 `mmlu_pro_fmt32` 从同一次生成里截出；加了 `--audit-fmt32 100`。v1 的参照是 RUN-0009 重评分后的 JSON。运行报告：`reports/0011-health-v2-base.md`。

| 文件 | 内容 |
|---|---|
| `health_<键>.json` ×8 | `python scripts/health_check.py --model <键> --device cuda --dtype bf16 --audit-fmt32 100 --out …` 的输出，协议 v2。`prefix_audit` 字段是审计结果 |
| `compare_v1_v2_<键>.json` ×8 | `python scripts/health_check.py --compare experiments/P0/0009_ifeval_rescore/health_<键>.json <本目录>/health_<键>.json --out …` 的输出 |
| `health_summary_v2.md` | 由 `probe_scripts/p14_v2_summary.py` 从上面的 JSON 生成的表：分数、逐题核对、审计、截断率、计算量、解析失败 |
| `summary_v2.json` | 同一脚本输出的机器可读版，含逐题核对中有差异的 id 与字段（本次全部为空） |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出（没有 "Would install"） |
| `disk_ledger.tsv` | 磁盘收支，单位字节。三列依次是：根盘可用、`/dev/shm/visint` 占用、`$PROJ` 占用 |
| `logs/health_<键>.log` | 每个模型健康检查的 stdout/stderr，含 `time` 的输出 |
| `logs/compare_<键>.log` | 每次比较的 stdout/stderr |
| `logs/driver.log` | 驱动脚本的日志：权重核对、退出码、每个模型的删除与释放量 |
| `logs/gpu_batch2.csv`、`logs/gpu_apps_batch2.csv` | 第二批（后 4 个模型）每 5 s 一次的 nvidia-smi 采样，分别记整卡（显存、利用率）和按进程（显存）。第一批没有采样 |
| `probe_scripts/` | `run_health_v2.sh`（驱动）、`p14_v2_summary.py`（汇总与逐题核对）、`diskcheck11.sh` |

## 逐题核对的口径（`p14_v2_summary.py`）

- 按 `id` 配对，比较两边记录**共有的字段**，不比 `wall_time_s`。
- v2 记录多了一个 `stopped` 字段，v1 没有，所以不参与比较，在表里单列为"仅一侧有的字段"。
- 三组对照：
  - GSM8K、IFEval：v2 对 RUN-0009 的 v1 记录；
  - `mmlu_pro_fmt32`：v2 截出的记录对 RUN-0007 的 `mmlu_pro` 记录（32 token）。

## 复现

```bash
source $PROJ/scripts/env.sh
export NLTK_DATA=$PROJ/nltk_data
cd $REPO
bash experiments/P0/0011_health_v2_base/probe_scripts/run_health_v2.sh <模型键> [...]
python experiments/P0/0011_health_v2_base/probe_scripts/p14_v2_summary.py experiments/P0/0011_health_v2_base \
  experiments/P0/0009_ifeval_rescore experiments/P0/0007_health_base <out.json> > health_summary_v2.md
```

驱动脚本按 RUN-0009 的 `weights_manifest.json` 下载并核对权重（`p12_verify_weights.py --download`）。每个模型跑完后，用 `scan_cache_dir(...).delete_revisions(<revision>)` 把它从内存盘删掉；只保留 qwen3.5-0.8b。
