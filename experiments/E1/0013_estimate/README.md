# experiments/E1/0013_estimate — RUN-0013 §4：吞吐与算力估算（新的生成上限）

服务器 CC，b200，2026-10-01 21:14–23:02 UTC。运行报告：`reports/0013-e1-accept-estimate-v2.md`。

## 做了什么

驱动脚本是 `probe_scripts/run_estimate13.sh`，由 `experiments/E1/0013_acceptance/probe_scripts/chain13.sh` 在验收结束后启动。每个模型执行：

```bash
python scripts/run_e1.py --estimate --model <键> --task all --batch-sizes 8 16 32 --estimate-think --out experiments/E1/0013_estimate
```

附加口径另跑，只测 `--task refexp_box ovd`，其余参数相同：

- qwen3.5-0.8b、qwen3.5-9b 加 `--policy qwen:1024`；
- internvl3.5-2b-hf、internvl3.5-4b-hf 加 `--policy internvl:native`。

环境与验收相同：先 `source $PROJ/scripts/env.sh`，再 export `VISINT_DATA_ROOT=$PROJ/data`、`NLTK_DATA=$PROJ/nltk_data`、`TMPDIR=/dev/shm/visint/tmp`。

## 完成范围（任务书 §5：累计超过 8 小时，跑完当前模型就停）

| 模型 | 主口径 | 附加口径 |
|---|---|---|
| qwen3.5-0.8b | 完成 | qwen:1024 完成 |
| qwen3.5-2b | 完成 | — |
| gemma-4-e2b-it | 完成 | — |
| internvl3.5-2b-hf | 完成 | internvl:native 完成 |
| qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf、qwen3.5-9b | **未开始** | qwen3.5-9b 的 qwen:1024、internvl3.5-4b-hf 的 internvl:native 也未开始 |

internvl3.5-2b-hf 跑完时，累计已经 29,492 s，超过了 28,800 s，所以 qwen3.5-4b 没有启动。完成和未完成的模型与 RUN-0012 相同。

## 文件

| 路径 | 内容 |
|---|---|
| `<键>/estimate/estimate.json`、`estimate.md` | 主口径的估算：每个任务在每个批大小上的吞吐、推荐批大小、各子集的 n 与外推的 GPU 小时；开思考的实测 |
| `<键>/estimate__qwen-1024/`、`<键>/estimate__internvl-native/` | 附加口径的估算，结构同上 |
| `estimate_summary.md`、`estimate_summary.json` | 由 `probe_scripts/p16_estimate_summary.py` 从 estimate.json 生成，含五张表（见下）。数字只从 JSON 读，不重算外推 |
| `logs/estimate_<键>[__<口径>].log` | 每次 run_e1 的 stdout/stderr，含 `time` 的输出 |
| `logs/driver.log` | 驱动日志，验收与估算共用 |
| `logs/chain_gpu.csv`、`logs/chain_apps.csv` | 每 5 s 一次的 nvidia-smi 采样，覆盖整个 RUN-0013（验收与估算） |
| `disk_ledger.tsv` | 本运行的磁盘收支，验收与估算共用，单位字节。三列依次是：根盘可用、`/dev/shm/visint` 占用、`$PROJ` 占用 |
| `probe_scripts/` | `run_estimate13.sh`；`p16_estimate_summary.py`，从 RUN-0012 原样复用；`diskcheck13.sh` |

`estimate_summary.md` 的五张表：

1. 主口径的估算汇总；
2. 每个模型的合计；
3. 附加口径；
4. 逐批大小明细；
5. 开思考的实测。

## 口径（取自 `src/e1/estimate.py`，原样记录）

- 墙钟只含 `model.generate`，预热不计。
- 推荐批大小：在可行的批大小里，取样本/秒最高的。不可行指 OOM，或峰值显存超过总显存的 90%。
- GPU 小时：等于 n ÷ 样本每秒。
- 开思考：
  - 用 min(推荐批大小, 8) 的批大小、预算 4096 token，在 8 条开发样本上实测；
  - think8192 的值是按"每条耗时 × 8192/4096"算出的上界，没有实测。
- 附加口径：只外推 extra 子集。
