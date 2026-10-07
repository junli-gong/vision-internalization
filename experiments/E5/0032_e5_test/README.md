# experiments/E5/0032_e5_test — RUN-0032：E5 test 划分（零样本基座与文本思维链 7/9、LISA-7B-v1 对照、平凡基线）

服务器 CC，b200，2026-10-06 00:27 – 23:56 UTC。运行报告：`reports/0032-e5-test.md`。

## 运行用的代码

- 驱动 `probe_scripts/run_e5_32.sh` 在一个固定于任务开始时 HEAD `44ef274` 的 git worktree 里运行（`/dev/shm/visint/tmp/wt0032`），跑完已删除。
- 前置校验：
  - `configs/e5_prompts.json` 的 sha256 是 `c0e59d35…`；
  - `B_HEADLINE_BINS = (1, 2, 3)`；
  - LISA clone 的 HEAD 是 `3cb2d43…`，没有改动。
  - 以上见 `logs/worktree.txt`。`synth_pin_digests.py check` 300/300 相同，见 `logs/digests.log`。
- 输出用绝对路径写进本目录。

## 做了什么（任务书 §3；GPU 时间按 15 h 上限记账）

1. **平凡基线（CPU）：** `run_e5.py --system baseline --task a c d e` 与 `--task b --n-per-bin 200`。
2. **模型。** 按优先级依次跑；每次开始新项目之前检查累计 GPU 时间是否超过 15 h。每个模型执行：

   ```bash
   python scripts/run_e5.py --system zeroshot cot --model <键> --task a c d e --out <本目录>
   python scripts/run_e5.py --system zeroshot cot --model <键> --task b --n-per-bin 200 --out <本目录>
   ```

   - 完成的有 qwen3.5-2b、LISA、qwen3.5-0.8b、qwen3.5-4b、qwen3.5-9b、internvl3.5-2b-hf、internvl3.5-4b-hf。
   - 累计 GPU 时间在 internvl3.5-4b-hf 之后是 83,953 s，超过 15 h，驱动在 gemma-4-e2b-it 之前停下。两个 Gemma 没有跑。
3. **LISA**（RUN-0019 的四步，test 划分）：
   1. 导出：`--export-inputs --task a b c d --subset eval`，另导一份 `--task b --n-per-bin 200`。
   2. 彩排：`--limit 5`，并生成 `overlays/`。
   3. 全量：两份导出各跑一次。
   4. 评分：`--system external --model lisa-7b-v1`。a、c、d 用第一份（每档 100 条）；b 用第二份（每档 200 条，`--n-per-bin 200`）。
   - 导出目录（`$PROJ/tmp`）用完已删除。删除前的体积：7,575,081 B 共 1,504 个文件；6,016,257 B 共 801 个文件。

## 隔离（硬约束 1）

- LISA 的输出只在本目录的三个外部预测目录里：`lisa_predictions/`、`lisa_predictions_b200/`、`lisa_predictions_limit5/`。
- 评分由 `run_e5.py --system external` 写进 `external/`；`overlays/` 是由彩排输出画出的目检图。
- 没有写 `$PROJ/data`，也没有写任何训练或数据目录。

## 文件

| 路径 | 内容 |
|---|---|
| `zeroshot/<模型>/<任务>/`、`cot/<模型>/<任务>/` | run_e5 的输出：`predictions.jsonl`、`metrics.json`（headline、overall、by_bin、by_group、parse、truncation、thinking_end、gate、baselines_headline）、`compute.json`、`config.yaml`、`meta.json`、`README.md`、`logs/` |
| `external/lisa-7b-v1/<任务>/` | LISA 的评分输出（含对"当前 mask"的 IoU，D59-5） |
| `baseline/<基线>/<任务>/` | 平凡基线与参照线 |
| `lisa_predictions*/`、`overlays/`、`weights_download.json` | 外部预测目录、彩排目检图、LISA 与 CLIP 权重的逐文件核对 |
| `e5_summary.md`、`e5_summary.json` | 由 `probe_scripts/p31_e5_summary.py` 从各 metrics.json 原样取出的表：逐模型 × 系统 × 任务、门槛、LISA、平凡基线、用量 |
| `logs/` | 每步日志 `<项目>__<步骤>.log`、`driver.log`、`worktree.txt`、`digests.log`；`e5test_gpu.csv`、`e5test_apps.csv` 是每 5 s 一次的 nvidia-smi 采样 |
| `disk_ledger.tsv`、`pip_dryrun.txt` | 磁盘收支；dry-run 输出（没有 "Would install"） |
| `probe_scripts/` | `run_e5_32.sh`、`push_item.sh`、`prep_push.py`、`p31_e5_summary.py`、`p23_overlays.py`、`hold_before_next.sh`（没有触发，见报告 §4）、`gpu_sampler.sh`、`diskcheck32.sh` |

- 超过 20 MB 的文件只有 4 个思维链 b 类的 `predictions.jsonl`：`cot/{qwen3.5-0.8b,qwen3.5-4b,internvl3.5-2b-hf,internvl3.5-4b-hf}/viewpoint/`。入库的是 `.gz`，`LARGE_FILES.sha256` 记原文件与 gz 的 sha256 和字节数，原文件留在服务器上。
