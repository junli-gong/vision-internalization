# experiments/E1/0016_estimate — RUN-0016：E1 算力估算 v3（协议 v2 代码）

服务器 CC，b200，2026-10-02 00:50–05:44 UTC。运行报告：`reports/0016-e1-estimate-v3.md`。

## 做了什么

**1. 检测开发图标注的落地**（`python scripts/data/land_e1.py coco_train2014_e1dev --root $PROJ/data`）：

- 结果：1 个文件，2,870,001 B，核对 0 处不通过，rc=0；
- 清单副本：`coco_train2014_e1dev_MANIFEST.json`；
- 日志：`logs/land_coco_train2014_e1dev.log`。

**2. 8 个模型的估算**：驱动脚本 `probe_scripts/run_estimate16.sh`，先跑 4 个大模型，再跑 4 个小模型。每个模型执行：

```bash
python scripts/run_e1.py --estimate --model <键> --task all --batch-sizes 16 32 64 \
  --estimate-n 64 --estimate-think --estimate-think-n 64 --out experiments/E1/0016_estimate
```

**3. 附加口径**：只跑 `--task refexp_box ovd --batch-sizes 16 32 64`，不加 `--estimate-think`，紧跟在对应模型的主口径之后：

- qwen3.5-9b、qwen3.5-0.8b 加 `--policy qwen:1024`；
- internvl3.5-4b-hf、internvl3.5-2b-hf 加 `--policy internvl:native`。

**环境**：先 `source $PROJ/scripts/env.sh`，再 export：

- `VISINT_DATA_ROOT=$PROJ/data`
- `NLTK_DATA=$PROJ/nltk_data`
- `TMPDIR=/dev/shm/visint/tmp`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 文件

| 路径 | 内容 |
|---|---|
| `<键>/estimate/estimate.json`、`estimate.md` | 主口径：每个任务在 16、32、64 上的吞吐，外推用的批大小（`extrapolated_with`），各子集的 n 与外推的 GPU 小时，开思考与 free 子集的实测 |
| `<键>/estimate__qwen-1024/`、`<键>/estimate__internvl-native/` | 附加口径 |
| `estimate_summary.md`、`estimate_summary.json` | 由 `probe_scripts/p20_estimate_summary.py` 生成（在 RUN-0013 的 `p16_estimate_summary.py` 基础上改），表见下 |
| `coco_train2014_e1dev_MANIFEST.json` | 落地清单副本 |
| `logs/estimate_<键>[__<口径>].log` | 每次 run_e1 的 stdout/stderr，含 `time` 的输出 |
| `logs/driver.log` | 驱动日志：权重核对、退出码、删除与释放量、累计耗时 |
| `logs/est_gpu.csv`、`logs/est_apps.csv` | 每 5 s 一次的 nvidia-smi 采样，整卡与按进程各一份，覆盖整个估算 |
| `disk_ledger.tsv` | 磁盘收支，单位字节。三列依次是：根盘可用、`/dev/shm/visint` 占用、`$PROJ` 占用 |
| `pip_dryrun.txt` | `pip install --dry-run -r requirements.txt` 的输出（没有 "Would install"） |
| `probe_scripts/` | `run_estimate16.sh`、`p20_estimate_summary.py`、`gpu_sampler.sh`、`diskcheck16.sh` |

`estimate_summary.md` 里的表：

1. 估算汇总（逐模型 × 逐任务）；
2. 每个模型的合计；
3. 附加口径；
4. 开思考的实测；
5. free 子集的实测；
6. 同模型的思考吞吐对照（与 RUN-0013）；
7. nvidia-smi 峰值（与 RUN-0013）。

## 口径

- **GPU 小时的外推**：原样取自 `src/e1/estimate.py`。主评测按冻结的批大小 16 的实测外推，32 与 64 只测不外推。
- **思考的批大小**：min(推荐批大小, 64)，以 estimate.json 记录的为准。
- **nvidia-smi 峰值的归属**：用 `logs/driver.log` 里每个模型的起止时间（含附加口径），把 `*_apps.csv` 的样本归到模型，取 used_memory 的最大值。RUN-0013 的对照数字用同样的方法，从它已提交的日志与采样文件算出。
