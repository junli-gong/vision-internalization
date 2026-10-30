[RUN-TASK 0099] CPU，只读：test 的逐句元数据（预注册 U-6）与定性图（U-5）（D154、D158）。在登录节点上跑，不进 srun，不用 GPU；约 15 分钟

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 的 RUN-0097、07 的 RUN-0098。

## 为什么

- 负责人要定性图与按类别、形状复杂度的分档（D154），只用已有输出。规则写在预注册 U-5、U-6。
- 抽样清单已在开发机上算好、先提交：`experiments/U/0099_qual_select/selection.json`。HPC 上作图必须与它逐条相同（`--expect-selection`）。
- **D28-9**：带 COCO 原图的 PDF、PNG 与 bundle **只留在 HPC，不进 git**。仓库只收元数据、`selection.json`、`figures.json` 与图注。

## 文件（工作树 wt0099 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `scripts/analysis/test_metadata.py` | `aeb65aa879f4d5f44d6d6ad9fa2ffd07976c2deea889dbda8c2328c4cc9d249c` |
| `scripts/figures/make_qualitative.py` | `38c10cafb74d9b454ec7117fd932dd9364b927a8edff1bb5ed5de5cf8c30d5bf` |
| `scripts/figures/make_paper_figures.py` | `8bfd79d337e84cf5b8986c7dbdd650932e8c98f80e957a931b9c668c9a6a8ded` |
| `experiments/U/0099_qual_select/selection.json` | `b8af74e0dab773e10a1a45be430f0908eebe68787b99c7c0cc6258605f4d6041` |
| `configs/w2_llmseg_subsets.json` | `dfd53ce8e9a89d85ba828de6c0b3e331f83bfcf7d476cfc7ed1751e64c7497ae` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0099（稀疏），核对上表 6 个 sha256。
   - 确认 `$VI/b200/data/coco_train2014_e1dev/instances_train2014_e1dev.json` 在：元数据的大类取它的 categories 表。
   - 确认 RUN-0092 的 trace 在 `$VI/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_trace/`。
2. **旁路 venv，只为 matplotlib**（D158）：
   ```bash
   VI=/home/mail/chenyuma/visint
   $VI/env/bin/python -m venv --system-site-packages $VI/plot/venv
   $VI/plot/venv/bin/python -m pip install --no-cache-dir "matplotlib==3.11.2" --dry-run   # 不应动 numpy
   $VI/plot/venv/bin/python -m pip install --no-cache-dir "matplotlib==3.11.2"
   $VI/plot/venv/bin/python -m pip freeze --path $VI/plot/venv/lib/python3.12/site-packages   # 贴进报告
   ```
   - 如果 dry-run 要升级或降级 numpy，停下来报告，不装。
3. **元数据**（我们的环境，约 2–5 分钟）：
   ```bash
   cd $VI/wt/wt0099 && export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
   nice -n 10 $VI/env/bin/python scripts/analysis/test_metadata.py --data-root $VI/b200/data \
     --out $VI/repo/experiments/U/0099_test_metadata \
     --check-preds $VI/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap \
     --check-preds $VI/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c
   ```
   - 推回仓库：`experiments/U/0099_test_metadata/`（jsonl 约 1.5 MB 与 summary.json）。
4. **定性图**（旁路 venv，约 3–5 分钟；输出放仓库以外）：
   ```bash
   nice -n 10 $VI/plot/venv/bin/python scripts/figures/make_qualitative.py --data-root $VI/b200/data \
     --x2-run $VI/repo/experiments/X/0092_x2_test_hpcgpu06 --w2c-run $VI/repo/experiments/W2/0072_w2_test_hpcgpu06 \
     --expect-selection experiments/U/0099_qual_select/selection.json --out $VI/private/0099_qualitative
   ```
   - `$VI/private/` 不在仓库里。图与 bundle 都留在那里，不提交、不推送。
   - 只把 `selection.json`、`figures.json` 与 `*_caption.tex` 拷进 `$VI/repo/experiments/U/0099_qualitative_meta/`，再推回。
   - 报告里写清每张图的文件名、大小、页数。
5. 写 `reports/0099-hpc-cpu-qual-meta.md`，内容包括：
   - 两个命令的 stdout；
   - 元数据 summary.json 里的句数、各大类句数、c 的分位数；
   - `--expect-selection` 的核对结果；
   - 图的清单。
   最后再跑一次 `squeue`。
