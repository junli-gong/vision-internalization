# RUN-0099 报告：hpcmgt（CPU，只读），test 逐句元数据（U-6）与定性图（U-5）（D154、D158）

任务书 `prompts/server_hpc/0099-hpc-cpu-qual-meta.md`。不进 srun，不用 GPU，没碰 RUN-0097、RUN-0098。只贴数字，不解读。

## 运行

- 经 hpcmgt 上脱离会话的 `$VI/tmp/run0099.sh`（10-30 03:47:39Z 起，03:52:01Z 结束）。前 squeue：157296 RUNNING 75-15:13:25 hpcgpu06；157297 RUNNING 77-12:45:34 hpcgpu07。
- fetch 第 1 次 rc=0；wt0099（稀疏，另检出 `experiments/U/0099_qual_select/`）HEAD `0dce4efa7526168d9df3fe4dbc036bba431992fe`，status 空；6 个 sha256 与任务书逐字相同。
- 输入都在：`coco_train2014_e1dev/instances_train2014_e1dev.json`、RUN-0092 的 `x2_trace/` 与 `x2_spline_trimap/`、RUN-0072 的 `w2c/`。

## 旁路 venv（`$VI/plot/venv`，`--system-site-packages`）

- dry-run rc=0：`Requirement already satisfied: numpy>=1.25 in /home/mail/chenyuma/visint/env/lib/python3.12/site-packages (from matplotlib==3.11.2) (2.5.2)`；`Would install contourpy-1.4.0 cycler-0.12.1 fonttools-4.68.0 kiwisolver-1.5.1 matplotlib-3.11.2 pyparsing-3.3.3 python-dateutil-3.0.0 tzdata-2026.6`。不动 numpy，正式装 rc=0。
- `pip freeze --path $VI/plot/venv/lib/python3.12/site-packages`：

```
contourpy==1.4.0
cycler==0.12.1
fonttools==4.68.0
kiwisolver==1.5.1
matplotlib==3.11.2
pyparsing==3.3.3
python-dateutil==3.0.0
tzdata==2026.6
```

- venv 里 import 的 numpy 是 `$VI/env` 的 2.5.2，matplotlib 3.11.2。

## 元数据（`$VI/env` 的 python；rc=0，03:49:54Z–03:50:34Z）

stdout 原文：

```
[metadata] test_refcoco_unc_testA: 1000 句
[metadata] test_refcoco_unc_testB: 1000 句
[metadata] test_refcoco+_unc_testA: 1000 句
[metadata] test_refcoco+_unc_testB: 1000 句
[metadata] test_refcocog_umd_test: 1000 句
[check-preds] /home/mail/chenyuma/visint/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap test_refcoco_unc_testA: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap test_refcoco_unc_testB: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap test_refcoco+_unc_testA: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap test_refcoco+_unc_testB: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/X/0092_x2_test_hpcgpu06/qwen3.5-9b/refseg/x2_spline_trimap test_refcocog_umd_test: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c test_refcoco_unc_testA: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c test_refcoco_unc_testB: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c test_refcoco+_unc_testA: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c test_refcoco+_unc_testB: same
[check-preds] /home/mail/chenyuma/visint/repo/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c test_refcocog_umd_test: same
写出 /home/mail/chenyuma/visint/repo/experiments/U/0099_test_metadata/test_metadata.jsonl（5000 句，sha256 0819d5a0f50ae40b880db89ca92661e9c1951b9951e44ed722cee89199809681）与 /home/mail/chenyuma/visint/repo/experiments/U/0099_test_metadata/summary.json；c 的四分位 22.031117424874427 / 32.250701299469036 / 48.10754957514112；人 2221 句
```

`summary.json` 摘录：

- 句数 5,000（5 个划分各 1,000）。`n_zero_area` 0；`n_multi_component` 566；`n_components_ne_external_contours` 0。
- 大类句数：person 2221、animal 687、food 568、furniture 409、vehicle 343、kitchen 227、indoor 157、accessory 137、electronic 102、outdoor 60、appliance 49、sports 40（合计 5,000）。
- c 的分位数：0 → 13.843；0.05 → 16.174；0.25 → 22.031；0.5 → 32.251；0.75 → 48.108；0.95 → 83.673；1 → 461.224。
- 相对面积分位数：0 → 0.00299；0.25 → 0.0531；0.5 → 0.0869；0.75 → 0.1467；1 → 0.9756。
- 输出：`test_metadata.jsonl` 1,996,390 字节，`summary.json` 12,103 字节。

## 定性图（旁路 venv 的 python；rc=0，03:50:34Z–03:51:45Z；输出在仓库以外 `$VI/private/0099_qualitative`）

stdout 原文：

```
写出 /home/mail/chenyuma/visint/private/0099_qualitative/selection.json（25 条；正文 10 条）
  k0_random1     refcoco_unc/testA/33859      X-2 IoU 0.8156
  k0_random2     refcoco_unc/testA/16417      X-2 IoU 0.8365
  k0_random3     refcoco_unc/testA/38776      X-2 IoU 0.8710
  k0_failure1    refcoco_unc/testA/78182      X-2 IoU 0.0137
  k0_failure2    refcoco_unc/testA/89736      X-2 IoU 0.0264
  k1_random1     refcoco_unc/testB/39007      X-2 IoU 0.8598
  k1_random2     refcoco_unc/testB/76806      X-2 IoU 0.6963
  k1_random3     refcoco_unc/testB/20698      X-2 IoU 0.7361
  k1_failure1    refcoco_unc/testB/34544      X-2 IoU 0.0389
  k1_failure2    refcoco_unc/testB/129918     X-2 IoU 0.0954
  k2_random1     refcoco+_unc/testA/80504     X-2 IoU 0.7376
  k2_random2     refcoco+_unc/testA/132626    X-2 IoU 0.7290
  k2_random3     refcoco+_unc/testA/14877     X-2 IoU 0.5121
  k2_failure1    refcoco+_unc/testA/89410     X-2 IoU 0.0403
  k2_failure2    refcoco+_unc/testA/100945    X-2 IoU 0.1833
  k3_random1     refcoco+_unc/testB/114309    X-2 IoU 0.2965
  k3_random2     refcoco+_unc/testB/77177     X-2 IoU 0.8320
  k3_random3     refcoco+_unc/testB/45490     X-2 IoU 0.9531
  k3_failure1    refcoco+_unc/testB/34395     X-2 IoU 0.0000
  k3_failure2    refcoco+_unc/testB/37107     X-2 IoU 0.0000
  k4_random1     refcocog_umd/test/103049     X-2 IoU 0.6773
  k4_random2     refcocog_umd/test/59026      X-2 IoU 0.8340
  k4_random3     refcocog_umd/test/86648      X-2 IoU 0.8808
  k4_failure1    refcocog_umd/test/8489       X-2 IoU 0.0000
  k4_failure2    refcocog_umd/test/33194      X-2 IoU 0.0001
写出 6 张图与图注（/home/mail/chenyuma/visint/private/0099_qualitative）
```

- **`--expect-selection` 核对：通过**（`check_expected` 比较池与 25 条的 key、id、split、type、rank、main、also、iou；不同时脚本"什么都不写"并退出，这里写出了图，rc=0）。
- 拷进仓库的 `selection.json` 与已提交的 `experiments/U/0099_qual_select/selection.json` 字节不同，逐字段比较后只有两处：`numpy`（开发机 2.5.3，HPC 2.5.2）与 `inputs` 里的 `path`（开发机是相对路径，HPC 是绝对路径）。`entries`、`splits`（池）等其余字段相同。

图的清单（只在 HPC，不进 git，D28-9）：

| 文件 | 字节 | 页数 |
|---|---|---|
| `fig_qual_main.pdf` | 2,751,819 | 1 |
| `fig_qual_main.png` | 3,181,693 | — |
| `fig_qual_appendix_0_refcoco_unc_testA.pdf` | 5,896,315 | 1 |
| `fig_qual_appendix_0_refcoco_unc_testA.png` | 4,097,646 | — |
| `fig_qual_appendix_1_refcoco_unc_testB.pdf` | 4,265,197 | 1 |
| `fig_qual_appendix_1_refcoco_unc_testB.png` | 3,021,598 | — |
| `fig_qual_appendix_2_refcocoplus_unc_testA.pdf` | 4,819,295 | 1 |
| `fig_qual_appendix_2_refcocoplus_unc_testA.png` | 3,544,137 | — |
| `fig_qual_appendix_3_refcocoplus_unc_testB.pdf` | 4,179,851 | 1 |
| `fig_qual_appendix_3_refcocoplus_unc_testB.png` | 3,124,246 | — |
| `fig_qual_appendix_4_refcocog_umd_test.pdf` | 4,448,436 | 1 |
| `fig_qual_appendix_4_refcocog_umd_test.png` | 3,368,044 | — |

- 页数：`pdfinfo` 与按 `/Type /Page` 计数一致，都是 1。
- 进仓库的（`experiments/U/0099_qualitative_meta/`）：`selection.json`（51,094 字节）、`figures.json`（46,389 字节）、6 个 `*_caption.tex`（1,444–1,485 字节）。

## 推送

- 一次性同步 `ONCE=1 sync_push.sh … 0099 experiments/U/0099_test_metadata experiments/U/0099_qualitative_meta`，03:52:01Z ok（提交信息为同步脚本的固定格式 `[results] 0099: sync <时间>`）。图与 bundle 没有提交。
- 后 squeue（03:52:01Z）：157296 RUNNING 75-15:17:47 hpcgpu06；157297 RUNNING 77-12:49:56 hpcgpu07。
