# E5 · external · lisa-7b-v1 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…）；生成器任务版本 5
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）
- 凸起先验基线 copy_convex：A = 1 档（val 上选定后冻结，src/synth/relief.py convex_prior_codes（D59-4，DEV-0028））

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2132（95% CI [0.1923, 0.2359]），有效率 1.0000，条件 0.2132（中位数 0.1634），n = 315
- mask IoU（全分辨率）：严格 0.2124（95% CI [0.1915, 0.2348]），有效率 1.0000，条件 0.2124（中位数 0.1661），n = 315
- 对当前 mask 的 IoU（网格，D59-5）：严格 0.6878（95% CI [0.6532, 0.7224]），有效率 1.0000，条件 0.6878（中位数 0.7928），n = 315
- 对当前 mask 的 IoU（全分辨率，D59-5）：严格 0.6885（95% CI [0.6544, 0.7229]），有效率 1.0000，条件 0.6885（中位数 0.7819），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2132（中位数 0.1634）vs 最强非真值基线 translate 0.2630（中位数 0.2370）；差 -0.0498，CI [-0.0662, -0.0334]，逐样本差的中位数 -0.0138 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.0013（1/800）；分原因：{'external_failure': 1}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
