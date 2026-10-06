# E5 · baseline · convex_hull · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.7081（95% CI [0.6893, 0.7265]），有效率 1.0000，条件 0.7081（中位数 0.7258），n = 300
- mask IoU（全分辨率）：严格 0.7087（95% CI [0.6904, 0.7269]），有效率 1.0000，条件 0.7087（中位数 0.7354），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.7081（中位数 0.7258）vs 最强非真值基线 convex_hull 0.7081（中位数 0.7258）；差 0.0000，CI [0.0000, 0.0000]，逐样本差的中位数 0.0000 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
