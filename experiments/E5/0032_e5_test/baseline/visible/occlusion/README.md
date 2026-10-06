# E5 · baseline · visible · occlusion（c）· eval

- 样本：test 划分，每档 100 条，n = 300（id 指纹 d3c7f8190c0755e8…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.5895（95% CI [0.5707, 0.6101]），有效率 1.0000，条件 0.5895（中位数 0.6057），n = 300
- mask IoU（全分辨率）：严格 0.5964（95% CI [0.5781, 0.6157]），有效率 1.0000，条件 0.5964（中位数 0.6079），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5895（中位数 0.6057）vs 最强非真值基线 convex_hull 0.7081（中位数 0.7258）；差 -0.1187，CI [-0.1379, -0.0984]，逐样本差的中位数 -0.1173 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.0000（0/300）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
