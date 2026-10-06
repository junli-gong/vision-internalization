# E5 · baseline · copy · physics（d）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 0ea63f1c30af02dc…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2906（95% CI [0.2651, 0.3157]），有效率 1.0000，条件 0.2906（中位数 0.2419），n = 400
- mask IoU（全分辨率）：严格 0.2939（95% CI [0.2684, 0.3185]），有效率 1.0000，条件 0.2939（中位数 0.2511），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2906（中位数 0.2419）vs 最强非真值基线 copy 0.2906（中位数 0.2419）；差 0.0000，CI [0.0000, 0.0000]，逐样本差的中位数 0.0000 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
