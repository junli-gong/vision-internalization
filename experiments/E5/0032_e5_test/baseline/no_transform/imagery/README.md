# E5 · baseline · no_transform · imagery（e）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 f94cb7b097c8d72b…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2146（95% CI [0.1923, 0.2377]），有效率 1.0000，条件 0.2146（中位数 0.1304），n = 400
- mask IoU（全分辨率）：严格 0.2114（95% CI [0.1897, 0.2333]），有效率 1.0000，条件 0.2114（中位数 0.1323），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2146（中位数 0.1304）vs 最强非真值基线 no_transform 0.2146（中位数 0.1304）；差 0.0000，CI [0.0000, 0.0000]，逐样本差的中位数 0.0000 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
