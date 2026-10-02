# E5 · baseline · visible · occlusion（c）· dev

- 样本：val 划分，每档 4 条，n = 12（id 指纹 ea15bc3fe20152d0…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.5861（95% CI [0.4999, 0.6728]），有效率 1.0000，条件 0.5861（中位数 0.5785），n = 12
- mask IoU（全分辨率）：严格 0.5866（95% CI [0.5027, 0.6715]），有效率 1.0000，条件 0.5866（中位数 0.5852），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.5861 vs 最强非真值基线 convex_hull 0.7138 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/12）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
