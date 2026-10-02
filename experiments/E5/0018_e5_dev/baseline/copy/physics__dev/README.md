# E5 · baseline · copy · physics（d）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 477e636d2c1074b0…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.3807（95% CI [0.2402, 0.5207]），有效率 1.0000，条件 0.3807（中位数 0.4129），n = 16
- mask IoU（全分辨率）：严格 0.3832（95% CI [0.2549, 0.5161]），有效率 1.0000，条件 0.3832（中位数 0.3717），n = 16

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.3807 vs 最强非真值基线 copy 0.3807 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
