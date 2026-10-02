# E5 · external · lisa-7b-v1 · rotation3d（a）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 cd451c7bac7ccbe5…）；生成器任务版本 4
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.6766（95% CI [0.5762, 0.7691]），有效率 1.0000，条件 0.6766（中位数 0.7481），n = 12
- mask IoU（全分辨率）：严格 0.6679（95% CI [0.5719, 0.7600]），有效率 1.0000，条件 0.6679（中位数 0.7323），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.6766 vs 最强非真值基线 copy 0.7111 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
