# E5 · external · lisa-7b-v1 · physics（d）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 477e636d2c1074b0…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2442（95% CI [0.1116, 0.3905]），有效率 1.0000，条件 0.2442（中位数 0.0417），n = 16
- mask IoU（全分辨率）：严格 0.2388（95% CI [0.1153, 0.3744]），有效率 1.0000，条件 0.2388（中位数 0.0710），n = 16

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2442 vs 最强非真值基线 copy 0.3807 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
