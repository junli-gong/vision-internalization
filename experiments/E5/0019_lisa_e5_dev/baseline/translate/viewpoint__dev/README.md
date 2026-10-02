# E5 · baseline · translate · viewpoint（b）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 fc8d41eb198c180a…）；生成器任务版本 4
- 标题口径：环绕移动（全部档）

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2979（95% CI [0.1561, 0.4659]），有效率 1.0000，条件 0.2979（中位数 0.1765），n = 9
- mask IoU（全分辨率）：严格 0.2953（95% CI [0.1571, 0.4534]），有效率 1.0000，条件 0.2953（中位数 0.1696），n = 9

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2979 vs 最强非真值基线 translate 0.2979 → 不通过
- 判定条件：unavailable
- imagining：None

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
