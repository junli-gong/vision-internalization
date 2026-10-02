# E5 · baseline · copy · viewpoint（b）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 fc8d41eb198c180a…）；生成器任务版本 4
- 标题口径：环绕移动（全部档）

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2970（95% CI [0.1444, 0.4707]），有效率 1.0000，条件 0.2970（中位数 0.1587），n = 9
- mask IoU（全分辨率）：严格 0.2957（95% CI [0.1451, 0.4661]），有效率 1.0000，条件 0.2957（中位数 0.1657），n = 9

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2970 vs 最强非真值基线 translate 0.2979 → 不通过
- 判定条件：unavailable
- imagining：None

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
