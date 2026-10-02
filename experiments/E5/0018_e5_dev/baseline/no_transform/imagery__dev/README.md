# E5 · baseline · no_transform · imagery（e）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 ec45558d08c54b9f…）；生成器任务版本 2
- 标题口径：全部样本

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2245（95% CI [0.1311, 0.3246]），有效率 1.0000，条件 0.2245（中位数 0.2474），n = 16
- mask IoU（全分辨率）：严格 0.2264（95% CI [0.1316, 0.3257]），有效率 1.0000，条件 0.2264（中位数 0.2339），n = 16

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2245 vs 最强非真值基线 no_transform 0.2245 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
