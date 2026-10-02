# E5 · baseline · copy · rotation3d（a）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 cd451c7bac7ccbe5…）；生成器任务版本 4
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.3376（95% CI [1.0138, 1.7011]），有效率 1.0000，条件 1.3376（中位数 1.2003），n = 12
- mask IoU（网格）：严格 0.7111（95% CI [0.6335, 0.7832]），有效率 1.0000，条件 0.7111（中位数 0.7524），n = 12
- mask IoU（全分辨率）：严格 0.7033（95% CI [0.6235, 0.7761]），有效率 1.0000，条件 0.7033（中位数 0.7338），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.7111 vs 最强非真值基线 copy 0.7111 → 不通过
- 判定条件：evaluated，系统 1.3376，gt_affine_upper 1.2007，copy_flat 0.9858 → pass=False，pass_vs_flat=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/16）；分原因：{}

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
