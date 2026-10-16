# E5 · 训练后（base_warmup → E5）· qwen3.5-2b · n1000 · seed 1 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest c79a75d5930793d0…；起点 base_warmup（manifest e13b53a97ceec206…）；训练运行身份 9c4a6847a9627865…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.6536（95% CI [1.5162, 1.8180]），有效率 1.0000，条件 1.6536（中位数 1.3221），n = 300
- mask IoU（网格）：严格 0.4258（95% CI [0.4099, 0.4420]），有效率 1.0000，条件 0.4258（中位数 0.4155），n = 300
- mask IoU（全分辨率）：严格 0.4224（95% CI [0.4060, 0.4383]），有效率 1.0000，条件 0.4224（中位数 0.4097），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.4258 vs 最强非真值基线 copy 0.6878；差 -0.2620，CI [-0.2836, -0.2408] → pass=False
- 判定条件：evaluated，系统起伏误差 1.6536，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.6521，CI [0.5174, 0.8140] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.4176，CI [0.3228, 0.5248] → 胜过=False
  - vs copy_convex：基线 0.9363；差 0.7173，CI [0.5848, 0.8755] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
