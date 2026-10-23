# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n4000 · seed 2 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest 60a63d05b888137c…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 9b164c1419156de0…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.4207（95% CI [1.3026, 1.5749]），有效率 1.0000，条件 1.4207（中位数 1.1500），n = 300
- mask IoU（网格）：严格 0.5947（95% CI [0.5780, 0.6117]），有效率 1.0000，条件 0.5947（中位数 0.5956），n = 300
- mask IoU（全分辨率）：严格 0.5858（95% CI [0.5696, 0.6024]），有效率 1.0000，条件 0.5858（中位数 0.5897），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5947 vs 最强非真值基线 copy 0.6878；差 -0.0931，CI [-0.1088, -0.0780] → pass=False
- 判定条件：evaluated，系统起伏误差 1.4207，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.4192，CI [0.3048, 0.5668] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.1847，CI [0.0520, 0.3010] → 胜过=False
  - vs copy_convex：基线 0.9363；差 0.4844，CI [0.3704, 0.6260] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
