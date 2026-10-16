# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n4000 · seed 2 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest fa8699f4866e5ebd…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 181e2256a0658d0a…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.0018（95% CI [1.9995, 2.0062]），有效率 0.0067，条件 2.2745（中位数 2.2745），n = 300
- mask IoU（网格）：严格 0.0040（95% CI [0.0000, 0.0119]），有效率 1.0000，条件 0.0040（中位数 0.0000），n = 300
- mask IoU（全分辨率）：严格 0.0040（95% CI [0.0000, 0.0118]），有效率 1.0000，条件 0.0040（中位数 0.0000），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0040 vs 最强非真值基线 copy 0.6878；差 -0.6838，CI [-0.7012, -0.6659] → pass=False
- 判定条件：evaluated，系统起伏误差 2.0018，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 1.0003，CI [0.9874, 1.0117] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.7658，CI [0.6101, 0.9001] → 胜过=False
  - vs copy_convex：基线 0.9363；差 1.0655，CI [1.0473, 1.0825] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
