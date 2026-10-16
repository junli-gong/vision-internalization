# E5 · 训练后（e3b → E5）· qwen3.5-2b · n4000 · seed 2 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest ec3409b1ae6b94e6…；起点 e3b（manifest a4073791af6478a6…）；训练运行身份 bb513e754e5776c2…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.9902（95% CI [1.9747, 2.0009]），有效率 0.0133，条件 1.2622（中位数 1.1109），n = 300
- mask IoU（网格）：严格 0.0064（95% CI [0.0011, 0.0136]），有效率 1.0000，条件 0.0064（中位数 0.0000），n = 300
- mask IoU（全分辨率）：严格 0.0063（95% CI [0.0011, 0.0133]），有效率 1.0000，条件 0.0063（中位数 0.0000），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0064 vs 最强非真值基线 copy 0.6878；差 -0.6813，CI [-0.6989, -0.6636] → pass=False
- 判定条件：evaluated，系统起伏误差 1.9902，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.9887，CI [0.9708, 1.0032] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.7542，CI [0.5972, 0.8885] → 胜过=False
  - vs copy_convex：基线 0.9363；差 1.0538，CI [1.0310, 1.0737] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
