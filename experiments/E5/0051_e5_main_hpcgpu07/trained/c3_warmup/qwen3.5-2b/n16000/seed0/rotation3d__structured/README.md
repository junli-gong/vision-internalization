# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n16000 · seed 0 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest 3e78b64d04b4ad21…；起点 c3_warmup（manifest dca6a89d9a3e920c…）；训练运行身份 4b5a7b9d735677b5…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.0014（95% CI [2.0000, 2.0041]），有效率 0.0033，条件 2.4105（中位数 2.4105），n = 300
- mask IoU（网格）：严格 0.0019（95% CI [0.0000, 0.0056]），有效率 1.0000，条件 0.0019（中位数 0.0000），n = 300
- mask IoU（全分辨率）：严格 0.0019（95% CI [0.0000, 0.0056]），有效率 1.0000，条件 0.0019（中位数 0.0000），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0019 vs 最强非真值基线 copy 0.6878；差 -0.6859，CI [-0.7031, -0.6689] → pass=False
- 判定条件：evaluated，系统起伏误差 2.0014，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.9999，CI [0.9870, 1.0104] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.7654，CI [0.6085, 0.9004] → 胜过=False
  - vs copy_convex：基线 0.9363；差 1.0651，CI [1.0464, 1.0815] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
