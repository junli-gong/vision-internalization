# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 0 · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…，与零样本相同的清单）；生成器任务版本 4
- 检查点：E5 manifest 85f41063b47214a5…；起点 c3_warmup（manifest dca6a89d9a3e920c…）；训练运行身份 f208ccb9ae2c34ce…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.5529（95% CI [1.4196, 1.7172]），有效率 1.0000，条件 1.5529（中位数 1.2161），n = 300
- mask IoU（网格）：严格 0.5711（95% CI [0.5547, 0.5872]），有效率 1.0000，条件 0.5711（中位数 0.5795），n = 300
- mask IoU（全分辨率）：严格 0.5643（95% CI [0.5486, 0.5799]），有效率 1.0000，条件 0.5643（中位数 0.5771），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.5711 vs 最强非真值基线 copy 0.6878；差 -0.1166，CI [-0.1332, -0.1011] → pass=False
- 判定条件：evaluated，系统起伏误差 1.5529，n = 300 → pass=False
  - vs copy_flat：基线 1.0015；差 0.5514，CI [0.4198, 0.7096] → 胜过=False
  - vs gt_affine_upper：基线 1.2360；差 0.3169，CI [0.2092, 0.4302] → 胜过=False
  - vs copy_convex：基线 0.9363；差 0.6166，CI [0.4890, 0.7713] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/400）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
