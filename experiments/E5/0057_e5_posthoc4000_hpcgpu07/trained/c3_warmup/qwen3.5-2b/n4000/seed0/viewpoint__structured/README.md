# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 0 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 85f41063b47214a5…；起点 c3_warmup（manifest dca6a89d9a3e920c…）；训练运行身份 f208ccb9ae2c34ce…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.2146（95% CI [1.1607, 1.2732]），有效率 0.9714，条件 1.1915（中位数 1.1528），n = 315
- mask IoU（网格）：严格 0.4371（95% CI [0.4139, 0.4614]），有效率 1.0000，条件 0.4371（中位数 0.4631），n = 315
- mask IoU（全分辨率）：严格 0.4350（95% CI [0.4122, 0.4587]），有效率 1.0000，条件 0.4350（中位数 0.4561），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.4371 vs 最强非真值基线 translate 0.2630；差 0.1741，CI [0.1469, 0.2009] → pass=True
- 判定条件：evaluated，系统起伏误差 1.2146，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.0763，CI [0.0078, 0.1453] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.4442，CI [0.3805, 0.5097] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.0743，CI [0.0038, 0.1437] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
