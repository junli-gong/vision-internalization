# E5 · 训练后（base_warmup → E5）· qwen3.5-2b · n1000 · seed 1 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest c79a75d5930793d0…；起点 base_warmup（manifest e13b53a97ceec206…）；训练运行身份 9c4a6847a9627865…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.6252（95% CI [1.5619, 1.6904]），有效率 0.7492，条件 1.4998（中位数 1.4647），n = 315
- mask IoU（网格）：严格 0.1430（95% CI [0.1265, 0.1595]），有效率 1.0000，条件 0.1430（中位数 0.0916），n = 315
- mask IoU（全分辨率）：严格 0.1428（95% CI [0.1263, 0.1592]），有效率 1.0000，条件 0.1428（中位数 0.0937），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.1430 vs 最强非真值基线 translate 0.2630；差 -0.1200，CI [-0.1485, -0.0935] → pass=False
- 判定条件：evaluated，系统起伏误差 1.6252，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.4870，CI [0.4180, 0.5599] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.8548，CI [0.7772, 0.9339] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.4849，CI [0.4153, 0.5572] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
