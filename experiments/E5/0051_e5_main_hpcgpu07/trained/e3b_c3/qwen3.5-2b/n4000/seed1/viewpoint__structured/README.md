# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n4000 · seed 1 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 6e6aebc2a23df08a…；起点 e3b_c3（manifest 7c72a5007e461cca…）；训练运行身份 aea71d1028b8ccc2…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.9068（95% CI [1.8695, 1.9465]），有效率 0.1841，条件 1.4941（中位数 1.4921），n = 315
- mask IoU（网格）：严格 0.0658（95% CI [0.0473, 0.0835]），有效率 1.0000，条件 0.0658（中位数 0.0000），n = 315
- mask IoU（全分辨率）：严格 0.0655（95% CI [0.0471, 0.0830]），有效率 1.0000，条件 0.0655（中位数 0.0000），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0658 vs 最强非真值基线 translate 0.2630；差 -0.1972，CI [-0.2251, -0.1725] → pass=False
- 判定条件：evaluated，系统起伏误差 1.9068，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.7686，CI [0.7146, 0.8247] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 1.1364，CI [1.0762, 1.1976] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.7665，CI [0.7119, 0.8247] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
