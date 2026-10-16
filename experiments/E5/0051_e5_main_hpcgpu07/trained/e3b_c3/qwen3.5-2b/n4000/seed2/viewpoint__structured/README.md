# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n4000 · seed 2 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest fa8699f4866e5ebd…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 181e2256a0658d0a…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.9869（95% CI [1.9728, 1.9975]），有效率 0.0286，条件 1.5420（中位数 1.4821），n = 315
- mask IoU（网格）：严格 0.0121（95% CI [0.0049, 0.0210]），有效率 1.0000，条件 0.0121（中位数 0.0000），n = 315
- mask IoU（全分辨率）：严格 0.0120（95% CI [0.0049, 0.0208]），有效率 1.0000，条件 0.0120（中位数 0.0000），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0121 vs 最强非真值基线 translate 0.2630；差 -0.2509，CI [-0.2755, -0.2274] → pass=False
- 判定条件：evaluated，系统起伏误差 1.9869，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.8486，CI [0.8013, 0.8903] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 1.2165，CI [1.1599, 1.2726] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.8466，CI [0.8008, 0.8894] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
