# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n1000 · seed 2 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 17a7e4ccdbafc9d0…；起点 c3_warmup（manifest 1ef34cdf543d80fc…）；训练运行身份 14753a441ea8f515…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.6146（95% CI [1.5454, 1.6952]），有效率 0.8095，条件 1.5239（中位数 1.4817），n = 315
- mask IoU（网格）：严格 0.1855（95% CI [0.1672, 0.2055]），有效率 1.0000，条件 0.1855（中位数 0.1414），n = 315
- mask IoU（全分辨率）：严格 0.1849（95% CI [0.1671, 0.2045]），有效率 1.0000，条件 0.1849（中位数 0.1401），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.1855 vs 最强非真值基线 translate 0.2630；差 -0.0776，CI [-0.1064, -0.0485] → pass=False
- 判定条件：evaluated，系统起伏误差 1.6146，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.4763，CI [0.3947, 0.5628] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.8441，CI [0.7644, 0.9279] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.4742，CI [0.3927, 0.5581] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
