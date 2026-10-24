# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n4000 · seed 2 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest d27c641a10cecd88…；起点 c3_warmup（manifest 1ef34cdf543d80fc…）；训练运行身份 2271e52ec42558cd…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.2172（95% CI [1.1618, 1.2694]），有效率 0.9651，条件 1.1888（中位数 1.0998），n = 315
- mask IoU（网格）：严格 0.4251（95% CI [0.4018, 0.4483]），有效率 1.0000，条件 0.4251（中位数 0.4545），n = 315
- mask IoU（全分辨率）：严格 0.4233（95% CI [0.3997, 0.4466]），有效率 1.0000，条件 0.4233（中位数 0.4531），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.4251 vs 最强非真值基线 translate 0.2630；差 0.1621，CI [0.1359, 0.1884] → pass=True
- 判定条件：evaluated，系统起伏误差 1.2172，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.0789，CI [0.0129, 0.1436] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.4467，CI [0.3816, 0.5174] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.0769，CI [0.0102, 0.1423] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
