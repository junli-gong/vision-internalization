# E5 · 训练后（e3b → E5）· qwen3.5-2b · n4000 · seed 2 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest ec3409b1ae6b94e6…；起点 e3b（manifest a4073791af6478a6…）；训练运行身份 bb513e754e5776c2…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.9839（95% CI [1.9671, 1.9979]），有效率 0.0159，条件 0.9834（中位数 0.9503），n = 315
- mask IoU（网格）：严格 0.0079（95% CI [0.0017, 0.0156]），有效率 1.0000，条件 0.0079（中位数 0.0000），n = 315
- mask IoU（全分辨率）：严格 0.0077（95% CI [0.0017, 0.0153]），有效率 1.0000，条件 0.0077（中位数 0.0000），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0079 vs 最强非真值基线 translate 0.2630；差 -0.2552，CI [-0.2801, -0.2311] → pass=False
- 判定条件：evaluated，系统起伏误差 1.9839，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.8456，CI [0.7964, 0.8900] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 1.2134，CI [1.1590, 1.2664] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.8436，CI [0.7952, 0.8876] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
