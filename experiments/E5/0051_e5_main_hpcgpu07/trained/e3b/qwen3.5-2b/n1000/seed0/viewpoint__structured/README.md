# E5 · 训练后（e3b → E5）· qwen3.5-2b · n1000 · seed 0 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest c9d40666daeec412…；起点 e3b（manifest acb7f58d1ae3aa0c…）；训练运行身份 3169f1982035ddfa…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.5449（95% CI [1.4894, 1.6061]），有效率 0.7016，条件 1.3514（中位数 1.3468），n = 315
- mask IoU（网格）：严格 0.1774（95% CI [0.1566, 0.1998]），有效率 1.0000，条件 0.1774（中位数 0.1073），n = 315
- mask IoU（全分辨率）：严格 0.1773（95% CI [0.1568, 0.1995]），有效率 1.0000，条件 0.1773（中位数 0.1081），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.1774 vs 最强非真值基线 translate 0.2630；差 -0.0857，CI [-0.1099, -0.0633] → pass=False
- 判定条件：evaluated，系统起伏误差 1.5449，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.4066，CI [0.3397, 0.4766] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.7745，CI [0.7036, 0.8470] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.4046，CI [0.3357, 0.4743] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
