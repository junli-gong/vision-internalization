# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n4000 · seed 0 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 9a021e6e0f255ed2…；起点 e3b_c3（manifest bf73188b8c88c760…）；训练运行身份 bfcd7a5e2cf97769…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.2723（95% CI [1.2205, 1.3230]），有效率 0.9460，条件 1.2308（中位数 1.2155），n = 315
- mask IoU（网格）：严格 0.4076（95% CI [0.3817, 0.4304]），有效率 1.0000，条件 0.4076（中位数 0.4253），n = 315
- mask IoU（全分辨率）：严格 0.4056（95% CI [0.3799, 0.4285]），有效率 1.0000，条件 0.4056（中位数 0.4252），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.4076 vs 最强非真值基线 translate 0.2630；差 0.1445，CI [0.1169, 0.1712] → pass=True
- 判定条件：evaluated，系统起伏误差 1.2723，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.1340，CI [0.0684, 0.2001] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.5019，CI [0.4333, 0.5650] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.1320，CI [0.0660, 0.1968] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
