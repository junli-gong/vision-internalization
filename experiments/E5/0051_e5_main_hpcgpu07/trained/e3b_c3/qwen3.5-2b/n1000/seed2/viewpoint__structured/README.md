# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n1000 · seed 2 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest 18d0fdc3d8341fea…；起点 e3b_c3（manifest aaacbbab3e2f2ef3…）；训练运行身份 4dde32f7428a15cd…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.3983（95% CI [1.3417, 1.4553]），有效率 0.8667，条件 1.3057（中位数 1.2813），n = 315
- mask IoU（网格）：严格 0.2830（95% CI [0.2586, 0.3066]），有效率 1.0000，条件 0.2830（中位数 0.2508），n = 315
- mask IoU（全分辨率）：严格 0.2816（95% CI [0.2574, 0.3051]），有效率 1.0000，条件 0.2816（中位数 0.2437），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2830 vs 最强非真值基线 translate 0.2630；差 0.0200，CI [-0.0014, 0.0414] → pass=False
- 判定条件：evaluated，系统起伏误差 1.3983，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.2600，CI [0.1995, 0.3225] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.6278，CI [0.5570, 0.6981] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.2580，CI [0.1987, 0.3194] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
