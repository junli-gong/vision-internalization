# E5 · 训练后（e3b_c3 → E5）· qwen3.5-2b · n1000 · seed 0 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest b80f19148fa05130…；起点 e3b_c3（manifest bf73188b8c88c760…）；训练运行身份 45f67904502944e9…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.4311（95% CI [1.3756, 1.4869]），有效率 0.8508，条件 1.3313（中位数 1.2998），n = 315
- mask IoU（网格）：严格 0.2688（95% CI [0.2491, 0.2901]），有效率 1.0000，条件 0.2688（中位数 0.2486），n = 315
- mask IoU（全分辨率）：严格 0.2680（95% CI [0.2488, 0.2892]），有效率 1.0000，条件 0.2680（中位数 0.2471），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2688 vs 最强非真值基线 translate 0.2630；差 0.0057，CI [-0.0156, 0.0271] → pass=False
- 判定条件：evaluated，系统起伏误差 1.4311，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.2928，CI [0.2294, 0.3563] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.6607，CI [0.5858, 0.7308] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.2908，CI [0.2293, 0.3533] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
