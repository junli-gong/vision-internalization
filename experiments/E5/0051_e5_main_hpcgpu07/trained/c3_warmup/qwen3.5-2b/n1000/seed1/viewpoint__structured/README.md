# E5 · 训练后（c3_warmup → E5）· qwen3.5-2b · n1000 · seed 1 · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…，与零样本相同的清单）；生成器任务版本 5
- 检查点：E5 manifest a7a2eac74cfbd2c2…；起点 c3_warmup（manifest e547056affbd7f0b…）；训练运行身份 67c0d40b563d0d7b…
- 解码：结构强制（D83-3，v3 的正式口径；relief，每个格子位置只允许 33 个值 token，行末只允许行分隔，最后只允许停止 token，长度恒为 1057；generate_batch + StructuredGridProcessor），单轮，关思考，贪心，批大小 64
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 1.5299（95% CI [1.4626, 1.5952]），有效率 0.7968，条件 1.4101（中位数 1.3985），n = 315
- mask IoU（网格）：严格 0.2297（95% CI [0.2073, 0.2531]），有效率 1.0000，条件 0.2297（中位数 0.1983），n = 315
- mask IoU（全分辨率）：严格 0.2290（95% CI [0.2067, 0.2523]），有效率 1.0000，条件 0.2290（中位数 0.2017），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.2297 vs 最强非真值基线 translate 0.2630；差 -0.0333，CI [-0.0585, -0.0093] → pass=False
- 判定条件：evaluated，系统起伏误差 1.5299，n = 315 → pass=False
  - vs copy_flat：基线 1.1383；差 0.3916，CI [0.3162, 0.4632] → 胜过=False
  - vs gt_affine_upper：基线 0.7704；差 0.7595，CI [0.6737, 0.8369] → 胜过=False
  - vs copy_convex：基线 1.1403；差 0.3896，CI [0.3153, 0.4595] → 胜过=False
- imagining：False

## 解析

- 解析失败率：0.0000（0/800）；分原因：{}
- 截断率：0.0000

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
