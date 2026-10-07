# E5 · zeroshot · internvl3.5-4b-hf · viewpoint（b）· eval

- 样本：test 划分，每档 200 条，n = 800（id 指纹 8ebef79045e7135b…）；生成器任务版本 5
- 标题口径：环绕移动，档 [1, 2, 3]（下限 ≥ 20°）
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4；格式 relief_image_view
- 凸起先验基线 copy_convex：A = 1 档（val 上选定后冻结，src/synth/relief.py convex_prior_codes（D59-4，DEV-0028））

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.0000（95% CI [2.0000, 2.0000]），有效率 0.0000，条件 null（中位数 null），n = 315
- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 315
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 315

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0000（中位数 0.0000）vs 最强非真值基线 translate 0.2630（中位数 0.2370）；差 -0.2630，CI [-0.2864, -0.2393]，逐样本差的中位数 -0.2370 → pass=False
- 判定条件：evaluated，系统起伏误差 2.0000（中位数 2.0000），n = 315 → pass=False
  - vs copy_flat：基线 1.1383（中位数 1.0203）；差 0.8617，CI [0.8157, 0.9025]，逐样本差的中位数 0.9797，系统更低的比例 0.0063 → 胜过=False
  - vs gt_affine_upper：基线 0.7704（中位数 0.6551）；差 1.2296，CI [1.1736, 1.2848]，逐样本差的中位数 1.3449，系统更低的比例 0.0381 → 胜过=False
  - vs copy_convex：基线 1.1403（中位数 1.0141）；差 0.8597，CI [0.8144, 0.9007]，逐样本差的中位数 0.9859，系统更低的比例 0.0063 → 胜过=False
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.9975（798/800）；分原因：{'truncated': 415, 'wrong_row_count': 383}
- 截断率：0.5188；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
