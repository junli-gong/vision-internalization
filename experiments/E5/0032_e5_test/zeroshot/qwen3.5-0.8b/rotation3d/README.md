# E5 · zeroshot · qwen3.5-0.8b · rotation3d（a）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 d69b689316eef31d…）；生成器任务版本 4
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4；格式 relief_image
- 凸起先验基线 copy_convex：A = 2 档（val 上选定后冻结，src/synth/relief.py convex_prior_codes（D59-4，DEV-0028））

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.0000（95% CI [2.0000, 2.0000]），有效率 0.0000，条件 null（中位数 null），n = 300
- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 300
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 300

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0000（中位数 0.0000）vs 最强非真值基线 copy 0.6878（中位数 0.7091）；差 -0.6878，CI [-0.7046, -0.6710]，逐样本差的中位数 -0.7091 → pass=False
- 判定条件：evaluated，系统起伏误差 2.0000（中位数 2.0000），n = 300 → pass=False
  - vs copy_flat：基线 1.0015（中位数 1.0126）；差 0.9985，CI [0.9865, 1.0089]，逐样本差的中位数 0.9874，系统更低的比例 0.0000 → 胜过=False
  - vs gt_affine_upper：基线 1.2360（中位数 0.9639）；差 0.7640，CI [0.6067, 0.8989]，逐样本差的中位数 1.0361，系统更低的比例 0.1433 → 胜过=False
  - vs copy_convex：基线 0.9363（中位数 0.9258）；差 1.0637，CI [1.0457, 1.0789]，逐样本差的中位数 1.0742，系统更低的比例 0.0000 → 胜过=False
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：1.0000（400/400）；分原因：{'truncated': 400}
- 截断率：1.0000；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
