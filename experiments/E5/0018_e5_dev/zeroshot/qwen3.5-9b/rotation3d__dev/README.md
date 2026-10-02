# E5 · zeroshot · qwen3.5-9b · rotation3d（a）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 cd451c7bac7ccbe5…）；生成器任务版本 4
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 relief_image

## 标题（严格 / 有效率 / 条件读数）

- 起伏误差（越低越好）：严格 2.0000（95% CI [2.0000, 2.0000]），有效率 0.0000，条件 null（中位数 null），n = 12
- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 12
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.0000 vs 最强非真值基线 copy 0.7111 → 不通过
- 判定条件：evaluated，系统 2.0000，gt_affine_upper 1.2007，copy_flat 0.9858 → pass=False，pass_vs_flat=False
- imagining：False

## 解析

- 解析失败率：1.0000（16/16）；分原因：{'truncated': 1, 'wrong_col_count': 7, 'wrong_row_count': 8}
- 截断率：0.0625；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
