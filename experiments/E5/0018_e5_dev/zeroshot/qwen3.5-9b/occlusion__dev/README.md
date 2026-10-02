# E5 · zeroshot · qwen3.5-9b · occlusion（c）· dev

- 样本：val 划分，每档 4 条，n = 12（id 指纹 ea15bc3fe20152d0…）；生成器任务版本 2
- 标题口径：全部样本
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 binary_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 12
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.0000 vs 最强非真值基线 convex_hull 0.7138 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：1.0000（12/12）；分原因：{'truncated': 3, 'wrong_row_count': 9}
- 截断率：0.2500；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
