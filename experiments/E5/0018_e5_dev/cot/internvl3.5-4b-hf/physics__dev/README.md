# E5 · cot · internvl3.5-4b-hf · physics（d）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 477e636d2c1074b0…）；生成器任务版本 2
- 标题口径：全部样本
- 生成：enable_thinking=True，max_new_tokens=4096，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 polygon_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0625，条件 0.0000（中位数 0.0000），n = 16
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0625，条件 0.0000（中位数 0.0000），n = 16

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.0000 vs 最强非真值基线 copy 0.3807 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.9375（15/16）；分原因：{'thinking_unclosed': 14, 'truncated': 1}
- 截断率：0.9375；思考段未闭合：0.8750；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
