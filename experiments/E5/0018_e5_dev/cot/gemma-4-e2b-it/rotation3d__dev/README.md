# E5 · cot · gemma-4-e2b-it · rotation3d（a）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 cd451c7bac7ccbe5…）；生成器任务版本 4
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）
- 生成：enable_thinking=True，max_new_tokens=4096，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 polygon_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2273（95% CI [0.1218, 0.3582]），有效率 0.6667，条件 0.3410（中位数 0.2652），n = 12
- mask IoU（全分辨率）：严格 0.2272（95% CI [0.1209, 0.3561]），有效率 0.6667，条件 0.3408（中位数 0.2550），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2273 vs 最强非真值基线 copy 0.7111 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.3125（5/16）；分原因：{'malformed': 5}
- 截断率：0.0000；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
