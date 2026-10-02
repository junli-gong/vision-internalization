# E5 · cot · gemma-4-e4b-it · rotation3d（a）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 cd451c7bac7ccbe5…）；生成器任务版本 4
- 标题口径：档 [1, 2, 3]（下限 ≥ 30°）
- 生成：enable_thinking=True，max_new_tokens=4096，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 polygon_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2048（95% CI [0.0518, 0.4068]），有效率 0.5000，条件 0.4096（中位数 0.3075），n = 12
- mask IoU（全分辨率）：严格 0.1994（95% CI [0.0533, 0.3906]），有效率 0.5000，条件 0.3989（中位数 0.3165），n = 12

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2048 vs 最强非真值基线 copy 0.7111 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.5625（9/16）；分原因：{'malformed': 8, 'thinking_unclosed': 1}
- 截断率：0.0625；思考段未闭合：0.0625；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
