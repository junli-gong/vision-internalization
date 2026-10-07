# E5 · zeroshot · internvl3.5-4b-hf · physics（d）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 0ea63f1c30af02dc…）；生成器任务版本 2
- 标题口径：全部样本
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4；格式 binary_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0250，条件 0.0000（中位数 0.0000），n = 400
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0250，条件 0.0000（中位数 0.0000），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0000（中位数 0.0000）vs 最强非真值基线 copy 0.2906（中位数 0.2419）；差 -0.2906，CI [-0.3157, -0.2651]，逐样本差的中位数 -0.2419 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.9750（390/400）；分原因：{'no_grid': 1, 'thinking_unclosed': 3, 'truncated': 178, 'wrong_col_count': 7, 'wrong_row_count': 201}
- 截断率：0.4525；思考段未闭合：0.0075；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
