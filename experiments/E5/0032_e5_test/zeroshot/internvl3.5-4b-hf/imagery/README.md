# E5 · zeroshot · internvl3.5-4b-hf · imagery（e）· eval

- 样本：test 划分，每档 100 条，n = 400（id 指纹 f94cb7b097c8d72b…）；生成器任务版本 2
- 标题口径：全部样本
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4；格式 binary_canvas

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0100，条件 0.0000（中位数 0.0000），n = 400
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0100，条件 0.0000（中位数 0.0000），n = 400

## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）

- 先决条件（iou_grid）：系统 0.0000（中位数 0.0000）vs 最强非真值基线 no_transform 0.2146（中位数 0.1304）；差 -0.2146，CI [-0.2377, -0.1923]，逐样本差的中位数 -0.1304 → pass=False
- 判定条件：not_applicable
- imagining：False
- 注：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 已作废（D59-4），不再输出。

## 解析

- 解析失败率：0.9900（396/400）；分原因：{'no_grid': 24, 'truncated': 173, 'wrong_row_count': 199}
- 截断率：0.4325；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
