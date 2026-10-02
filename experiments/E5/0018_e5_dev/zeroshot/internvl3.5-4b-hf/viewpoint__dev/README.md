# E5 · zeroshot · internvl3.5-4b-hf · viewpoint（b）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 fc8d41eb198c180a…）；生成器任务版本 4
- 标题口径：环绕移动（全部档）
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 binary_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 9
- mask IoU（全分辨率）：严格 0.0000（95% CI [0.0000, 0.0000]），有效率 0.0000，条件 null（中位数 null），n = 9

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.0000 vs 最强非真值基线 translate 0.2979 → 不通过
- 判定条件：unavailable
- imagining：None

## 解析

- 解析失败率：1.0000（16/16）；分原因：{'wrong_row_count': 16}
- 截断率：0.0000；思考段未闭合：0.0000；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
