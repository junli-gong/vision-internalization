# E5 · cot · gemma-4-e4b-it · viewpoint（b）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 fc8d41eb198c180a…）；生成器任务版本 4
- 标题口径：环绕移动（全部档）
- 生成：enable_thinking=True，max_new_tokens=4096，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 polygon_image

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.1124（95% CI [0.0243, 0.2329]），有效率 0.6667，条件 0.1685（中位数 0.1255），n = 9
- mask IoU（全分辨率）：严格 0.1091（95% CI [0.0201, 0.2341]），有效率 0.6667，条件 0.1637（中位数 0.1046），n = 9

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.1124 vs 最强非真值基线 translate 0.2979 → 不通过
- 判定条件：unavailable
- imagining：None

## 解析

- 解析失败率：0.4375（7/16）；分原因：{'malformed': 6, 'thinking_unclosed': 1}
- 截断率：0.0625；思考段未闭合：0.0625；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
