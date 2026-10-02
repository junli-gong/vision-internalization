# E5 · cot · gemma-4-e2b-it · imagery（e）· dev

- 样本：val 划分，每档 4 条，n = 16（id 指纹 ec45558d08c54b9f…）；生成器任务版本 2
- 标题口径：全部样本
- 生成：enable_thinking=True，max_new_tokens=4096，greedy，generate_batch，批大小 64
- 提示配置 sha256：d84b6efc141c1bc281e6ea3adb5226602395ae1337e133bf5d0d34f7ed47402b；格式 polygon_canvas

## 标题（严格 / 有效率 / 条件读数）

- mask IoU（网格）：严格 0.2089（95% CI [0.0729, 0.3724]），有效率 0.6250，条件 0.3343（中位数 0.2402），n = 16
- mask IoU（全分辨率）：严格 0.2070（95% CI [0.0721, 0.3659]），有效率 0.6250，条件 0.3313（中位数 0.2350），n = 16

## 两级门槛（D28-13）

- 先决条件（iou_grid）：系统 0.2089 vs 最强非真值基线 no_transform 0.2245 → 不通过
- 判定条件：not_applicable
- imagining：False

## 解析

- 解析失败率：0.3750（6/16）；分原因：{'malformed': 4, 'thinking_unclosed': 1, 'truncated': 1}
- 截断率：0.1250；思考段未闭合：0.0625；拒答（诊断）：0 条

## 文件

- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/
