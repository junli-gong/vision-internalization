# E1 · gemma-4-e2b-it · refexp_box · dev

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=128，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'detect {expression}'`
- 样本数 n = 100（id 指纹 f5c5d7fd72cf219f…）；有 GT：True

## 标题指标

- acc@0.5（all）：0.0800，95% CI [0.0300, 0.1400]，n = 100

## 解析与截断

- 解析失败率：0.8700（87/100）；分原因：{'multiple_boxes': 3, 'no_box': 72, 'truncated': 12}
- 截断率：0.1200；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
