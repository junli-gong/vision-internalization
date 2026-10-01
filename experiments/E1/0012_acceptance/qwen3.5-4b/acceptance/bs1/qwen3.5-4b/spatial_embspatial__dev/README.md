# E1 · qwen3.5-4b · spatial_embspatial · dev

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5；适配器 qwen3.5-4b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=64，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'Question: {question}\nOptions:\n{options}\nPlease select the correct answer from the options above.'`
- 样本数 n = 50（id 指纹 a4453369d042eb01…）；有 GT：True

## 标题指标

- accuracy（all）：0.3000，95% CI [0.1800, 0.4200]，n = 50

## 解析与截断

- 解析失败率：0.7000（35/50）；分原因：{'no_choice': 1, 'truncated': 34}
- 截断率：0.6800；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
