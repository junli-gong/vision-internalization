# E1 · qwen3.5-9b · ovd · dev

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=512，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'Locate every instance that belongs to the following categories: "{categories}". Report bbox coordinates in JSON format.'`
- 样本数 n = 100（id 指纹 ff5e94e59b57a9ec…）；有 GT：False

## 标题指标

（没有 GT 或空集合，不算指标）

## 解析与截断

- 解析失败率：0.3600（36/100）；分原因：{'truncated': 36}
- 截断率：0.3600；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
