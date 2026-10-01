# E1 · qwen3.5-0.8b · count · dev

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5；适配器 qwen3.5-0.8b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=64，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'How many {object} are there in the image? Answer with a single number.'`
- 样本数 n = 49（id 指纹 9cc83f756324ac38…）；有 GT：True

## 标题指标

- accuracy（all）：0.7347，95% CI [0.5918, 0.8571]，n = 49

## 解析与截断

- 解析失败率：0.0000（0/49）；分原因：{}
- 截断率：0.0000；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
