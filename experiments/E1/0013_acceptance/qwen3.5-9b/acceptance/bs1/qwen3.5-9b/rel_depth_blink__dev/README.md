# E1 · qwen3.5-9b · rel_depth_blink · dev

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first
- 提示配置 sha256：e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6；模板：`'{prompt}'`
- 样本数 n = 12（id 指纹 03fb367a6edcb5d6…）；有 GT：True

## 标题指标

- accuracy（all）：0.9167，95% CI [0.7500, 1.0000]，n = 12

## 解析与截断

- 解析失败率：0.0000（0/12）；分原因：{}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
