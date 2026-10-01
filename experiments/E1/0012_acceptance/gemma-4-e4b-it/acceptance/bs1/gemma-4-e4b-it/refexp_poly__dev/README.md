# E1 · gemma-4-e4b-it · refexp_poly · dev

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4；适配器 gemma-4-e4b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first
- 提示配置 sha256：974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 100（id 指纹 f5c5d7fd72cf219f…）；有 GT：True

## 标题指标

- cIoU（all）：0.0381，95% CI [0.0053, 0.0806]，n = 100
- gIoU（all）：0.0237，95% CI [0.0054, 0.0471]，n = 100

## 解析与截断

- 解析失败率：0.9300（93/100）；分原因：{'malformed': 93}
- 截断率：0.0000；思考段未闭合：0.0000

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
