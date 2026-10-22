# E1 · qwen3.5-9b · refexp_poly · eval

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.2546，95% CI [0.2434, 0.2661]，n = 8000
- gIoU（all）：0.2148，95% CI [0.2069, 0.2225]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.2546 | [0.2434, 0.2661] | 0.3289 | 2631 / 8000 | 0.6083 | [0.5887, 0.6277] |
| gIoU | all | 0.2148 | [0.2069, 0.2225] | 0.3289 | 2631 / 8000 | 0.6531 | [0.6415, 0.6651] |

- 无效 5369 条 = 拒答 0 + 其他解析失败 5369

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 0 | 0.0000 | 0.2632 | 0.2237 | 0.6661 | [0.6408, 0.6914] |
| non_person | 4398 | 0 | 0.0000 | 0.3827 | 0.2786 | 0.5772 | [0.5527, 0.6053] |

## 解析与截断

- 解析失败率：0.6711（5369/8000）；分原因：{'malformed': 2, 'truncated': 5367}
- 截断率：0.6709；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'wrong_shape:bbox_2d': 2}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
