# E1 · qwen3.5-0.8b · refexp_poly · eval

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5；适配器 qwen3.5-0.8b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.0636，95% CI [0.0570, 0.0701]，n = 8000
- gIoU（all）：0.0427，95% CI [0.0390, 0.0461]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0636 | [0.0570, 0.0701] | 0.0895 | 716 / 8000 | 0.4234 | [0.3929, 0.4541] |
| gIoU | all | 0.0427 | [0.0390, 0.0461] | 0.0895 | 716 / 8000 | 0.4769 | [0.4535, 0.5013] |

- 无效 7284 条 = 拒答 0 + 其他解析失败 7284

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 0 | 0.0000 | 0.0972 | 0.0671 | 0.4374 | [0.3993, 0.4812] |
| non_person | 4398 | 0 | 0.0000 | 0.0832 | 0.0608 | 0.4115 | [0.3706, 0.4552] |

## 解析与截断

- 解析失败率：0.9105（7284/8000）；分原因：{'malformed': 1, 'truncated': 7283}
- 截断率：0.9104；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'wrong_shape:bbox_2d': 1}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
