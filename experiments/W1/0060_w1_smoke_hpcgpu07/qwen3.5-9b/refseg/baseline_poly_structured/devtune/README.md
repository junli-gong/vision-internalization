> 事后诊断（D72-3；DEV-0036）：结构强制的多边形，变体 poly_structured（至少 3 个点、至多 64 个点，每个点恰好两个 0–1000 的整数，语法见 configs/e1_posthoc.json 的 format_forced_poly）；样本取 E1 的 w1:devtune 子集的 id，关思考。

- 点数分布：{'n': 32, 'min': 3, 'p05': 3.0, 'p50': 4.0, 'p95': 5.449999999999999, 'max': 13, 'mean': 4.1875, 'n_distinct': 5, 'top': [[4, 23], [3, 6], [5, 1], [6, 1], [13, 1]]}；写满上限的 0 条
- 近矩形（恰好 4 个点、面积 / 外接框面积 ≥ 0.95）：4/32（占全部 0.1250；占有效 0.1250）

# E1 · qwen3.5-9b · refexp_poly · forced_poly

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 poly_structured（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 32（id 指纹 2ebdfae52f54db14…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.4032，95% CI [0.3240, 0.4773]，n = 32
- gIoU（all）：0.4236，95% CI [0.3485, 0.4977]，n = 32

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.4032 | [0.3240, 0.4773] | 1.0000 | 32 / 32 | 0.4032 | [0.3240, 0.4773] |
| gIoU | all | 0.4236 | [0.3485, 0.4977] | 1.0000 | 32 / 32 | 0.4236 | [0.3485, 0.4977] |

- 无效 0 条 = 拒答 0 + 其他解析失败 0

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 15 | 0 | 0.0000 | 1.0000 | 0.3138 | 0.3138 | [0.2261, 0.4100] |
| non_person | 17 | 0 | 0.0000 | 1.0000 | 0.4527 | 0.4527 | [0.3604, 0.5596] |

## 解析与截断

- 解析失败率：0.0000（0/32）；分原因：{}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
