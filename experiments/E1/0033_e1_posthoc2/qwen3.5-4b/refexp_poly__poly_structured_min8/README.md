> 事后诊断（D72-3；DEV-0036）：结构强制的多边形，变体 poly_structured_min8（至少 8 个点、至多 64 个点，每个点恰好两个 0–1000 的整数，语法见 configs/e1_posthoc.json 的 format_forced_poly）；样本取 E1 的 think 子集的 id，关思考。

- 点数分布：{'n': 200, 'min': 8, 'p05': 8.0, 'p50': 9.0, 'p95': 64.0, 'max': 64, 'mean': 20.485, 'n_distinct': 15, 'top': [[8, 95], [64, 40], [9, 25], [10, 18], [11, 6], [12, 3], [13, 3], [14, 2], [15, 2], [17, 1]]}；写满上限的 40 条
- 近矩形（恰好 4 个点、面积 / 外接框面积 ≥ 0.95）：0/200（占全部 0.0000；占有效 0.0000）

# E1 · qwen3.5-4b · refexp_poly · forced_poly

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5；适配器 qwen3.5-4b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 poly_structured_min8（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 200（id 指纹 32412dfa8eeb0ed7…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.2521，95% CI [0.2144, 0.2902]，n = 200
- gIoU（all）：0.2534，95% CI [0.2237, 0.2832]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.2521 | [0.2144, 0.2902] | 1.0000 | 200 / 200 | 0.2521 | [0.2144, 0.2902] |
| gIoU | all | 0.2534 | [0.2237, 0.2832] | 1.0000 | 200 / 200 | 0.2534 | [0.2237, 0.2832] |

- 无效 0 条 = 拒答 0 + 其他解析失败 0

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 101 | 0 | 0.0000 | 1.0000 | 0.2643 | 0.2643 | [0.2137, 0.3173] |
| non_person | 99 | 0 | 0.0000 | 1.0000 | 0.2379 | 0.2379 | [0.1823, 0.2980] |

## 解析与截断

- 解析失败率：0.0000（0/200）；分原因：{}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
