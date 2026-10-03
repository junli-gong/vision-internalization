# E1 · qwen3.5-4b · refexp_poly · think

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5；适配器 qwen3.5-4b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 200（id 指纹 32412dfa8eeb0ed7…）；有 GT：True
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- cIoU（all）：0.0016，95% CI [0.0000, 0.0052]，n = 200
- gIoU（all）：0.0023，95% CI [0.0000, 0.0068]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0016 | [0.0000, 0.0052] | 0.0050 | 1 / 200 | 0.4545 | [0.4545, 0.4545] |
| gIoU | all | 0.0023 | [0.0000, 0.0068] | 0.0050 | 1 / 200 | 0.4545 | [0.4545, 0.4545] |

- 无效 199 条 = 拒答 0 + 其他解析失败 199

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 101 | 0 | 0.0000 | 0.0099 | 0.0030 | 0.4545 | [0.4545, 0.4545] |
| non_person | 99 | 0 | 0.0000 | 0.0000 | 0.0000 | null | — |

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 191 | 0.9948 | 0 | 144.0000 | cIoU 0.0018；gIoU 0.0024 |
| 强制收尾 | 9 | 1.0000 | 0 | 4096.0000 | cIoU 0.0000；gIoU 0.0000 |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 0 | — | — | — | — |

## 解析与截断

- 解析失败率：0.9950（199/200）；分原因：{'malformed': 199}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'wrong_shape:bbox_2d': 83, 'wrong_shape:point_2d': 104}
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
