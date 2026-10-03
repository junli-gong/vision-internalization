# E1 · qwen3.5-2b · refexp_poly · eval

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.0005，95% CI [0.0001, 0.0009]，n = 8000
- gIoU（all）：0.0003，95% CI [0.0001, 0.0006]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0005 | [0.0001, 0.0009] | 0.0015 | 12 / 8000 | 0.1588 | [0.0858, 0.3402] |
| gIoU | all | 0.0003 | [0.0001, 0.0006] | 0.0015 | 12 / 8000 | 0.2301 | [0.1095, 0.3723] |

- 无效 7988 条 = 拒答 0 + 其他解析失败 7988

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 0 | 0.0000 | 0.0011 | 0.0003 | 0.5213 | [0.3510, 0.5930] |
| non_person | 4398 | 0 | 0.0000 | 0.0018 | 0.0006 | 0.1201 | [0.0575, 0.2532] |

## 解析与截断

- 解析失败率：0.9985（7988/8000）；分原因：{'malformed': 7888, 'no_polygon': 15, 'truncated': 85}
- 截断率：0.0106；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'wrong_shape:bbox_2d': 906, 'wrong_shape:point_2d': 6979}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
