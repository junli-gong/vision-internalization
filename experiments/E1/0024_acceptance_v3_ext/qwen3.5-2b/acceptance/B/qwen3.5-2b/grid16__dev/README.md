# E1 · qwen3.5-2b · grid16 · dev

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 grid16）
- 批配置：batch_size=48、left 填充、顺序 selection、same_letters；不是冻结的协议配置（冻结值 64）
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Segment {expression} in the image as a 16x16 binary grid. Split the image into 16 rows (top to bottom) and 16 columns (left to right) of equal size. Output exactly 16 lines, each with 16 digits and no spaces: 1 if the cell is mostly covered by the object, otherwise 0. Output only the grid.'`
- 样本数 n = 500（id 指纹 88bae50d33d05961…）；有 GT：True

## 标题指标（严格读数）

- gIoU（all）：0.0003，95% CI [0.0000, 0.0007]，n = 500
- cIoU（all）：0.0012，95% CI [0.0000, 0.0033]，n = 500

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| gIoU | all | 0.0003 | [0.0000, 0.0007] | 0.0140 | 7 / 500 | 0.0192 | [0.0000, 0.0449] |
| cIoU | all | 0.0012 | [0.0000, 0.0033] | 0.0140 | 7 / 500 | 0.0494 | [0.0000, 0.0753] |

- 无效 493 条 = 拒答 0 + 其他解析失败 493

## 拒答按被指代物体分列（person / 非 person；指标 gIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 243 | 0 | 0.0000 | 0.0000 | 0.0000 | null | — |
| non_person | 257 | 0 | 0.0000 | 0.0272 | 0.0005 | 0.0192 | [0.0000, 0.0449] |

## 解析与截断

- 解析失败率：0.9860（493/500）；分原因：{'truncated': 488, 'wrong_row_count': 5}
- 截断率：0.9760；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
