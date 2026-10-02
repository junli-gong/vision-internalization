# E1 · qwen3.5-4b · grid16 · dev

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5；适配器 qwen3.5-4b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 grid16）
- 批配置：batch_size=48、left 填充、顺序 selection、same_letters；不是冻结的协议配置（冻结值 64）
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Segment {expression} in the image as a 16x16 binary grid. Split the image into 16 rows (top to bottom) and 16 columns (left to right) of equal size. Output exactly 16 lines, each with 16 digits and no spaces: 1 if the cell is mostly covered by the object, otherwise 0. Output only the grid.'`
- 样本数 n = 500（id 指纹 88bae50d33d05961…）；有 GT：True

## 标题指标（严格读数）

- gIoU（all）：0.0081，95% CI [0.0035, 0.0141]，n = 500
- cIoU（all）：0.0585，95% CI [0.0290, 0.0893]，n = 500

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| gIoU | all | 0.0081 | [0.0035, 0.0141] | 0.9980 | 499 / 500 | 0.0081 | [0.0037, 0.0141] |
| cIoU | all | 0.0585 | [0.0290, 0.0893] | 0.9980 | 499 / 500 | 0.0586 | [0.0304, 0.0903] |

- 无效 1 条 = 拒答 0 + 其他解析失败 1

## 拒答按被指代物体分列（person / 非 person；指标 gIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 243 | 0 | 0.0000 | 1.0000 | 0.0094 | 0.0094 | [0.0036, 0.0161] |
| non_person | 257 | 0 | 0.0000 | 0.9961 | 0.0069 | 0.0069 | [0.0008, 0.0170] |

## 解析与截断

- 解析失败率：0.0020（1/500）；分原因：{'wrong_row_count': 1}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
