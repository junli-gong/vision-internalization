# E1 · gemma-4-e2b-it · refexp_poly · eval

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.2405，95% CI [0.2343, 0.2472]，n = 8000
- gIoU（all）：0.2293，95% CI [0.2246, 0.2343]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.2405 | [0.2343, 0.2472] | 0.9876 | 7901 / 8000 | 0.2423 | [0.2367, 0.2493] |
| gIoU | all | 0.2293 | [0.2246, 0.2343] | 0.9876 | 7901 / 8000 | 0.2321 | [0.2277, 0.2370] |

- 无效 99 条 = 拒答 1 + 其他解析失败 98

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 1 | 0.0003 | 0.9892 | 0.2416 | 0.2433 | [0.2345, 0.2525] |
| non_person | 4398 | 0 | 0.0000 | 0.9864 | 0.2397 | 0.2416 | [0.2322, 0.2512] |

## 解析与截断

- 解析失败率：0.0124（99/8000）；分原因：{'malformed': 74, 'no_polygon': 10, 'too_few_points': 4, 'truncated': 11}
- 截断率：0.0014；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 1, 'wrong_shape:box_2d': 33}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
