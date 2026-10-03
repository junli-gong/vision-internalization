# E1 · gemma-4-e4b-it · refexp_poly · eval

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4；适配器 gemma-4-e4b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.0577，95% CI [0.0521, 0.0636]，n = 8000
- gIoU（all）：0.0376，95% CI [0.0349, 0.0404]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0577 | [0.0521, 0.0636] | 0.1383 | 1106 / 8000 | 0.3088 | [0.2871, 0.3316] |
| gIoU | all | 0.0376 | [0.0349, 0.0404] | 0.1383 | 1106 / 8000 | 0.2720 | [0.2571, 0.2880] |

- 无效 6894 条 = 拒答 18 + 其他解析失败 6876

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 13 | 0.0036 | 0.1371 | 0.0516 | 0.2923 | [0.2643, 0.3219] |
| non_person | 4398 | 5 | 0.0011 | 0.1392 | 0.0626 | 0.3209 | [0.2899, 0.3514] |

## 解析与截断

- 解析失败率：0.8618（6894/8000）；分原因：{'malformed': 6786, 'no_polygon': 79, 'too_few_points': 28, 'truncated': 1}
- 截断率：0.0001；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 18, 'wrong_shape:box_2d': 6771, 'wrong_shape:point': 7}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
