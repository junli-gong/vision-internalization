# E1 · gemma-4-e4b-it · refexp_poly · dev

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4；适配器 gemma-4-e4b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 500（id 指纹 7f6058f6ff154ed0…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.0691，95% CI [0.0494, 0.0923]，n = 500
- gIoU（all）：0.0514，95% CI [0.0381, 0.0656]，n = 500

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0691 | [0.0494, 0.0923] | 0.1740 | 87 / 500 | 0.3244 | [0.2568, 0.3911] |
| gIoU | all | 0.0514 | [0.0381, 0.0656] | 0.1740 | 87 / 500 | 0.2956 | [0.2409, 0.3556] |

- 无效 413 条 = 拒答 0 + 其他解析失败 413

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 238 | 0 | 0.0000 | 0.1807 | 0.0512 | 0.2510 | [0.1681, 0.3360] |
| non_person | 262 | 0 | 0.0000 | 0.1679 | 0.0860 | 0.3876 | [0.2982, 0.4606] |

## 解析与截断

- 解析失败率：0.8260（413/500）；分原因：{'malformed': 410, 'no_polygon': 2, 'too_few_points': 1}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'wrong_shape:box_2d': 409, 'wrong_shape:point': 1}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
