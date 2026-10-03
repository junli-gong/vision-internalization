# E1 · internvl3.5-4b-hf · refexp_poly · eval

- 模型：OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed（internvl3.5；适配器 internvl3.5-4b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.1823，95% CI [0.1765, 0.1888]，n = 8000
- gIoU（all）：0.1653，95% CI [0.1615, 0.1691]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.1823 | [0.1765, 0.1888] | 0.9539 | 7631 / 8000 | 0.1918 | [0.1855, 0.1979] |
| gIoU | all | 0.1653 | [0.1615, 0.1691] | 0.9539 | 7631 / 8000 | 0.1733 | [0.1694, 0.1773] |

- 无效 369 条 = 拒答 6 + 其他解析失败 363

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 5 | 0.0014 | 0.9742 | 0.1448 | 0.1493 | [0.1424, 0.1561] |
| non_person | 4398 | 1 | 0.0002 | 0.9372 | 0.2107 | 0.2252 | [0.2153, 0.2353] |

## 解析与截断

- 解析失败率：0.0461（369/8000）；分原因：{'malformed': 49, 'no_polygon': 6, 'out_of_range': 1, 'too_few_points': 7, 'truncated': 306}
- 截断率：0.0382；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 6}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
