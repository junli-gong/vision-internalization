# E1 · internvl3.5-2b-hf · refexp_poly · eval

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5；适配器 internvl3.5-2b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 8000（id 指纹 87c8c303e8dd475e…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.0763，95% CI [0.0722, 0.0804]，n = 8000
- gIoU（all）：0.0594，95% CI [0.0570, 0.0615]，n = 8000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0763 | [0.0722, 0.0804] | 0.7591 | 6073 / 8000 | 0.0962 | [0.0916, 0.1012] |
| gIoU | all | 0.0594 | [0.0570, 0.0615] | 0.7591 | 6073 / 8000 | 0.0782 | [0.0753, 0.0811] |

- 无效 1927 条 = 拒答 2 + 其他解析失败 1925

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 3602 | 1 | 0.0003 | 0.7479 | 0.0709 | 0.0906 | [0.0848, 0.0974] |
| non_person | 4398 | 1 | 0.0002 | 0.7683 | 0.0807 | 0.1007 | [0.0933, 0.1085] |

## 解析与截断

- 解析失败率：0.2409（1927/8000）；分原因：{'ambiguous': 259, 'malformed': 227, 'no_polygon': 2, 'out_of_range': 6, 'too_few_points': 1007, 'truncated': 426}
- 截断率：0.0532；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 2}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
