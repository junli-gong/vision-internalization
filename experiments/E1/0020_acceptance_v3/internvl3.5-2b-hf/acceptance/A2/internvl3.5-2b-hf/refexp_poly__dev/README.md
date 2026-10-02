# E1 · internvl3.5-2b-hf · refexp_poly · dev

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5；适配器 internvl3.5-2b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 polygon）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Outline {expression} in the image with a polygon. Report the polygon vertices in JSON format as a list of [x, y] points: [[x1, y1], [x2, y2], ...]. Coordinates are relative to the image size and range from 0 to 1000, with x first. Output only one JSON block.'`
- 样本数 n = 500（id 指纹 7f6058f6ff154ed0…）；有 GT：True

## 标题指标（严格读数）

- cIoU（all）：0.0890，95% CI [0.0672, 0.1127]，n = 500
- gIoU（all）：0.0588，95% CI [0.0492, 0.0684]，n = 500

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| cIoU | all | 0.0890 | [0.0672, 0.1127] | 0.7760 | 388 / 500 | 0.1096 | [0.0851, 0.1370] |
| gIoU | all | 0.0588 | [0.0492, 0.0684] | 0.7760 | 388 / 500 | 0.0757 | [0.0645, 0.0882] |

- 无效 112 条 = 拒答 0 + 其他解析失败 112

## 拒答按被指代物体分列（person / 非 person；指标 cIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 238 | 0 | 0.0000 | 0.7395 | 0.0875 | 0.1138 | [0.0849, 0.1426] |
| non_person | 262 | 0 | 0.0000 | 0.8092 | 0.0905 | 0.1061 | [0.0693, 0.1533] |

## 解析与截断

- 解析失败率：0.2240（112/500）；分原因：{'ambiguous': 14, 'malformed': 6, 'too_few_points': 61, 'truncated': 31}
- 截断率：0.0620；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
