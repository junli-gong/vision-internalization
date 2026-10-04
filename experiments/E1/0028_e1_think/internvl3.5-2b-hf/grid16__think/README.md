# E1 · internvl3.5-2b-hf · grid16 · think

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5；适配器 internvl3.5-2b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 grid16）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Segment {expression} in the image as a 16x16 binary grid. Split the image into 16 rows (top to bottom) and 16 columns (left to right) of equal size. Output exactly 16 lines, each with 16 digits and no spaces: 1 if the cell is mostly covered by the object, otherwise 0. Output only the grid.'`
- 样本数 n = 200（id 指纹 9e8242d61125c8ae…）；有 GT：True
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- gIoU（all）：0.0009，95% CI [0.0000, 0.0028]，n = 200
- cIoU（all）：0.0081，95% CI [0.0000, 0.0240]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| gIoU | all | 0.0009 | [0.0000, 0.0028] | 0.0250 | 5 / 200 | 0.0367 | [0.0000, 0.1101] |
| cIoU | all | 0.0081 | [0.0000, 0.0240] | 0.0250 | 5 / 200 | 0.1387 | [0.0000, 0.1743] |

- 无效 195 条 = 拒答 0 + 其他解析失败 195

## 拒答按被指代物体分列（person / 非 person；指标 gIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 104 | 0 | 0.0000 | 0.0385 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| non_person | 96 | 0 | 0.0000 | 0.0104 | 0.0019 | 0.1835 | [0.1835, 0.1835] |

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 0 | — | — | — | — |
| 强制收尾 | 200 | 0.9750 | 30 | 4096.0000 | gIoU 0.0009；cIoU 0.0081 |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 0 | — | — | — | — |

## 解析与截断

- 解析失败率：0.9750（195/200）；分原因：{'multiple_blocks': 1, 'truncated': 30, 'wrong_col_count': 2, 'wrong_row_count': 162}
- 截断率：0.1500；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
