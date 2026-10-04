# E1 · internvl3.5-4b-hf · grid16 · think

- 模型：OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed（internvl3.5；适配器 internvl3.5-4b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 grid16）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Segment {expression} in the image as a 16x16 binary grid. Split the image into 16 rows (top to bottom) and 16 columns (left to right) of equal size. Output exactly 16 lines, each with 16 digits and no spaces: 1 if the cell is mostly covered by the object, otherwise 0. Output only the grid.'`
- 样本数 n = 200（id 指纹 9e8242d61125c8ae…）；有 GT：True
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- gIoU（all）：0.0965，95% CI [0.0796, 0.1142]，n = 200
- cIoU（all）：0.1216，95% CI [0.1016, 0.1439]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| gIoU | all | 0.0965 | [0.0796, 0.1142] | 0.7700 | 154 / 200 | 0.1254 | [0.1053, 0.1472] |
| cIoU | all | 0.1216 | [0.1016, 0.1439] | 0.7700 | 154 / 200 | 0.1278 | [0.1073, 0.1496] |

- 无效 46 条 = 拒答 0 + 其他解析失败 46

## 拒答按被指代物体分列（person / 非 person；指标 gIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 104 | 0 | 0.0000 | 0.8077 | 0.1049 | 0.1299 | [0.1011, 0.1655] |
| non_person | 96 | 0 | 0.0000 | 0.7292 | 0.0874 | 0.1199 | [0.0970, 0.1458] |

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 24 | 0.0417 | 0 | 3401.5000 | gIoU 0.1297；cIoU 0.1301 |
| 强制收尾 | 176 | 0.2557 | 39 | 4096.0000 | gIoU 0.0920；cIoU 0.1202 |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 0 | — | — | — | — |

## 解析与截断

- 解析失败率：0.2300（46/200）；分原因：{'multiple_blocks': 2, 'no_grid': 1, 'truncated': 39, 'wrong_col_count': 2, 'wrong_row_count': 2}
- 截断率：0.1950；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
