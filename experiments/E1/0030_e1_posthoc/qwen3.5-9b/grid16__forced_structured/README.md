> 事后诊断（D66-2；E1 预注册 §10.2，写于运行之前）：格式强制，读法 structured（固定长度 271 个 token，不判截断）；样本取 E1 的 think 子集的 id，关思考。规则见 configs/e1_posthoc.json 的 format_forced。

# E1 · qwen3.5-9b · grid16 · forced

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=271，greedy，image_first；读法 forced_structured（解析 grid16）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Segment {expression} in the image as a 16x16 binary grid. Split the image into 16 rows (top to bottom) and 16 columns (left to right) of equal size. Output exactly 16 lines, each with 16 digits and no spaces: 1 if the cell is mostly covered by the object, otherwise 0. Output only the grid.'`
- 样本数 n = 200（id 指纹 9e8242d61125c8ae…）；有 GT：True

## 标题指标（严格读数）

- gIoU（all）：0.0038，95% CI [0.0000, 0.0153]，n = 200
- cIoU（all）：0.0197，95% CI [0.0000, 0.0691]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| gIoU | all | 0.0038 | [0.0000, 0.0153] | 1.0000 | 200 / 200 | 0.0038 | [0.0000, 0.0153] |
| cIoU | all | 0.0197 | [0.0000, 0.0691] | 1.0000 | 200 / 200 | 0.0197 | [0.0000, 0.0691] |

- 无效 0 条 = 拒答 0 + 其他解析失败 0

## 拒答按被指代物体分列（person / 非 person；指标 gIoU）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 104 | 0 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| non_person | 96 | 0 | 0.0000 | 1.0000 | 0.0080 | 0.0080 | [0.0000, 0.0240] |

## 解析与截断

- 解析失败率：0.0000（0/200）；分原因：{}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
