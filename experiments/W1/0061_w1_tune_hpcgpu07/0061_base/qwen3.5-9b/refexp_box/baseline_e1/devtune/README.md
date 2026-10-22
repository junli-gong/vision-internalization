# E1 · qwen3.5-9b · refexp_box · eval

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Locate {expression} in the image. Report bbox coordinates in JSON format.'`
- 样本数 n = 96（id 指纹 b54c531e67174095…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.8750，95% CI [0.8021, 0.9375]，n = 96

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.8750 | [0.8021, 0.9375] | 0.9583 | 92 / 96 | 0.9130 | [0.8587, 0.9674] |

- 无效 4 条 = 拒答 0 + 其他解析失败 4

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 49 | 0 | 0.0000 | 1.0000 | 0.8980 | 0.8980 | [0.7959, 0.9796] |
| non_person | 47 | 0 | 0.0000 | 0.9149 | 0.8511 | 0.9302 | [0.8599, 1.0000] |

## 解析与截断

- 解析失败率：0.0417（4/96）；分原因：{'multiple_boxes': 3, 'truncated': 1}
- 截断率：0.0104；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
