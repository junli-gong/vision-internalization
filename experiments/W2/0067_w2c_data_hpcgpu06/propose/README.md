# E1 · qwen3.5-9b · refexp_box · eval

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Locate {expression} in the image. Report bbox coordinates in JSON format.'`
- 样本数 n = 16000（id 指纹 eb4ee124cf20e389…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.8742，95% CI [0.8694, 0.8791]，n = 16000

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.8742 | [0.8694, 0.8791] | 0.9807 | 15691 / 16000 | 0.8914 | [0.8867, 0.8963] |

- 无效 309 条 = 拒答 1 + 其他解析失败 308

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 7687 | 1 | 0.0001 | 0.9900 | 0.9256 | 0.9350 | [0.9296, 0.9407] |
| non_person | 8313 | 0 | 0.0000 | 0.9721 | 0.8267 | 0.8504 | [0.8427, 0.8577] |

## 解析与截断

- 解析失败率：0.0193（309/16000）；分原因：{'malformed': 1, 'multiple_boxes': 287, 'no_box': 1, 'truncated': 20}
- 截断率：0.0013；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 1}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
