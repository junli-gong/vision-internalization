# E1 · qwen3.5-2b · ovd · think

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 boxes）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Locate every instance that belongs to the following categories: "{categories}". Report bbox coordinates in JSON format.'`
- 样本数 n = 200（id 指纹 67f9cd4ee611db6d…）；有 GT：True
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- F1@0.5（coco_val2017）：0.5031，95% CI [0.4481, 0.5569]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| F1@0.5 | coco_val2017 | 0.5031 | [0.4481, 0.5569] | 0.9250 | 185 / 200 | 0.5946 | [0.5536, 0.6383] |

- 无效 15 条 = 拒答 0 + 其他解析失败 15

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 183 | 0.0164 | 2 | 188.0000 | F1@0.5（合并）0.5869 |
| 强制收尾 | 17 | 0.7059 | 12 | 4096.0000 | F1@0.5（合并）0.1023 |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 0 | — | — | — | — |

## 解析与截断

- 解析失败率：0.0750（15/200）；分原因：{'no_box': 1, 'truncated': 14}
- 截断率：0.0700；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
