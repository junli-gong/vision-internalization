# E1 · qwen3.5-0.8b · count · think

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5；适配器 qwen3.5-0.8b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 count）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'How many {object} are there in the image? Answer with a single number.'`
- 样本数 n = 200（id 指纹 dda174e94e4c2e10…）；有 GT：True
- 计数问句：countbenchqa。这套问句出自 Google 的 PaliGemma（CountBenchQA，big_vision 仓库），Gemma 家族可能见过（D36-2）。报告计数时一律注明。
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- accuracy（all）：0.7650，95% CI [0.7050, 0.8200]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.7650 | [0.7050, 0.8200] | 0.9600 | 192 / 200 | 0.7969 | [0.7396, 0.8542] |

- 无效 8 条 = 拒答 0 + 其他解析失败 8

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 180 | 0.0000 | 0 | 255.5000 | accuracy 0.8222 |
| 强制收尾 | 18 | 0.3333 | 4 | 4096.0000 | accuracy 0.2778 |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 2 | 1.0000 | 0 | 132.5000 | accuracy 0.0000 |

## 解析与截断

- 解析失败率：0.0400（8/200）；分原因：{'multiple_numbers': 2, 'thinking_unclosed': 2, 'truncated': 4}
- 截断率：0.0200；思考段未闭合：0.0100

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
