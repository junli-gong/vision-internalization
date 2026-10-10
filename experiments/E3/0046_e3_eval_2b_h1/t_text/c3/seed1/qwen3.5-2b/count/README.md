# E1 · qwen3.5-2b · count · eval

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 count）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'How many {object} are there in the image? Answer with a single number.'`
- 样本数 n = 442（id 指纹 39b1f0c9bc094646…）；有 GT：True
- 计数问句：countbenchqa。这套问句出自 Google 的 PaliGemma（CountBenchQA，big_vision 仓库），Gemma 家族可能见过（D36-2）。报告计数时一律注明。

## 标题指标（严格读数）

- accuracy（all）：0.6946，95% CI [0.6493, 0.7398]，n = 442

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.6946 | [0.6493, 0.7398] | 1.0000 | 442 / 442 | 0.6946 | [0.6493, 0.7398] |

- 无效 0 条 = 拒答 0 + 其他解析失败 0

## 解析与截断

- 解析失败率：0.0000（0/442）；分原因：{}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
