# E1 · gemma-4-e2b-it · rel_depth_nyu · think

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 choice）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?\nSelect from the following choices.\n(A) A is closer\n(B) B is closer'`
- 样本数 n = 200（id 指纹 07699439489a3ac7…）；有 GT：True
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- accuracy（all）：0.7150，95% CI [0.6550, 0.7750]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.7150 | [0.6550, 0.7750] | 0.9950 | 199 / 200 | 0.7186 | [0.6533, 0.7789] |

- 无效 1 条 = 拒答 0 + 其他解析失败 1

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 200 | 0.0050 | 0 | 398.0000 | accuracy 0.7150 |
| 强制收尾 | 0 | — | — | — | — |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 0 | — | — | — | — |

## 解析与截断

- 解析失败率：0.0050（1/200）；分原因：{'multiple_choices': 1}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
