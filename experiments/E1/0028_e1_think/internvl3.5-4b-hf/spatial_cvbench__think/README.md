# E1 · internvl3.5-4b-hf · spatial_cvbench · think

- 模型：OpenGVLab/InternVL3_5-4B-HF@6bd4487402110ef9889ba50eb7aefeb302526fed（internvl3.5；适配器 internvl3.5-4b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=True，max_new_tokens=5120，greedy，image_first；读法 text（解析 choice）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；模板：`'{prompt}'`
- 样本数 n = 200（id 指纹 897e97d1e3f83e5e…）；有 GT：True
- 预算强制收尾（D59-1）：思考 4096 token，作答 1024 token（generate_batch_budgeted，规则见 src/adapters/budget.py）

## 标题指标（严格读数）

- accuracy（all）：0.7450，95% CI [0.6800, 0.8000]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.7450 | [0.6800, 0.8000] | 0.9650 | 193 / 200 | 0.7720 | [0.7098, 0.8238] |

- 无效 7 条 = 拒答 0 + 其他解析失败 7

## 思考段的结束方式（预算强制收尾，D59-1）

| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |
|---|---|---|---|---|---|
| 自然收尾 | 173 | 0.0173 | 0 | 142.0000 | accuracy 0.8092 |
| 强制收尾 | 27 | 0.1481 | 4 | 4096.0000 | accuracy 0.3333 |
| 没写思考段（直接作答） | 0 | — | — | — | — |
| 思考段里就停了 | 0 | — | — | — | — |

## 解析与截断

- 解析失败率：0.0350（7/200）；分原因：{'multiple_choices': 3, 'truncated': 4}
- 截断率：0.0200；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：不适用（开思考）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
