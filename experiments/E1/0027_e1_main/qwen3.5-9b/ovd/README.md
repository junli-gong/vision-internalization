# E1 · qwen3.5-9b · ovd · eval

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，image_first；读法 text（解析 boxes）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Locate every instance that belongs to the following categories: "{categories}". Report bbox coordinates in JSON format.'`
- 样本数 n = 963（id 指纹 b0f4bb4f362373b5…）；有 GT：True

## 标题指标（严格读数）

- F1@0.5（coco_val2017）：0.5522，95% CI [0.5196, 0.5847]，n = 500
- F1@0.5（odinw13_mean_over_datasets）：0.7577，95% CI —，n = 13

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| F1@0.5 | coco_val2017 | 0.5522 | [0.5196, 0.5847] | 0.9000 | 450 / 500 | 0.6279 | [0.6012, 0.6564] |
| F1@0.5 | odinw13_mean_over_datasets | 0.7577 | — | 0.9892 | 458 / 463 | 0.7640 | — |

- 无效 55 条 = 拒答 0 + 其他解析失败 55

## 解析与截断

- 解析失败率：0.0571（55/963）；分原因：{'out_of_range': 1, 'truncated': 54}
- 截断率：0.0561；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
