# E1 · gemma-4-e4b-it · ovd · eval

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4；适配器 gemma-4-e4b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，image_first；读法 text（解析 boxes）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'detect {categories}'`
- 样本数 n = 963（id 指纹 b0f4bb4f362373b5…）；有 GT：True

## 标题指标（严格读数）

- F1@0.5（coco_val2017）：0.4366，95% CI [0.4105, 0.4633]，n = 500
- F1@0.5（odinw13_mean_over_datasets）：0.1529，95% CI —，n = 13

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| F1@0.5 | coco_val2017 | 0.4366 | [0.4105, 0.4633] | 0.9600 | 480 / 500 | 0.4425 | [0.4152, 0.4684] |
| F1@0.5 | odinw13_mean_over_datasets | 0.1529 | — | 0.2873 | 133 / 463 | 0.5240 | — |

- 无效 350 条 = 拒答 119 + 其他解析失败 231

## 解析与截断

- 解析失败率：0.3634（350/963）；分原因：{'ambiguous': 5, 'malformed': 1, 'no_box': 344}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 119}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
