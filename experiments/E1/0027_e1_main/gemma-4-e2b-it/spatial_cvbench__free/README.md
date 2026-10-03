# E1 · gemma-4-e2b-it · spatial_cvbench · free

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 choice）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'{prompt}'`
- 样本数 n = 200（id 指纹 897e97d1e3f83e5e…）；有 GT：True

## 标题指标（严格读数）

- accuracy（all）：0.6750，95% CI [0.6100, 0.7400]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.6750 | [0.6100, 0.7400] | 0.9000 | 180 / 200 | 0.7500 | [0.6889, 0.8167] |

- 无效 20 条 = 拒答 3 + 其他解析失败 17

## 解析与截断

- 解析失败率：0.1000（20/200）；分原因：{'multiple_choices': 17, 'no_choice': 2, 'truncated': 1}
- 截断率：0.0050；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 3}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
