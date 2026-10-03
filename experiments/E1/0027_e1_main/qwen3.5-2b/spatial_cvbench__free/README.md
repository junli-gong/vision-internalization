# E1 · qwen3.5-2b · spatial_cvbench · free

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=1024，greedy，image_first；读法 text（解析 choice）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'{prompt}'`
- 样本数 n = 200（id 指纹 897e97d1e3f83e5e…）；有 GT：True

## 标题指标（严格读数）

- accuracy（all）：0.7550，95% CI [0.6900, 0.8150]，n = 200

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.7550 | [0.6900, 0.8150] | 0.9550 | 191 / 200 | 0.7906 | [0.7277, 0.8482] |

- 无效 9 条 = 拒答 1 + 其他解析失败 8

## 解析与截断

- 解析失败率：0.0450（9/200）；分原因：{'multiple_choices': 4, 'truncated': 5}
- 截断率：0.0250；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 1}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
