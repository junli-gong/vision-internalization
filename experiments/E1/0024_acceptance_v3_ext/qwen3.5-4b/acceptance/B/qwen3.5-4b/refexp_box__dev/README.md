# E1 · qwen3.5-4b · refexp_box · dev

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5；适配器 qwen3.5-4b）
- 图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=48、left 填充、顺序 selection、same_letters；不是冻结的协议配置（冻结值 64）
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Locate {expression} in the image. Report bbox coordinates in JSON format.'`
- 样本数 n = 600（id 指纹 41845f242e31ea5f…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.8583，95% CI [0.8317, 0.8850]，n = 600

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.8583 | [0.8317, 0.8850] | 0.9767 | 586 / 600 | 0.8788 | [0.8515, 0.9010] |

- 无效 14 条 = 拒答 0 + 其他解析失败 14

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 283 | 0 | 0.0000 | 0.9894 | 0.8834 | 0.8929 | [0.8536, 0.9286] |
| non_person | 317 | 0 | 0.0000 | 0.9653 | 0.8360 | 0.8660 | [0.8268, 0.9020] |

## 解析与截断

- 解析失败率：0.0233（14/600）；分原因：{'multiple_boxes': 13, 'truncated': 1}
- 截断率：0.0017；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
