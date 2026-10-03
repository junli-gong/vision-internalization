# E1 · gemma-4-e2b-it · refexp_box · eval

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'detect {expression}'`
- 样本数 n = 56857（id 指纹 23b7590046996716…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.1455，95% CI [0.1425, 0.1484]，n = 56857

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.1455 | [0.1425, 0.1484] | 0.2036 | 11576 / 56857 | 0.7148 | [0.7069, 0.7231] |

- 无效 45281 条 = 拒答 24679 + 其他解析失败 20602

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 26649 | 15531 | 0.5828 | 0.1709 | 0.1363 | 0.7978 | [0.7866, 0.8092] |
| non_person | 30208 | 9148 | 0.3028 | 0.2325 | 0.1537 | 0.6611 | [0.6502, 0.6720] |

## 解析与截断

- 解析失败率：0.7964（45281/56857）；分原因：{'empty': 2, 'malformed': 342, 'multiple_boxes': 577, 'no_box': 40122, 'out_of_range': 3, 'truncated': 4235}
- 截断率：0.0745；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 24679}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
