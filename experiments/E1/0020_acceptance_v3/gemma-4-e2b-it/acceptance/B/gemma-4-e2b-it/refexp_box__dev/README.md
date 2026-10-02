# E1 · gemma-4-e2b-it · refexp_box · dev

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=48、left 填充、顺序 selection、same_letters；不是冻结的协议配置（冻结值 64）
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'detect {expression}'`
- 样本数 n = 600（id 指纹 41845f242e31ea5f…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.1800，95% CI [0.1517, 0.2084]，n = 600

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.1800 | [0.1517, 0.2084] | 0.2533 | 152 / 600 | 0.7105 | [0.6382, 0.7829] |

- 无效 448 条 = 拒答 284 + 其他解析失败 164

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 283 | 178 | 0.6290 | 0.1873 | 0.1625 | 0.8679 | [0.7736, 0.9434] |
| non_person | 317 | 106 | 0.3344 | 0.3123 | 0.1956 | 0.6263 | [0.5351, 0.7172] |

## 解析与截断

- 解析失败率：0.7467（448/600）；分原因：{'malformed': 2, 'multiple_boxes': 6, 'no_box': 414, 'truncated': 26}
- 截断率：0.0433；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 284}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
