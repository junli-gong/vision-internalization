# E1 · gemma-4-e4b-it · refexp_box · dev

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4；适配器 gemma-4-e4b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'detect {expression}'`
- 样本数 n = 600（id 指纹 41845f242e31ea5f…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.2017，95% CI [0.1700, 0.2350]，n = 600

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.2017 | [0.1700, 0.2350] | 0.2433 | 146 / 600 | 0.8288 | [0.7671, 0.8836] |

- 无效 454 条 = 拒答 273 + 其他解析失败 181

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 283 | 185 | 0.6537 | 0.1908 | 0.1696 | 0.8889 | [0.7963, 0.9630] |
| non_person | 317 | 88 | 0.2776 | 0.2902 | 0.2303 | 0.7935 | [0.7065, 0.8696] |

## 解析与截断

- 解析失败率：0.7567（454/600）；分原因：{'malformed': 3, 'multiple_boxes': 11, 'no_box': 421, 'truncated': 19}
- 截断率：0.0317；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 273}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
