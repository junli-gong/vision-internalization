# E1 · qwen3.5-0.8b · refexp_box · extra

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5；适配器 qwen3.5-0.8b）
- 图像口径：{'tag': 'qwen:1024', 'type': 'PixelBudget', 'pixel_budget_tokens': 1024, 'min_pixels': 1048576, 'max_pixels': 1048576}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 one_box）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'Locate {expression} in the image. Report bbox coordinates in JSON format.'`
- 样本数 n = 500（id 指纹 670f29b8f976ccb1…）；有 GT：True

## 标题指标（严格读数）

- acc@0.5（all）：0.8360，95% CI [0.8020, 0.8680]，n = 500

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| acc@0.5 | all | 0.8360 | [0.8020, 0.8680] | 0.9800 | 490 / 500 | 0.8531 | [0.8184, 0.8837] |

- 无效 10 条 = 拒答 0 + 其他解析失败 10

## 拒答按被指代物体分列（person / 非 person；指标 acc@0.5）

| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |
|---|---|---|---|---|---|---|---|
| person | 259 | 0 | 0.0000 | 0.9884 | 0.8842 | 0.8945 | [0.8554, 0.9336] |
| non_person | 241 | 0 | 0.0000 | 0.9710 | 0.7842 | 0.8077 | [0.7564, 0.8547] |

## 解析与截断

- 解析失败率：0.0200（10/500）；分原因：{'multiple_boxes': 8, 'truncated': 2}
- 截断率：0.0040；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
