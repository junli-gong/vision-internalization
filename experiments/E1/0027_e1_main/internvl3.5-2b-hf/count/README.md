# E1 · internvl3.5-2b-hf · count · eval

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5；适配器 internvl3.5-2b-hf）
- 图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first；读法 text（解析 count）
- 批配置：batch_size=64、left 填充、顺序 selection、same_letters；冻结的协议配置
- 提示配置 sha256：5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158；模板：`'How many {object} are there in the image? Answer with a single number.'`
- 样本数 n = 442（id 指纹 39b1f0c9bc094646…）；有 GT：True
- 计数问句：countbenchqa。这套问句出自 Google 的 PaliGemma（CountBenchQA，big_vision 仓库），Gemma 家族可能见过（D36-2）。报告计数时一律注明。

## 标题指标（严格读数）

- accuracy（all）：0.5769，95% CI [0.5317, 0.6244]，n = 442

## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）

| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |
|---|---|---|---|---|---|---|---|
| accuracy | all | 0.5769 | [0.5317, 0.6244] | 0.8869 | 392 / 442 | 0.6505 | [0.6020, 0.6964] |

- 无效 50 条 = 拒答 0 + 其他解析失败 50

## 解析与截断

- 解析失败率：0.1131（50/442）；分原因：{'multiple_numbers': 45, 'thinking_unclosed': 4, 'truncated': 1}
- 截断率：0.0090；思考段未闭合：0.0090

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：4 条（其中思考段未闭合 4 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
