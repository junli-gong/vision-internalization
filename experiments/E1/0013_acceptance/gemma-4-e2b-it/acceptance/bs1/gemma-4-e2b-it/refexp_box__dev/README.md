# E1 · gemma-4-e2b-it · refexp_box · dev

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4；适配器 gemma-4-e2b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=256，greedy，image_first
- 提示配置 sha256：e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6；模板：`'detect {expression}'`
- 样本数 n = 100（id 指纹 f5c5d7fd72cf219f…）；有 GT：True

## 标题指标

- acc@0.5（all）：0.0800，95% CI [0.0300, 0.1400]，n = 100

## 解析与截断

- 解析失败率：0.8800（88/100）；分原因：{'multiple_boxes': 3, 'no_box': 82, 'truncated': 3}
- 截断率：0.0300；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：{'refusal': 60}
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
