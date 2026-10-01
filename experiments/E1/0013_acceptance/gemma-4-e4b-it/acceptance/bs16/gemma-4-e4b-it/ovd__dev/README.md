# E1 · gemma-4-e4b-it · ovd · dev

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4；适配器 gemma-4-e4b-it）
- 图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 生成：enable_thinking=False，max_new_tokens=2048，greedy，image_first
- 提示配置 sha256：e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6；模板：`'detect {categories}'`
- 样本数 n = 100（id 指纹 ff5e94e59b57a9ec…）；有 GT：False

## 标题指标

（没有 GT 或空集合，不算指标）

## 解析与截断

- 解析失败率：0.0100（1/100）；分原因：{'no_box': 1}
- 截断率：0.0000；思考段未闭合：0.0000

## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）

- 失败的样子：（无）
- 关思考时自发写出思考标记：0 条（其中思考段未闭合 0 条）

## 文件

- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；
- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。
