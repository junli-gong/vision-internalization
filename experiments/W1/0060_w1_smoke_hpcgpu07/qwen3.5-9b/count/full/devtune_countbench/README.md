# W1 · qwen3.5-9b · count · full · devtune_countbench

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 提示：configs/w1.json sha256 6a4704590d47abf7…（frozen = False）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune_countbench：n = 16（划分清单 49 条，指纹 3559c5aec49d23d3…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- accuracy：0.8750，95% CI [0.6875, 1.0000]，n = 16

## 回路

- 失败：{'ok': 16}；结束方式：{'submission': 16}；答案来源：{'submit_count': 16}
- 轮数：{'n': 16, 'min': 1, 'p05': 1.0, 'p50': 1.0, 'p95': 2.0, 'max': 2, 'mean': 1.375, 'n_distinct': 2, 'top': [[1, 10], [2, 6]]}
- 计数的工具调用：{'n': 16, 'min': 0, 'p05': 0.0, 'p50': 0.0, 'p95': 1.0, 'max': 1, 'mean': 0.375, 'n_distinct': 2, 'top': [[0, 10], [1, 6]]}
- 被拒的调用 0（{}）；解析失败的调用 0；工具报错 0
- 各工具执行次数：{'submit_count': 16, 'tile_view': 3, 'zoom': 3}
- 思考段的结束方式（逐轮）：{'natural': 22}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
