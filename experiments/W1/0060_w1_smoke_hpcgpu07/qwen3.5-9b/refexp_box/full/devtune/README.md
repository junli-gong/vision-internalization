# W1 · qwen3.5-9b · refexp_box · full · devtune

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 提示：configs/w1.json sha256 6a4704590d47abf7…（frozen = False）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune：n = 16（划分清单 300 条，指纹 21e50b3bb13e25be…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- acc@0.5：0.3750，95% CI [0.1250, 0.6250]，n = 16

## 回路

- 失败：{'no_answer': 4, 'ok': 12}；结束方式：{'round_cap': 4, 'submission': 12}；答案来源：{'ok': 4, 'submit_box': 12}
- 轮数：{'n': 16, 'min': 1, 'p05': 1.75, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 4.5625, 'n_distinct': 6, 'top': [[6, 9], [2, 3], [1, 1], [3, 1], [4, 1], [5, 1]]}
- 计数的工具调用：{'n': 16, 'min': 0, 'p05': 0.75, 'p50': 5.0, 'p95': 6.0, 'max': 6, 'mean': 3.8125, 'n_distinct': 7, 'top': [[5, 5], [6, 4], [1, 3], [0, 1], [2, 1], [3, 1], [4, 1]]}
- 被拒的调用 0（{}）；解析失败的调用 0；工具报错 0
- 各工具执行次数：{'draw_box': 2, 'edges': 1, 'grabcut': 4, 'read_pixels': 9, 'show': 4, 'submit_box': 12, 'tile_view': 3, 'watershed': 1, 'zoom': 37}
- 思考段的结束方式（逐轮）：{'forced': 2, 'natural': 71}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
