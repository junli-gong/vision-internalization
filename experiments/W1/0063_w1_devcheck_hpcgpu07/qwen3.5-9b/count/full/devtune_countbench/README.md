# W1 · qwen3.5-9b · count · full · devtune_countbench

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 协议开关：最后一轮的工具 submit_only；满额之后的工具 submit_only；每轮状态提示 无；提示走的消息 status → tool、last_round → user；工作台选项 {'view_frames': False}
- 提示：configs/w1.json sha256 e4cc841907284009…（frozen = True）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune_countbench：n = 49（划分清单 49 条，指纹 3559c5aec49d23d3…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- accuracy：0.8367，95% CI [0.7347, 0.9388]，n = 49

## 回路

- 失败：{'ok': 49}；结束方式：{'submission': 49}；答案来源：{'submit_count': 49}
- 轮数：{'n': 49, 'min': 1, 'p05': 1.0, 'p50': 1.0, 'p95': 6.0, 'max': 6, 'mean': 2.163265306122449, 'n_distinct': 6, 'top': [[1, 27], [2, 10], [6, 5], [5, 3], [3, 2], [4, 2]]}
- 计数的工具调用：{'n': 49, 'min': 0, 'p05': 0.0, 'p50': 0.0, 'p95': 5.0, 'max': 5, 'mean': 1.163265306122449, 'n_distinct': 6, 'top': [[0, 27], [1, 10], [5, 5], [4, 3], [2, 2], [3, 2]]}
- 被拒的调用 0（{}）；解析失败的调用 0；工具报错 6
- 各工具执行次数：{'components': 7, 'draw_box': 1, 'edges': 9, 'grabcut': 3, 'morph': 1, 'run_python': 3, 'show': 2, 'submit_count': 49, 'superpixels': 1, 'tile_view': 11, 'zoom': 19}
- 思考段的结束方式（逐轮）：{'forced': 3, 'natural': 103}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
