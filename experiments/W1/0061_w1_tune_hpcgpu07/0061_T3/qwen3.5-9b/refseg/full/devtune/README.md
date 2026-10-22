# W1 · qwen3.5-9b · refseg · full · devtune

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 协议开关：最后一轮的工具 submit_only；满额之后的工具 submit_only；每轮状态提示 有；工作台选项 {'view_frames': False}
- 提示：configs/w1_tune/T3.json sha256 16e7a6f3ae66355b…（frozen = False）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune：n = 96（划分清单 300 条，指纹 21e50b3bb13e25be…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- cIoU：0.3130，95% CI [0.2237, 0.3894]，n = 96
- gIoU：0.1926，95% CI [0.1406, 0.2475]，n = 96

## 回路

- 失败：{'no_answer': 54, 'ok': 41, 'wrong_submission': 1}；结束方式：{'round_cap': 54, 'submission': 42}；答案来源：{'ok': 54, 'submit_box': 1, 'submit_mask': 41}
- 轮数：{'n': 96, 'min': 5, 'p05': 6.0, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 5.989583333333333, 'n_distinct': 2, 'top': [[6, 95], [5, 1]]}
- 计数的工具调用：{'n': 96, 'min': 4, 'p05': 5.0, 'p50': 5.0, 'p95': 5.0, 'max': 6, 'mean': 5.0, 'n_distinct': 3, 'top': [[5, 94], [4, 1], [6, 1]]}
- 被拒的调用 51（{'final_round': 51}）；解析失败的调用 0；工具报错 5
- 各工具执行次数：{'components': 22, 'draw_box': 2, 'draw_points': 1, 'draw_polygon': 1, 'edges': 13, 'grabcut': 80, 'morph': 1, 'refine': 24, 'run_python': 2, 'select_superpixels': 27, 'show': 92, 'submit_box': 1, 'submit_mask': 44, 'superpixels': 41, 'tile_view': 50, 'zoom': 124}
- 思考段的结束方式（逐轮）：{'forced': 13, 'natural': 562}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
