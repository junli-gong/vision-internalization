# W1 · qwen3.5-9b · refseg · full · devtune

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 协议开关：最后一轮的工具 submit_only；满额之后的工具 submit_only；每轮状态提示 有；工作台选项 {'view_frames': False}
- 提示：configs/w1_tune/T2.json sha256 36c06a204cd359a1…（frozen = False）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune：n = 96（划分清单 300 条，指纹 21e50b3bb13e25be…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- cIoU：0.2406，95% CI [0.1631, 0.3265]，n = 96
- gIoU：0.2010，95% CI [0.1458, 0.2649]，n = 96

## 回路

- 失败：{'no_answer': 60, 'ok': 36}；结束方式：{'round_cap': 60, 'submission': 36}；答案来源：{'ok': 60, 'submit_mask': 36}
- 轮数：{'n': 96, 'min': 4, 'p05': 6.0, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 5.958333333333333, 'n_distinct': 2, 'top': [[6, 94], [4, 2]]}
- 计数的工具调用：{'n': 96, 'min': 3, 'p05': 5.0, 'p50': 5.0, 'p95': 5.0, 'max': 7, 'mean': 4.958333333333333, 'n_distinct': 4, 'top': [[5, 91], [3, 2], [4, 2], [7, 1]]}
- 被拒的调用 57（{'final_round': 57}）；解析失败的调用 3；工具报错 5
- 各工具执行次数：{'components': 24, 'draw_box': 5, 'draw_polygon': 2, 'edges': 12, 'grabcut': 84, 'refine': 28, 'run_python': 2, 'select_superpixels': 20, 'show': 86, 'submit_mask': 38, 'superpixels': 39, 'tile_view': 38, 'watershed': 1, 'zoom': 135}
- 思考段的结束方式（逐轮）：{'forced': 3, 'natural': 569}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
