# W1 · qwen3.5-9b · refseg · full · devtune

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 协议开关：最后一轮的工具 submit_only；满额之后的工具 submit_only；每轮状态提示 无；提示走的消息 status → tool、last_round → user；工作台选项 {'view_frames': False}
- 提示：configs/w1_tune/U2.json sha256 8cf0fe8ce06c5d90…（frozen = False）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune：n = 96（划分清单 300 条，指纹 21e50b3bb13e25be…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- cIoU：0.3946，95% CI [0.3215, 0.4667]，n = 96
- gIoU：0.3838，95% CI [0.3229, 0.4420]，n = 96

## 回路

- 失败：{'no_answer': 19, 'ok': 74, 'wrong_submission': 3}；结束方式：{'round_cap': 19, 'submission': 77}；答案来源：{'ok': 19, 'submit_box': 3, 'submit_mask': 74}
- 轮数：{'n': 96, 'min': 5, 'p05': 6.0, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 5.989583333333333, 'n_distinct': 2, 'top': [[6, 95], [5, 1]]}
- 计数的工具调用：{'n': 96, 'min': 4, 'p05': 5.0, 'p50': 5.0, 'p95': 5.0, 'max': 7, 'mean': 5.010416666666667, 'n_distinct': 4, 'top': [[5, 92], [4, 2], [6, 1], [7, 1]]}
- 被拒的调用 9（{'final_round': 9}）；解析失败的调用 1；工具报错 13
- 各工具执行次数：{'components': 12, 'delete_mask': 3, 'draw_box': 3, 'draw_points': 5, 'edges': 15, 'grabcut': 80, 'read_pixels': 3, 'refine': 23, 'run_python': 2, 'select_superpixels': 26, 'show': 89, 'submit_box': 3, 'submit_mask': 84, 'superpixels': 38, 'tile_view': 41, 'watershed': 4, 'zoom': 137}
- 思考段的结束方式（逐轮）：{'forced': 27, 'natural': 548}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
