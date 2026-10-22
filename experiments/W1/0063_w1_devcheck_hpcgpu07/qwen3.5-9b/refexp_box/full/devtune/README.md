# W1 · qwen3.5-9b · refexp_box · full · devtune

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 协议开关：最后一轮的工具 submit_only；满额之后的工具 submit_only；每轮状态提示 无；提示走的消息 status → tool、last_round → user；工作台选项 {'view_frames': False}
- 提示：configs/w1.json sha256 e4cc841907284009…（frozen = True）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune：n = 64（划分清单 300 条，指纹 21e50b3bb13e25be…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- acc@0.5：0.6250，95% CI [0.5000, 0.7344]，n = 64

## 回路

- 失败：{'no_answer': 1, 'ok': 63}；结束方式：{'round_cap': 1, 'submission': 63}；答案来源：{'ok': 1, 'submit_box': 63}
- 轮数：{'n': 64, 'min': 1, 'p05': 1.0, 'p50': 5.0, 'p95': 6.0, 'max': 6, 'mean': 4.359375, 'n_distinct': 6, 'top': [[6, 31], [2, 10], [3, 9], [1, 5], [5, 5], [4, 4]]}
- 计数的工具调用：{'n': 64, 'min': 0, 'p05': 0.0, 'p50': 4.0, 'p95': 5.0, 'max': 5, 'mean': 3.34375, 'n_distinct': 6, 'top': [[5, 30], [1, 10], [2, 9], [4, 6], [0, 5], [3, 4]]}
- 被拒的调用 0（{}）；解析失败的调用 1；工具报错 1
- 各工具执行次数：{'components': 4, 'draw_box': 5, 'edges': 8, 'grabcut': 13, 'read_pixels': 29, 'refine': 1, 'select_superpixels': 1, 'show': 13, 'submit_box': 64, 'superpixels': 1, 'tile_view': 14, 'watershed': 1, 'zoom': 124}
- 思考段的结束方式（逐轮）：{'forced': 20, 'natural': 259}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
