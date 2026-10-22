# W1 · qwen3.5-9b · refseg · full · devcheck

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 协议开关：最后一轮的工具 submit_only；满额之后的工具 submit_only；每轮状态提示 无；提示走的消息 status → tool、last_round → user；工作台选项 {'view_frames': False}
- 提示：configs/w1.json sha256 e4cc841907284009…（frozen = True）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devcheck：n = 500（划分清单 500 条，指纹 e947f25f58c98ad9…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- cIoU：0.4696，95% CI [0.4367, 0.5002]，n = 500
- gIoU：0.4337，95% CI [0.4103, 0.4604]，n = 500

## 回路

- 失败：{'no_answer': 58, 'ok': 427, 'wrong_submission': 15}；结束方式：{'round_cap': 58, 'submission': 442}；答案来源：{'ok': 58, 'submit_box': 14, 'submit_count': 1, 'submit_mask': 427}
- 轮数：{'n': 500, 'min': 3, 'p05': 6.0, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 5.976, 'n_distinct': 4, 'top': [[6, 492], [5, 5], [4, 2], [3, 1]]}
- 计数的工具调用：{'n': 500, 'min': 2, 'p05': 5.0, 'p50': 5.0, 'p95': 5.0, 'max': 7, 'mean': 4.976, 'n_distinct': 6, 'top': [[5, 486], [4, 7], [2, 2], [3, 2], [7, 2], [6, 1]]}
- 被拒的调用 33（{'after_submission': 1, 'final_round': 32}）；解析失败的调用 6；工具报错 35
- 各工具执行次数：{'combine_masks': 1, 'components': 107, 'delete_mask': 1, 'draw_box': 28, 'draw_points': 2, 'draw_polygon': 4, 'edges': 83, 'grabcut': 424, 'morph': 6, 'read_pixels': 13, 'refine': 133, 'run_python': 8, 'select_superpixels': 108, 'show': 469, 'submit_box': 14, 'submit_count': 1, 'submit_mask': 452, 'superpixels': 175, 'tile_view': 221, 'watershed': 4, 'zoom': 701}
- 思考段的结束方式（逐轮）：{'forced': 114, 'natural': 2874}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
