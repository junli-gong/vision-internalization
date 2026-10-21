# W1 · qwen3.5-9b · refseg · full · devtune

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 协议：至多 6 轮、8 次计数的工具调用（提交不计数：True），每轮思考 2048、作答 1024 token；关掉的工具组 无
- 提示：configs/w1.json sha256 6a4704590d47abf7…（frozen = False）；工具清单 sha256 dcc91402712dd16d…（22 个工具）
- 划分 devtune：n = 32（划分清单 300 条，指纹 21e50b3bb13e25be…）；批 16（行数 × 提示宽度 ≤ 48000），组 64；预检拒收 0 条

## 标题指标

- cIoU：0.1585，95% CI [0.0000, 0.3794]，n = 32
- gIoU：0.0540，95% CI [0.0000, 0.1351]，n = 32

## 回路

- 失败：{'no_answer': 30, 'ok': 2}；结束方式：{'round_cap': 30, 'submission': 2}；答案来源：{'ok': 30, 'submit_mask': 2}
- 轮数：{'n': 32, 'min': 6, 'p05': 6.0, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 6.0, 'n_distinct': 1, 'top': [[6, 32]]}
- 计数的工具调用：{'n': 32, 'min': 5, 'p05': 5.55, 'p50': 6.0, 'p95': 6.0, 'max': 6, 'mean': 5.9375, 'n_distinct': 2, 'top': [[6, 30], [5, 2]]}
- 被拒的调用 0（{}）；解析失败的调用 0；工具报错 3
- 各工具执行次数：{'combine_masks': 1, 'components': 6, 'draw_box': 3, 'edges': 7, 'grabcut': 27, 'read_pixels': 4, 'refine': 6, 'run_python': 2, 'select_superpixels': 16, 'show': 37, 'submit_mask': 2, 'superpixels': 19, 'tile_view': 13, 'watershed': 1, 'zoom': 48}
- 思考段的结束方式（逐轮）：{'natural': 192}

## 文件

- predictions.jsonl：逐条（每轮原文、工具调用与结果摘要、最终答案、得分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、workbench.jsonl、run_*.json）。
