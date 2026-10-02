# 吞吐与算力估算 · qwen3.5-0.8b · 口径 qwen:1024

- {'model': 'qwen3.5-0.8b', 'revision': '2fc06364715b967f1860aea9cf38778875588b17', 'policy': 'qwen:1024', 'count_questions': 'heuristic', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 1024, 'min_pixels': 1048576, 'max_pixels': 1048576}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': 'e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6', 'batch_sizes': [8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 256 | 32 | 14.929 | 56857 | — | 200 | — | 50 | — | 500 | 0.009 |
| ovd | 64 | 2048 | 32 | 1.107 | 963 | — | 200 | — | 50 | — | 100 | 0.025 |
| 合计 | | | | | | — | | — | | — | | 0.034 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
