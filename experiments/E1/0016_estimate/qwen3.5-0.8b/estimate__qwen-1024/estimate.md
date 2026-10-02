# 吞吐与算力估算 · qwen3.5-0.8b · 口径 qwen:1024

- {'model': 'qwen3.5-0.8b', 'revision': '2fc06364715b967f1860aea9cf38778875588b17', 'policy': 'qwen:1024', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 1024, 'min_pixels': 1048576, 'max_pixels': 1048576}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 10.460 | 56857 | — | 200 | — | 50 | — | 500 | 0.013 | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.592 | 963 | — | 200 | — | 50 | — | 100 | 0.047 | — | — |
| 合计 | | | | | | | | — | | — | | — | | 0.060 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
