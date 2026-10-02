# 吞吐与算力估算 · qwen3.5-0.8b · 口径 default

- {'model': 'qwen3.5-0.8b', 'revision': '2fc06364715b967f1860aea9cf38778875588b17', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 11.316 | 56857 | 1.396 | 200 | 0.066 | 50 | 0.033 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 64 | 16 | 8.296 | 8000 | 0.268 | 200 | 0.066 | 50 | 0.033 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 1.146 | 500 | 0.121 | 200 | 0.066 | 50 | 0.033 | — | — | — | — |
| count | text | 49 | 256 | 32 | 16 | 73.498 | 442 | 0.002 | 200 | 0.075 | 50 | 0.037 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.565 | 963 | 0.474 | 200 | 0.068 | 50 | 0.034 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 243.346 | 3270 | 0.004 | 200 | 0.043 | 50 | 0.022 | — | — | 200 | 0.025 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 186.004 | 112 | 0.000 | 112 | 0.136 | 50 | 0.122 | — | — | 112 | 0.033 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 23.147 | 2588 | 0.031 | 200 | 0.076 | 50 | 0.038 | — | — | 200 | 0.024 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 200.512 | 3590 | 0.005 | 200 | 0.076 | 50 | 0.038 | — | — | 200 | 0.015 |
| 合计 | | | | | | | | 2.300 | | 0.672 | | 0.389 | | — | | 0.098 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
