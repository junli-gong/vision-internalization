# 吞吐与算力估算 · qwen3.5-2b · 口径 default

- {'model': 'qwen3.5-2b', 'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 12.503 | 56857 | 1.263 | 200 | 0.007 | 50 | 0.004 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 64 | 16 | 11.231 | 8000 | 0.198 | 200 | 0.068 | 50 | 0.034 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 1.200 | 500 | 0.116 | 200 | 0.068 | 50 | 0.034 | — | — | — | — |
| count | text | 49 | 256 | 64 | 16 | 72.901 | 442 | 0.002 | 200 | 0.078 | 50 | 0.039 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.596 | 963 | 0.449 | 200 | 0.070 | 50 | 0.035 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 160.660 | 3270 | 0.006 | 200 | 0.045 | 50 | 0.023 | — | — | 200 | 0.033 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 130.837 | 112 | 0.000 | 112 | 0.138 | 50 | 0.123 | — | — | 112 | 0.023 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 81.315 | 2588 | 0.009 | 200 | 0.078 | 50 | 0.039 | — | — | 200 | 0.026 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 137.614 | 3590 | 0.007 | 200 | 0.079 | 50 | 0.039 | — | — | 200 | 0.015 |
| 合计 | | | | | | | | 2.049 | | 0.632 | | 0.371 | | — | | 0.096 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
