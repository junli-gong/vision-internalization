# 吞吐与算力估算 · qwen3.5-4b · 口径 default

- {'model': 'qwen3.5-4b', 'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 32 | 16 | 7.580 | 56857 | 2.084 | 200 | 0.009 | 50 | 0.004 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 16 | 16 | 11.279 | 8000 | 0.197 | 200 | 0.278 | 50 | 0.139 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 3.331 | 500 | 0.042 | 200 | 0.138 | 50 | 0.069 | — | — | — | — |
| count | text | 49 | 256 | 16 | 16 | 54.281 | 442 | 0.002 | 200 | 0.273 | 50 | 0.136 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.444 | 963 | 0.602 | 200 | 0.142 | 50 | 0.071 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 106.089 | 3270 | 0.009 | 200 | 0.061 | 50 | 0.030 | — | — | 200 | 0.035 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 87.659 | 112 | 0.000 | 112 | 0.194 | 50 | 0.173 | — | — | 112 | 0.013 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 20.111 | 2588 | 0.036 | 200 | 0.157 | 50 | 0.079 | — | — | 200 | 0.033 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 91.795 | 3590 | 0.011 | 200 | 0.151 | 50 | 0.076 | — | — | 200 | 0.059 |
| 合计 | | | | | | | | 2.982 | | 1.403 | | 0.777 | | — | | 0.140 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
