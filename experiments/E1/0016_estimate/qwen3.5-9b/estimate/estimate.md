# 吞吐与算力估算 · qwen3.5-9b · 口径 default

- {'model': 'qwen3.5-9b', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 9.152 | 56857 | 1.726 | 200 | 0.009 | 50 | 0.004 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 32 | 16 | 2.578 | 8000 | 0.862 | 200 | 0.061 | 50 | 0.031 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 3.031 | 500 | 0.046 | 200 | 0.143 | 50 | 0.071 | — | — | — | — |
| count | text | 49 | 256 | 64 | 16 | 48.326 | 442 | 0.003 | 200 | 0.157 | 50 | 0.079 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.418 | 963 | 0.639 | 200 | 0.147 | 50 | 0.074 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 82.746 | 3270 | 0.011 | 200 | 0.065 | 50 | 0.032 | — | — | 200 | 0.027 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 69.152 | 112 | 0.000 | 112 | 0.186 | 50 | 0.166 | — | — | 112 | 0.017 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 53.155 | 2588 | 0.014 | 200 | 0.157 | 50 | 0.079 | — | — | 200 | 0.035 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 72.371 | 3590 | 0.014 | 200 | 0.158 | 50 | 0.079 | — | — | 200 | 0.037 |
| 合计 | | | | | | | | 3.314 | | 1.085 | | 0.616 | | — | | 0.116 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
