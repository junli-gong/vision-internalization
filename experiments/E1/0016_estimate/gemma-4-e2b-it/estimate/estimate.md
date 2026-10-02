# 吞吐与算力估算 · gemma-4-e2b-it · 口径 default

- {'model': 'gemma-4-e2b-it', 'revision': '3e22461f65e89153144f8adb70e3b8c2cc9845a7', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'SoftTokenBudget', 'max_soft_tokens': 280}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 4.014 | 56857 | 3.935 | 200 | 0.015 | 50 | 0.007 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 64 | 16 | 2.872 | 8000 | 0.774 | 200 | 0.076 | 50 | 0.038 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 3.025 | 500 | 0.046 | 200 | 0.018 | 50 | 0.009 | — | — | — | — |
| count | text | 49 | 256 | 64 | 16 | 64.874 | 442 | 0.002 | 200 | 0.023 | 50 | 0.011 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 2.128 | 963 | 0.126 | 200 | 0.023 | 50 | 0.011 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 91.748 | 3270 | 0.010 | 200 | 0.026 | 50 | 0.013 | — | — | 200 | 0.035 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 69.713 | 112 | 0.000 | 112 | 0.068 | 50 | 0.061 | — | — | 112 | 0.023 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 53.078 | 2588 | 0.014 | 200 | 0.026 | 50 | 0.013 | — | — | 200 | 0.053 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 83.179 | 3590 | 0.012 | 200 | 0.048 | 50 | 0.024 | — | — | 200 | 0.070 |
| 合计 | | | | | | | | 4.918 | | 0.322 | | 0.187 | | — | | 0.180 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
