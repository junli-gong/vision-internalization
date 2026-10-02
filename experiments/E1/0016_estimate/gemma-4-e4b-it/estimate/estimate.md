# 吞吐与算力估算 · gemma-4-e4b-it · 口径 default

- {'model': 'gemma-4-e4b-it', 'revision': 'ee0ef6023621cff504d758262d4e04895a5af4a2', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'SoftTokenBudget', 'max_soft_tokens': 280}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 3.204 | 56857 | 4.930 | 200 | 0.035 | 50 | 0.018 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 64 | 16 | 3.502 | 8000 | 0.635 | 200 | 0.032 | 50 | 0.016 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 2.205 | 500 | 0.063 | 200 | 0.126 | 50 | 0.063 | — | — | — | — |
| count | text | 49 | 256 | 64 | 16 | 56.222 | 442 | 0.002 | 200 | 0.028 | 50 | 0.014 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 1.173 | 963 | 0.228 | 200 | 0.102 | 50 | 0.051 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 75.869 | 3270 | 0.012 | 200 | 0.058 | 50 | 0.029 | — | — | 200 | 0.055 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 60.382 | 112 | 0.001 | 112 | 0.096 | 50 | 0.085 | — | — | 112 | 0.038 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 45.883 | 2588 | 0.016 | 200 | 0.043 | 50 | 0.021 | — | — | 200 | 0.031 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 67.174 | 3590 | 0.015 | 200 | 0.088 | 50 | 0.044 | — | — | 200 | 0.086 |
| 合计 | | | | | | | | 5.901 | | 0.608 | | 0.342 | | — | | 0.211 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
