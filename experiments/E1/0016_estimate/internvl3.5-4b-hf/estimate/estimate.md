# 吞吐与算力估算 · internvl3.5-4b-hf · 口径 default

- {'model': 'internvl3.5-4b-hf', 'revision': '6bd4487402110ef9889ba50eb7aefeb302526fed', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 19.044 | 56857 | 0.829 | 200 | 0.482 | 50 | 0.241 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 64 | 16 | 0.714 | 8000 | 3.111 | 200 | 0.418 | 50 | 0.209 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 3.422 | 500 | 0.041 | 200 | 0.419 | 50 | 0.210 | — | — | — | — |
| count | text | 49 | 256 | 64 | 16 | 36.430 | 442 | 0.003 | 200 | 0.519 | 50 | 0.260 | — | — | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.400 | 963 | 0.669 | 200 | 0.123 | 50 | 0.061 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 164.053 | 3270 | 0.006 | 200 | 0.079 | 50 | 0.040 | — | — | 200 | 0.017 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 94.972 | 112 | 0.000 | 112 | 0.368 | 50 | 0.329 | — | — | 112 | 0.028 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 57.729 | 2588 | 0.012 | 200 | 0.524 | 50 | 0.262 | — | — | 200 | 0.025 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 114.496 | 3590 | 0.009 | 200 | 0.432 | 50 | 0.216 | — | — | 200 | 0.007 |
| 合计 | | | | | | | | 4.680 | | 3.364 | | 1.827 | | — | | 0.077 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
