# 吞吐与算力估算 · internvl3.5-2b-hf · 口径 default

- {'model': 'internvl3.5-2b-hf', 'revision': '3f301ffcf3dcbb47893afae6650ea3e78d96fb6d', 'policy': 'default', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 64 | 16 | 22.968 | 56857 | 0.688 | 200 | 0.275 | 50 | 0.138 | 500 | — | — | — |
| refexp_poly | text | 64 | 1024 | 64 | 16 | 1.055 | 8000 | 2.106 | 200 | 0.200 | 50 | 0.100 | — | — | — | — |
| grid16 | text | 64 | 1024 | 64 | 16 | 5.120 | 500 | 0.027 | 200 | 0.200 | 50 | 0.100 | — | — | — | — |
| count | text | 49 | 256 | 16 | 16 | 7.300 | 442 | 0.017 | 200 | 0.372 | 50 | 0.186 | — | — | — | — |
| ovd | text | 64 | 2048 | 32 | 16 | 0.833 | 963 | 0.321 | 200 | 0.132 | 50 | 0.066 | 100 | — | — | — |
| rel_depth_nyu | letter | 64 | 1 | 64 | 16 | 255.584 | 3270 | 0.004 | 200 | 0.097 | 50 | 0.049 | — | — | 200 | 0.012 |
| rel_depth_blink | letter | 12 | 1 | 64 | 16 | 120.661 | 112 | 0.000 | 112 | 0.327 | 50 | 0.292 | — | — | 112 | 0.030 |
| spatial_cvbench | letter | 50 | 1 | 64 | 16 | 69.924 | 2588 | 0.010 | 200 | 0.305 | 50 | 0.152 | — | — | 200 | 0.020 |
| spatial_embspatial | letter | 50 | 1 | 64 | 16 | 166.366 | 3590 | 0.006 | 200 | 0.209 | 50 | 0.104 | — | — | 200 | 0.011 |
| 合计 | | | | | | | | 3.178 | | 2.118 | | 1.187 | | — | | 0.073 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
