# 吞吐与算力估算 · internvl3.5-2b-hf · 口径 default

- {'model': 'internvl3.5-2b-hf', 'revision': '3f301ffcf3dcbb47893afae6650ea3e78d96fb6d', 'policy': 'default', 'count_questions': 'heuristic', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': 'e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6', 'batch_sizes': [8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 256 | 32 | 31.891 | 56857 | 0.495 | 200 | 1.093 | 50 | 0.547 | 500 | — |
| refexp_poly | 64 | 1024 | 32 | 1.069 | 8000 | 2.078 | 200 | 0.424 | 50 | 0.212 | — | — |
| grid16 | 64 | 1024 | 32 | 8.980 | 500 | 0.015 | 200 | 0.417 | 50 | 0.209 | — | — |
| count | 49 | 256 | 16 | 4.833 | 442 | 0.025 | 200 | 0.412 | 50 | 0.206 | — | — |
| ovd | 64 | 2048 | 8 | 1.403 | 963 | 0.191 | 200 | 0.899 | 50 | 0.450 | 100 | — |
| rel_depth_nyu | 64 | 1024 | 32 | 6.397 | 3270 | 0.142 | 200 | 0.286 | 50 | 0.143 | — | — |
| rel_depth_blink | 12 | 1024 | 32 | 3.336 | 112 | 0.009 | 112 | 0.160 | 50 | 0.143 | — | — |
| spatial_cvbench | 50 | 1024 | 16 | 4.349 | 2588 | 0.165 | 200 | 0.407 | 50 | 0.204 | — | — |
| spatial_embspatial | 50 | 1024 | 32 | 6.062 | 3590 | 0.164 | 200 | 0.408 | 50 | 0.204 | — | — |
| 合计 | | | | | | 3.286 | | 4.507 | | 2.316 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
