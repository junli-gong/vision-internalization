# 吞吐与算力估算 · qwen3.5-2b · 口径 default

- {'model': 'qwen3.5-2b', 'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc', 'policy': 'default', 'count_questions': 'heuristic', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': 'e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6', 'batch_sizes': [8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 256 | 32 | 17.566 | 56857 | 0.899 | 200 | 0.015 | 50 | 0.007 | 500 | — |
| refexp_poly | 64 | 1024 | 32 | 15.329 | 8000 | 0.145 | 200 | 0.378 | 50 | 0.189 | — | — |
| grid16 | 64 | 1024 | 32 | 2.267 | 500 | 0.061 | 200 | 0.397 | 50 | 0.198 | — | — |
| count | 49 | 256 | 32 | 142.998 | 442 | 0.001 | 200 | 0.248 | 50 | 0.124 | — | — |
| ovd | 64 | 2048 | 32 | 1.147 | 963 | 0.233 | 200 | 0.262 | 50 | 0.131 | 100 | — |
| rel_depth_nyu | 64 | 1024 | 32 | 4.678 | 3270 | 0.194 | 200 | 0.383 | 50 | 0.191 | — | — |
| rel_depth_blink | 12 | 1024 | 32 | 1.307 | 112 | 0.024 | 112 | 0.224 | 50 | 0.200 | — | — |
| spatial_cvbench | 50 | 1024 | 32 | 2.416 | 2588 | 0.298 | 200 | 0.400 | 50 | 0.200 | — | — |
| spatial_embspatial | 50 | 1024 | 32 | 5.486 | 3590 | 0.182 | 200 | 0.398 | 50 | 0.199 | — | — |
| 合计 | | | | | | 2.037 | | 2.704 | | 1.440 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
