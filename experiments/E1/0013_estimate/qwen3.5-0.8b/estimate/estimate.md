# 吞吐与算力估算 · qwen3.5-0.8b · 口径 default

- {'model': 'qwen3.5-0.8b', 'revision': '2fc06364715b967f1860aea9cf38778875588b17', 'policy': 'default', 'count_questions': 'heuristic', 'image_policy': {'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': 'e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6', 'batch_sizes': [8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 256 | 32 | 17.076 | 56857 | 0.925 | 200 | 0.012 | 50 | 0.006 | 500 | — |
| refexp_poly | 64 | 1024 | 32 | 12.496 | 8000 | 0.178 | 200 | 0.368 | 50 | 0.184 | — | — |
| grid16 | 64 | 1024 | 32 | 2.255 | 500 | 0.062 | 200 | 0.374 | 50 | 0.187 | — | — |
| count | 49 | 256 | 32 | 128.814 | 442 | 0.001 | 200 | 0.393 | 50 | 0.196 | — | — |
| ovd | 64 | 2048 | 32 | 1.168 | 963 | 0.229 | 200 | 0.368 | 50 | 0.184 | 100 | — |
| rel_depth_nyu | 64 | 1024 | 32 | 3.284 | 3270 | 0.277 | 200 | 0.351 | 50 | 0.176 | — | — |
| rel_depth_blink | 12 | 1024 | 8 | 1.292 | 112 | 0.024 | 112 | 0.205 | 50 | 0.183 | — | — |
| spatial_cvbench | 50 | 1024 | 32 | 2.667 | 2588 | 0.270 | 200 | 0.392 | 50 | 0.196 | — | — |
| spatial_embspatial | 50 | 1024 | 32 | 6.567 | 3590 | 0.152 | 200 | 0.375 | 50 | 0.187 | — | — |
| 合计 | | | | | | 2.116 | | 2.837 | | 1.499 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
