# 吞吐与算力估算 · gemma-4-e2b-it · 口径 default

- {'model': 'gemma-4-e2b-it', 'revision': '3e22461f65e89153144f8adb70e3b8c2cc9845a7', 'policy': 'default', 'count_questions': 'heuristic', 'image_policy': {'type': 'SoftTokenBudget', 'max_soft_tokens': 280}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': 'e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6', 'batch_sizes': [8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 256 | 32 | 6.035 | 56857 | 2.617 | 200 | 0.100 | 50 | 0.050 | 500 | — |
| refexp_poly | 64 | 1024 | 32 | 4.449 | 8000 | 0.500 | 200 | 0.030 | 50 | 0.015 | — | — |
| grid16 | 64 | 1024 | 32 | 6.069 | 500 | 0.023 | 200 | 0.122 | 50 | 0.061 | — | — |
| count | 49 | 256 | 32 | 77.693 | 442 | 0.002 | 200 | 0.085 | 50 | 0.042 | — | — |
| ovd | 64 | 2048 | 32 | 4.078 | 963 | 0.066 | 200 | 0.160 | 50 | 0.080 | 100 | — |
| rel_depth_nyu | 64 | 1024 | 32 | 2.382 | 3270 | 0.381 | 200 | 0.121 | 50 | 0.060 | — | — |
| rel_depth_blink | 12 | 1024 | 32 | 1.328 | 112 | 0.023 | 112 | 0.096 | 50 | 0.085 | — | — |
| spatial_cvbench | 50 | 1024 | 32 | 1.760 | 2588 | 0.408 | 200 | 0.152 | 50 | 0.076 | — | — |
| spatial_embspatial | 50 | 1024 | 32 | 1.552 | 3590 | 0.643 | 200 | 0.279 | 50 | 0.139 | — | — |
| 合计 | | | | | | 4.663 | | 1.142 | | 0.609 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
