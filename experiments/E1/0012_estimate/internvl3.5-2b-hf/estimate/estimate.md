# 吞吐与算力估算 · internvl3.5-2b-hf · 口径 default

- {'model': 'internvl3.5-2b-hf', 'revision': '3f301ffcf3dcbb47893afae6650ea3e78d96fb6d', 'policy': 'default', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa', 'batch_sizes': [1, 8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 128 | 32 | 30.513 | 56857 | 0.518 | 200 | 1.100 | 50 | 0.550 | 500 | — |
| refexp_poly | 64 | 1024 | 32 | 1.075 | 8000 | 2.068 | 200 | 0.425 | 50 | 0.213 | — | — |
| grid16 | 64 | 1024 | 32 | 8.976 | 500 | 0.015 | 200 | 0.418 | 50 | 0.209 | — | — |
| count | 49 | 64 | 16 | 16.866 | 442 | 0.007 | 200 | 0.408 | 50 | 0.204 | — | — |
| ovd | 64 | 512 | 32 | 2.831 | 963 | 0.094 | 200 | 0.880 | 50 | 0.440 | 100 | — |
| rel_depth_nyu | 64 | 64 | 32 | 21.654 | 3270 | 0.042 | 200 | 0.296 | 50 | 0.148 | — | — |
| rel_depth_blink | 12 | 64 | 32 | 17.142 | 112 | 0.002 | 112 | 0.166 | 50 | 0.148 | — | — |
| spatial_cvbench | 50 | 64 | 16 | 16.248 | 2588 | 0.044 | 200 | 0.409 | 50 | 0.204 | — | — |
| spatial_embspatial | 50 | 64 | 32 | 22.462 | 3590 | 0.044 | 200 | 0.411 | 50 | 0.205 | — | — |
| 合计 | | | | | | 2.835 | | 4.514 | | 2.322 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
