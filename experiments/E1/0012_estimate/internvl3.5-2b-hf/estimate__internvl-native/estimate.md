# 吞吐与算力估算 · internvl3.5-2b-hf · 口径 internvl:native

- {'model': 'internvl3.5-2b-hf', 'revision': '3f301ffcf3dcbb47893afae6650ea3e78d96fb6d', 'policy': 'internvl:native', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': True, 'max_patches': 12}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '974d3d730e903a6f0a0727a13fceaa52437b3a048914b90836f8b918afcdc0fa', 'batch_sizes': [1, 8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 128 | 32 | 10.176 | 56857 | — | 200 | — | 50 | — | 500 | 0.014 |
| ovd | 64 | 512 | 32 | 1.060 | 963 | — | 200 | — | 50 | — | 100 | 0.026 |
| 合计 | | | | | | — | | — | | — | | 0.040 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
