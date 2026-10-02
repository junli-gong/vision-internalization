# 吞吐与算力估算 · internvl3.5-2b-hf · 口径 internvl:native

- {'model': 'internvl3.5-2b-hf', 'revision': '3f301ffcf3dcbb47893afae6650ea3e78d96fb6d', 'policy': 'internvl:native', 'count_questions': 'heuristic', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': True, 'max_patches': 12}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': 'e9fade2b64ecf337db03585ef3206ed8b90c523e4bec2b42dae8f0b6890744c6', 'batch_sizes': [8, 16, 32], 'gpu': 'NVIDIA B200'}

| 任务 | 开发样本 | max_new_tokens | 推荐批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | 64 | 256 | 32 | 10.178 | 56857 | — | 200 | — | 50 | — | 500 | 0.014 |
| ovd | 64 | 2048 | 8 | 0.368 | 963 | — | 200 | — | 50 | — | 100 | 0.075 |
| 合计 | | | | | | — | | — | | — | | 0.089 |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
