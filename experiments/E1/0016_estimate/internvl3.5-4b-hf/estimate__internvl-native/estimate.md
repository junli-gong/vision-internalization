# 吞吐与算力估算 · internvl3.5-4b-hf · 口径 internvl:native

- {'model': 'internvl3.5-4b-hf', 'revision': '6bd4487402110ef9889ba50eb7aefeb302526fed', 'policy': 'internvl:native', 'count_questions': 'countbenchqa', 'image_policy': {'type': 'TilePolicy', 'crop_to_patches': True, 'max_patches': 12}, 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '4dd027b4c63098f7b1f9feb3dd64b5ed6a2c1b8e9571c52b28db86f2098c39d2', 'batch_sizes': [16, 32, 64], 'frozen_batch_size': 16, 'gpu': 'NVIDIA B200'}

| 任务 | 读法 | 开发样本 | max_new_tokens | 推荐批大小 | 外推用的批大小 | 样本/秒 | n(eval) | GPU 小时(eval) | n(think) | GPU 小时(think) | n(think8192) | GPU 小时(8192, 上界) | n(extra) | GPU 小时(extra) | n(free) | GPU 小时(free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | text | 64 | 256 | 32 | 16 | 4.815 | 56857 | — | 200 | — | 50 | — | 500 | 0.029 | — | — |
| ovd | text | 64 | 2048 | 64 | 16 | 0.129 | 963 | — | 200 | — | 50 | — | 100 | 0.216 | — | — |
| 合计 | | | | | | | | — | | — | | — | | 0.245 | | — |

逐批大小的明细（样本/秒、token/秒、峰值显存、截断率）见 estimate.json。
