# X-2 · qwen3.5-9b · refseg · x2_select_cor1_randbox · devtune800

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5849008401192144（95% CI {'low': 0.5565887388220542, 'high': 0.6117581100639858, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5751243870840815（95% CI {'low': 0.5538363604247791, 'high': 0.5946931280809878, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
