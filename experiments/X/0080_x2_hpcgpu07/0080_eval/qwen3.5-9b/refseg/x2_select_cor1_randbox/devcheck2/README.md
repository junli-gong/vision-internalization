# X-2 · qwen3.5-9b · refseg · x2_select_cor1_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5858740105075078（95% CI {'low': 0.55492772174827, 'high': 0.6163786915789549, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5999577571576469（95% CI {'low': 0.575340894177979, 'high': 0.6239233149976773, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
