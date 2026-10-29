# X-2 · qwen3.5-9b · refseg · x2_select_cor3_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.601293754273547（95% CI {'low': 0.5715090033238606, 'high': 0.6319379382788176, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6114746970219515（95% CI {'low': 0.588750474598505, 'high': 0.6353143700634896, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
