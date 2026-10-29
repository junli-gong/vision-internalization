# X-2 · qwen3.5-9b · refseg · x2_select5_cor1_randbox · devcheck2

- 定义：{"kind": "randbox", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select5", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6803929740445247（95% CI {'low': 0.6470395229852269, 'high': 0.7125991269501658, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6722062331016102（95% CI {'low': 0.6458706154936259, 'high': 0.6977139924443698, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 4.0
