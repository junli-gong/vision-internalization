# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6415605964843168（95% CI {'low': 0.6121573426617524, 'high': 0.6746113600343187, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6339449808193879（95% CI {'low': 0.6096356113720328, 'high': 0.6613944051200629, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
