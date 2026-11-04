# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6830296220587255（95% CI {'low': 0.6528967397594606, 'high': 0.7187379212853932, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6782406104495312（95% CI {'low': 0.6538119872486695, 'high': 0.7035929361329972, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 6.0
