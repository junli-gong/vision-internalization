# X-2 · qwen3.5-9b · refseg · x2_select5_cor2 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7040851053155505（95% CI {'low': 0.6743343183752016, 'high': 0.7376787030559584, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7038251066371074（95% CI {'low': 0.6815025510891835, 'high': 0.7275748246783897, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 5.0
