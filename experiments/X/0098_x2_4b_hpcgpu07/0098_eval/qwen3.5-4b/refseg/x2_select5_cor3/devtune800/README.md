# X-2 · qwen3.5-4b · refseg · x2_select5_cor3 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6242555689151121（95% CI {'low': 0.5947082203514173, 'high': 0.6538184726642683, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6126333267038712（95% CI {'low': 0.5906188746938072, 'high': 0.6347025250727929, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 6.0
