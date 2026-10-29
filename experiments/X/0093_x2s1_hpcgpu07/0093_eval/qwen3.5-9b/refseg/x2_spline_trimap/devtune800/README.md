# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7336381416963781（95% CI {'low': 0.7107720403451596, 'high': 0.7554466069335116, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7346369511049039（95% CI {'low': 0.7185395857719905, 'high': 0.7495518367743514, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
