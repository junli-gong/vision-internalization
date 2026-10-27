# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6676122516593479（95% CI {'low': 0.6423679468998087, 'high': 0.6923184215722914, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6467479123497744（95% CI {'low': 0.6263551394030408, 'high': 0.6675533551339196, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 6.0
