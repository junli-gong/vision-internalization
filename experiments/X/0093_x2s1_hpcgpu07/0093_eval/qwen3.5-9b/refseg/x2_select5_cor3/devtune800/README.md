# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6468964506783696（95% CI {'low': 0.6139768794290857, 'high': 0.6757486973702954, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6355677158558147（95% CI {'low': 0.6136859071686651, 'high': 0.6575893800817131, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 6.0
