# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7573112637204303（95% CI {'low': 0.7365599282956815, 'high': 0.7770546323197245, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7600192150739842（95% CI {'low': 0.7447885986534785, 'high': 0.7737437233419785, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
