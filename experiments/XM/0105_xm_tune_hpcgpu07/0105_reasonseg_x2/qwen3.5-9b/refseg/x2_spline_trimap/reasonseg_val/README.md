# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · reasonseg_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/reasonseg_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5145358927192284（95% CI {'low': 0.43987188381806647, 'high': 0.5895310113416595, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.539585173281467（95% CI {'low': 0.49627142590538315, 'high': 0.5829992063834936, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 196}；fallback {'ok': 196}；每句模型调用 2.0
