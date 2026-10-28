# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7328234095774389（95% CI {'low': 0.7057255296213368, 'high': 0.7610974514853377, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7415931314919394（95% CI {'low': 0.7238828383327879, 'high': 0.75945655086889, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 499, 'parse_failure': 1}；每句模型调用 2.0
