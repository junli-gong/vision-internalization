# X-2 · qwen3.5-9b · refseg · x2_spline_raster · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7101796650954397（95% CI {'low': 0.6835364073881622, 'high': 0.7374005930830367, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7187162009613225（95% CI {'low': 0.7013256987762772, 'high': 0.7360640225232674, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 499, 'parse_failure': 1}；每句模型调用 2.0
