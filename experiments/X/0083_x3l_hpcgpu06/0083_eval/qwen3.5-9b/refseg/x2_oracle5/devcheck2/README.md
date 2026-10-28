# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7541378869656965（95% CI {'low': 0.7267439898390214, 'high': 0.7816853313839079, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7631479613362614（95% CI {'low': 0.7446008328657503, 'high': 0.7807716024971797, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
