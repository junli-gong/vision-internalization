# X-2 · qwen3.5-9b · refseg · x2_select · devcheck2

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6446891072713714（95% CI {'low': 0.6186198900638621, 'high': 0.6715039063136861, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.668768121631569（95% CI {'low': 0.6507043399204555, 'high': 0.6875932187472511, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
