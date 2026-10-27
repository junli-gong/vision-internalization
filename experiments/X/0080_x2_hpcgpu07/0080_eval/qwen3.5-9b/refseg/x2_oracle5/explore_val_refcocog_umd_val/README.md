# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7333002771545719（95% CI {'low': 0.6915651977914123, 'high': 0.7739265825701959, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7248701516272262（95% CI {'low': 0.6944819677983723, 'high': 0.7552760075361111, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
