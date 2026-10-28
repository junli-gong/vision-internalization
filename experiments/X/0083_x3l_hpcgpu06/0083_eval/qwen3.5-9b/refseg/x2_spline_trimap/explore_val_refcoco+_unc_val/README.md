# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcoco+_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7038672740494848（95% CI {'low': 0.6629896498534235, 'high': 0.74326188920858, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7118820578312466（95% CI {'low': 0.6877401406942852, 'high': 0.7355880514925567, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
