# X-2 · qwen3.5-4b · refseg · x2_oracle5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.703770495748498（95% CI {'low': 0.6626852636398516, 'high': 0.7398077535444957, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.714874265658061（95% CI {'low': 0.6888145937168726, 'high': 0.7374868294713792, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
