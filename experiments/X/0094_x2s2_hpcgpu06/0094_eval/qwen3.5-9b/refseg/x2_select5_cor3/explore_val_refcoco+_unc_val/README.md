# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · explore_val_refcoco+_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6366963423658416（95% CI {'low': 0.5968778624052443, 'high': 0.6734595804514779, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6238621391100064（95% CI {'low': 0.5880803459540763, 'high': 0.6558964063700111, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 6.0
