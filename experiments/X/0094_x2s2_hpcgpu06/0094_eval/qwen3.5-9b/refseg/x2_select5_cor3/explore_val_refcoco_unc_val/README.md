# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.66372313131771（95% CI {'low': 0.6198852245744035, 'high': 0.7023922070171004, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6540624054003158（95% CI {'low': 0.6235912684189274, 'high': 0.6796777028513861, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 6.0
