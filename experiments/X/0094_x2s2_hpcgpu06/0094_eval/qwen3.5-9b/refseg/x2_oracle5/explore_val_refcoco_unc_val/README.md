# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7814371758927052（95% CI {'low': 0.758058169808142, 'high': 0.8044453420023479, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7712677696543238（95% CI {'low': 0.7540424099308867, 'high': 0.7888983185029285, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
