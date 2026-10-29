# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7771048835776606（95% CI {'low': 0.7532033176092028, 'high': 0.8003796689717336, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.766477806705396（95% CI {'low': 0.748652269690317, 'high': 0.7841376996192854, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
