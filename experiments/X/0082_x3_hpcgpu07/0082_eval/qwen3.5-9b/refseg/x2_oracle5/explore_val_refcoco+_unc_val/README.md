# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.700313852720217（95% CI {'low': 0.6593474913978289, 'high': 0.7398301931949706, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.71485980700106（95% CI {'low': 0.6885074508991281, 'high': 0.7404120696888828, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
