# X-2 · qwen3.5-4b · refseg · x2_oracle5 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7735010898098127（95% CI {'low': 0.7506251275764313, 'high': 0.7968694067728438, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7639594942201845（95% CI {'low': 0.7459554251897346, 'high': 0.7818169614216938, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
