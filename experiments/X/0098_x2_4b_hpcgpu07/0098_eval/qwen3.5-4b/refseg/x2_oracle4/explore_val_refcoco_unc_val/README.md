# X-2 · qwen3.5-4b · refseg · x2_oracle4 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6648376868755036（95% CI {'low': 0.6366903742433799, 'high': 0.6920557189298662, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6725854259141651（95% CI {'low': 0.6515729917742992, 'high': 0.693369932100132, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
