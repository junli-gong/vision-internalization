# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6671190051210132（95% CI {'low': 0.6378657450197279, 'high': 0.6955231172359215, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6797254593536821（95% CI {'low': 0.6587089700984401, 'high': 0.7011374709421232, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
