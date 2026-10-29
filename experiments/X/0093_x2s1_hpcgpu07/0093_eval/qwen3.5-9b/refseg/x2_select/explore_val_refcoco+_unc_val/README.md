# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcoco+_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5882705847209604（95% CI {'low': 0.55185423058816, 'high': 0.6240313312891375, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6133647041220844（95% CI {'low': 0.5876658846395466, 'high': 0.6382499315283034, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
