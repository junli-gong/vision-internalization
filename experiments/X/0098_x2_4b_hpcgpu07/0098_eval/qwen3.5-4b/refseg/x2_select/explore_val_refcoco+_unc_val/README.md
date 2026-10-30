# X-2 · qwen3.5-4b · refseg · x2_select · explore_val_refcoco+_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5966350745146866（95% CI {'low': 0.5604580708617912, 'high': 0.6308599716683937, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6138154306165344（95% CI {'low': 0.5870020527152888, 'high': 0.6373507616182643, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
