# X-2 · qwen3.5-4b · refseg · x2_select_cor2 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6490414855576515（95% CI {'low': 0.6154150475691306, 'high': 0.6822677333581195, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6453964062598361（95% CI {'low': 0.6212054939173914, 'high': 0.6702065662927226, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 4.0
