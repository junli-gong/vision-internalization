# X-2 · qwen3.5-9b · refseg · x2_select5 · devtune800

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7349650207769081（95% CI {'low': 0.7115744509720437, 'high': 0.7544625244153703, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.73595539950595（95% CI {'low': 0.7203287275890874, 'high': 0.7498038855091534, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
