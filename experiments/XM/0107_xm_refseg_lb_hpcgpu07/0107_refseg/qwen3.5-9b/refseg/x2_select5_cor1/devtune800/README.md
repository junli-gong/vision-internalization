# X-2 · qwen3.5-9b · refseg · x2_select5_cor1 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6609331688380922（95% CI {'low': 0.6347180478555386, 'high': 0.6872248546339363, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.640313023356907（95% CI {'low': 0.6185326772894977, 'high': 0.6606319997483644, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
