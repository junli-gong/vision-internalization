# X-2 · qwen3.5-9b · refseg · x2_select · devtune800

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.649932236015705（95% CI {'low': 0.628780821394767, 'high': 0.6696225569696148, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6576086377298627（95% CI {'low': 0.6406046625629918, 'high': 0.6732300221278065, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
