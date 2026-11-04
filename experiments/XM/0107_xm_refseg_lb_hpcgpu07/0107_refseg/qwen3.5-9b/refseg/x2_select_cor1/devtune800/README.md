# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6311984212402463（95% CI {'low': 0.604650063354117, 'high': 0.6578330526671519, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6128113033308729（95% CI {'low': 0.5919586383924635, 'high': 0.6335806372846458, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
