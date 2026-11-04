# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5675975465205534（95% CI {'low': 0.5503319975020374, 'high': 0.5844747700345542, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5877720738685639（95% CI {'low': 0.5733234445989733, 'high': 0.6014188576003482, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
