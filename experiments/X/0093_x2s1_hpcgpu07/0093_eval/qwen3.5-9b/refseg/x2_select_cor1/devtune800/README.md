# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6092143971991624（95% CI {'low': 0.5775493831320594, 'high': 0.6392016014142441, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5913437756937951（95% CI {'low': 0.5687819566417089, 'high': 0.6117405386327913, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
