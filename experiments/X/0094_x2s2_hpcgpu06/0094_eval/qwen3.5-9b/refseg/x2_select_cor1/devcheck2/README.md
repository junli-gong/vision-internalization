# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6038773098468839（95% CI {'low': 0.5653272456987536, 'high': 0.6409331416646217, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6178583280030779（95% CI {'low': 0.591809989277625, 'high': 0.6433967652113303, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
