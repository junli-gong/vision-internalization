# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6591521085688634（95% CI {'low': 0.6334264368688857, 'high': 0.6827942352625067, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6547754561627722（95% CI {'low': 0.6350155717532657, 'high': 0.6733033273221172, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
