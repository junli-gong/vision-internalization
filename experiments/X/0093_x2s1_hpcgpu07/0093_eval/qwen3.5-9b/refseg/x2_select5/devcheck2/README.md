# X-2 · qwen3.5-9b · refseg · x2_select5 · devcheck2

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7358878841883522（95% CI {'low': 0.706159139581856, 'high': 0.7655023826691201, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7478745219775178（95% CI {'low': 0.7285979163609186, 'high': 0.7660104429375226, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 3.0
