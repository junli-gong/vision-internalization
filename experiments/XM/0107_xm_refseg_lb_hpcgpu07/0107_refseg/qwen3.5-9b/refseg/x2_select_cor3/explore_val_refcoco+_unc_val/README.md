# X-2 · qwen3.5-9b · refseg · x2_select_cor3 · explore_val_refcoco+_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.612496066506795（95% CI {'low': 0.573543958454749, 'high': 0.649609941444487, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5978703337845888（95% CI {'low': 0.5660913733208816, 'high': 0.6288995853371647, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
