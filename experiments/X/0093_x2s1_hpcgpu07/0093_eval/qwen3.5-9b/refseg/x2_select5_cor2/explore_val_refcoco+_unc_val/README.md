# X-2 · qwen3.5-9b · refseg · x2_select5_cor2 · explore_val_refcoco+_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6293680685580868（95% CI {'low': 0.5866985860041587, 'high': 0.6675238695813952, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6318516490891223（95% CI {'low': 0.600073276004866, 'high': 0.6608189231648351, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
