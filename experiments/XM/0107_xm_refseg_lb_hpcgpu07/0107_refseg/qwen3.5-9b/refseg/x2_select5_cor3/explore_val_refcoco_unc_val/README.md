# X-2 · qwen3.5-9b · refseg · x2_select5_cor3 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6912444597099615（95% CI {'low': 0.6490600335283131, 'high': 0.726726873303903, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6727264849583507（95% CI {'low': 0.644314297916671, 'high': 0.6973186371189403, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 6.0
