# X-2 · qwen3.5-9b · refseg · x2_select5_cor1 · explore_val_refcoco+_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5993148942957742（95% CI {'low': 0.555291815122367, 'high': 0.6414547840803739, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.595390618935497（95% CI {'low': 0.5594752469471238, 'high': 0.6286227165066416, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 4.0
