# X-2 · qwen3.5-9b · refseg · x2_select5_cor2 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6777970059917804（95% CI {'low': 0.6341847383594293, 'high': 0.7160730176426727, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6816932705185534（95% CI {'low': 0.6535662629940114, 'high': 0.7088724865327506, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
