# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.593948032073751（95% CI {'low': 0.549026035138266, 'high': 0.6357856088162851, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6013825645867342（95% CI {'low': 0.5721356261405031, 'high': 0.6306462855831775, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
