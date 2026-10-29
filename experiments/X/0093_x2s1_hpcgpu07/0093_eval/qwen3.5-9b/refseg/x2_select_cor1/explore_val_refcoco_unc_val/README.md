# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5874714574832804（95% CI {'low': 0.5440326072879395, 'high': 0.6269916730236119, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5891509547129591（95% CI {'low': 0.5559908021922679, 'high': 0.6169541444758877, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 3.0
