# X-2 · qwen3.5-9b · refseg · x2_select_cor3_randbox · explore_val_refcoco_unc_val

- 定义：{"kind": "randbox", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": "randbox", "base": "select", "rounds": 3}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5776058410527348（95% CI {'low': 0.536161918449868, 'high': 0.616759630945578, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.591001523216051（95% CI {'low': 0.5616058562002721, 'high': 0.6169858480164558, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 5.0
