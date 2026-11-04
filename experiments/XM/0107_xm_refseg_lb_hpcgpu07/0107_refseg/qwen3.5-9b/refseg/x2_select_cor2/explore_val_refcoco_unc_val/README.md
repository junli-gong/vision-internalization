# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.671279733784955（95% CI {'low': 0.634537754056172, 'high': 0.7049255327181766, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.666841861126112（95% CI {'low': 0.6421165561819182, 'high': 0.6913375823869242, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 4.0
