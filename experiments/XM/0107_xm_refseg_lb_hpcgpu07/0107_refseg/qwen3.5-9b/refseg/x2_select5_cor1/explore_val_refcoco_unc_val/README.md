# X-2 · qwen3.5-9b · refseg · x2_select5_cor1 · explore_val_refcoco_unc_val

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6594708318400053（95% CI {'low': 0.6150950666030826, 'high': 0.6980957467438392, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6521124869873622（95% CI {'low': 0.6189794052423525, 'high': 0.6779555020301893, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 4.0
