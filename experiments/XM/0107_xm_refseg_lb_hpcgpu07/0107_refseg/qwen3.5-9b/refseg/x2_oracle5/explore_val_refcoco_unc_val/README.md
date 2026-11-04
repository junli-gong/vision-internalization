# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7869868633097059（95% CI {'low': 0.764788099043953, 'high': 0.8090554452738538, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7745568325400168（95% CI {'low': 0.7576756774230078, 'high': 0.7916989106179113, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
