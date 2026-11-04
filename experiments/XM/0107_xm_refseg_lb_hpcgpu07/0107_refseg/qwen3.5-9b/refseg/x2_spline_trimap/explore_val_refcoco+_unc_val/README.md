# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · explore_val_refcoco+_unc_val

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6954685102094604（95% CI {'low': 0.6546032266061761, 'high': 0.7361178411886417, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7062487655542773（95% CI {'low': 0.6808473391424605, 'high': 0.7307131259219415, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 398, 'parse_failure': 2}；每句模型调用 2.0
