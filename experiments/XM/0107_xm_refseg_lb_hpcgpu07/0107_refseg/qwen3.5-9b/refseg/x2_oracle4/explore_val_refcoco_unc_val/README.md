# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6741007569130304（95% CI {'low': 0.6457475211853114, 'high': 0.7022765805714875, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6832730455848964（95% CI {'low': 0.6619953945422874, 'high': 0.7027639933171659, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
