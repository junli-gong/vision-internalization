# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcoco+_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0094_x2s2_hpcgpu06/0094_train/w2c/qwen3.5-9b/seed2/ckpt/final/model（manifest 836e69235e33b1c9…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6084095925739106（95% CI {'low': 0.5709368163689293, 'high': 0.6455308352432042, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6333461195012746（95% CI {'low': 0.609192077217176, 'high': 0.657812708730378, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
