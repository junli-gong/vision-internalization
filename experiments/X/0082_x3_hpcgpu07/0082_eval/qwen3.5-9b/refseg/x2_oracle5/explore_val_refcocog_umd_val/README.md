# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcocog_umd_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7503891568217891（95% CI {'low': 0.7072898595498229, 'high': 0.7913136420235527, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7349197616682426（95% CI {'low': 0.7057306728181966, 'high': 0.7643637942138054, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
