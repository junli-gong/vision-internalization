# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcoco+_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6076366978158884（95% CI {'low': 0.571777508694209, 'high': 0.6424028961863149, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6254264437834823（95% CI {'low': 0.6014954103444368, 'high': 0.6491899759929499, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
