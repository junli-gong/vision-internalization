# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.70779645636758（95% CI {'low': 0.6692295773536425, 'high': 0.7456589449025234, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7167883103575818（95% CI {'low': 0.6913064941890843, 'high': 0.7401748375361794, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
