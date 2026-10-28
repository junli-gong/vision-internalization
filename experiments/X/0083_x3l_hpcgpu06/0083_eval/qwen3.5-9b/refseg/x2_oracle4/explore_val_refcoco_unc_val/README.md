# X-2 · qwen3.5-9b · refseg · x2_oracle4 · explore_val_refcoco_unc_val

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6692398360773538（95% CI {'low': 0.6408142830782487, 'high': 0.6969254938735489, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6788871489612469（95% CI {'low': 0.6585668680926557, 'high': 0.6982960187442516, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 1.0
