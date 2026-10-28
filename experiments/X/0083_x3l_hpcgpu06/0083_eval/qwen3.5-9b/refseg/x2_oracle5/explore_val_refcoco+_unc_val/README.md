# X-2 · qwen3.5-9b · refseg · x2_oracle5 · explore_val_refcoco+_unc_val

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco+_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7107906442520883（95% CI {'low': 0.6681993213009536, 'high': 0.7525044671863566, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7266080347330017（95% CI {'low': 0.7027655166053272, 'high': 0.7497680027362162, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
