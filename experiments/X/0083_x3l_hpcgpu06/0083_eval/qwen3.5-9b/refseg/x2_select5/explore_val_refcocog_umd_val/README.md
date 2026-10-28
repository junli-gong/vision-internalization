# X-2 · qwen3.5-9b · refseg · x2_select5 · explore_val_refcocog_umd_val

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7075628525482835（95% CI {'low': 0.666572888257089, 'high': 0.7497242094095216, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7054761115343127（95% CI {'low': 0.6754208288034699, 'high': 0.7356978219078308, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 3.0
