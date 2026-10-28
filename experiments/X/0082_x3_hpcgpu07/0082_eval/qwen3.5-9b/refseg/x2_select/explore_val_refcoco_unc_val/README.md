# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcoco_unc_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcoco_unc_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6389600174594737（95% CI {'low': 0.6102627976694954, 'high': 0.6669415375574025, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6510207209936609（95% CI {'low': 0.6292960372310699, 'high': 0.6721082427017606, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 400}；fallback {'ok': 400}；每句模型调用 2.0
