# X-2 · qwen3.5-9b · refseg · x2_select · devtune800

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6482822801906534（95% CI {'low': 0.6267554756422349, 'high': 0.6690031960614379, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6555254678680262（95% CI {'low': 0.6390650999937797, 'high': 0.6709115441041648, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
