# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devtune800

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7433423926172584（95% CI {'low': 0.72137109088072, 'high': 0.7622448444233255, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7428185643615575（95% CI {'low': 0.72735868068793, 'high': 0.7567972974432402, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
