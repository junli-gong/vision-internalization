# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5479082294849903（95% CI {'low': 0.5310728040782412, 'high': 0.5636640884887009, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5684212927588856（95% CI {'low': 0.554411225871179, 'high': 0.5817743416849721, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
