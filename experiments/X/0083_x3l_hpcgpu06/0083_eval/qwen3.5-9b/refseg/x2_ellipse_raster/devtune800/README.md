# X-2 · qwen3.5-9b · refseg · x2_ellipse_raster · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "raster"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5470896127458307（95% CI {'low': 0.5300953944618567, 'high': 0.5650682377097724, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5685116961090015（95% CI {'low': 0.5548127615329528, 'high': 0.581713732971321, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
