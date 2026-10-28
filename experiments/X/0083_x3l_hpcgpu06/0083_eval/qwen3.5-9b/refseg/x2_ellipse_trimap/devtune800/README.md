# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5669560840158142（95% CI {'low': 0.5490852842937877, 'high': 0.5858178131232245, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5879241733431277（95% CI {'low': 0.57375590578665, 'high': 0.6018174817385707, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
