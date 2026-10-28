# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · devtune800

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.563860270932412（95% CI {'low': 0.5459048990416382, 'high': 0.5820811763757469, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5839191504103194（95% CI {'low': 0.5695704624157695, 'high': 0.5975606224658203, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
