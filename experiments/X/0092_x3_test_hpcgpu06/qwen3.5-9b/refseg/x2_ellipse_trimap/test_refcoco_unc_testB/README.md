# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcoco_unc_testB

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco_unc_testB；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5976465828406873（95% CI {'low': 0.5792440702455509, 'high': 0.6167705918879421, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.628248502534989（95% CI {'low': 0.614114223336899, 'high': 0.6426632498655819, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
