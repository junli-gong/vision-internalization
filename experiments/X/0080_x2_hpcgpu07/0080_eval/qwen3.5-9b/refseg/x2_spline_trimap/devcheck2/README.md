# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · devcheck2

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.733926240799275（95% CI {'low': 0.706621275311766, 'high': 0.7618176047825664, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7401659652406238（95% CI {'low': 0.7223234311618335, 'high': 0.7571862125869246, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 2.0
