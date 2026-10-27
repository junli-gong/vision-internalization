# X-2 · qwen3.5-9b · refseg · x2_select5_cor2 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "spline", "select5", "cor_select5"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select5", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6927928231529225（95% CI {'low': 0.6677087593300722, 'high': 0.7148087269865735, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6813718871390584（95% CI {'low': 0.6619671104331981, 'high': 0.6997458861934914, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 5.0
