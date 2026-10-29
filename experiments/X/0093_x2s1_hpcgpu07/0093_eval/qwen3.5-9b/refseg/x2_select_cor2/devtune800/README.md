# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0093_x2s1_hpcgpu07/0093_train/w2c/qwen3.5-9b/seed1/ckpt/final/model（manifest 333f446b3a3ca77a…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.664146317001459（95% CI {'low': 0.6416861311879191, 'high': 0.6869441541371704, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6550917709382655（95% CI {'low': 0.6353389730182385, 'high': 0.6727797753439358, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 4.0
