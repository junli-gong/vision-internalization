# X-2 · qwen3.5-4b · refseg · x2_select5 · devtune800

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7273550231818248（95% CI {'low': 0.7039571606925964, 'high': 0.7489109718305468, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7332235326134999（95% CI {'low': 0.7171370417854013, 'high': 0.7478740251472665, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
