# X-2 · qwen3.5-9b · refseg · x2_select5 · devtune800

- 定义：{"kind": "select5", "stages": ["propose", "spline", "select5"], "multi_round": false, "diagnostic": false, "ablation": null, "note": "挑选没在 K = 5 上训练过"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7363184328581553（95% CI {'low': 0.71493129097842, 'high': 0.7565147904427191, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.7391608835159948（95% CI {'low': 0.7236601894746763, 'high': 0.7532922204457124, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
