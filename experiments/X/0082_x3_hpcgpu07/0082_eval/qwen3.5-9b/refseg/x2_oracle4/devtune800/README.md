# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6565637890013996（95% CI {'low': 0.6351732063808473, 'high': 0.6788442709966114, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6628998129171351（95% CI {'low': 0.6470817496394716, 'high': 0.6780802915454915, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
