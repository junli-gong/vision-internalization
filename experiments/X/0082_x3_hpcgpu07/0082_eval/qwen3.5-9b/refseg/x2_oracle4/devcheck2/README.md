# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devcheck2

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6579837303673337（95% CI {'low': 0.6351889572700479, 'high': 0.6838961799472274, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6801701230765828（95% CI {'low': 0.6624774741074749, 'high': 0.6990478951193181, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 1.0
