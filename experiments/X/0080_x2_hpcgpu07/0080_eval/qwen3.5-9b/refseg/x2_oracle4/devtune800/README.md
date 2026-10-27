# X-2 · qwen3.5-9b · refseg · x2_oracle4 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 4}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6587737983790816（95% CI {'low': 0.6386254861939216, 'high': 0.6784687847691454, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6628465176114511（95% CI {'low': 0.6463493054743308, 'high': 0.6776858955755338, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 1.0
