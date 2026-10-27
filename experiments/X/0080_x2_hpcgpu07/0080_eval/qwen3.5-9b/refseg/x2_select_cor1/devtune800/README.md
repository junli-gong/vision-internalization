# X-2 · qwen3.5-9b · refseg · x2_select_cor1 · devtune800

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 1}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6090420668785783（95% CI {'low': 0.5805168038722047, 'high': 0.6378447924085009, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.593261725115446（95% CI {'low': 0.5724913397274397, 'high': 0.6129622951245036, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 3.0
