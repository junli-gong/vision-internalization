# X-2 · qwen3.5-9b · refseg · x2_select · explore_val_refcocog_umd_val

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/explore_val_refcocog_umd_val；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6335793409860518（95% CI {'low': 0.5781737297836164, 'high': 0.6852007014320841, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6338377162266634（95% CI {'low': 0.599676781288546, 'high': 0.6674200961585242, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 200}；fallback {'ok': 200}；每句模型调用 2.0
