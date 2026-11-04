# X-2 · qwen3.5-9b · refseg · x2_select_cor2 · devcheck2

- 定义：{"kind": "cor", "stages": ["propose", "select4", "cor_select"], "multi_round": true, "diagnostic": false, "ablation": null, "base": "select", "rounds": 2}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devcheck2；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6783754311322159（95% CI {'low': 0.6489426906697591, 'high': 0.7068438579829943, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6800285567577243（95% CI {'low': 0.6580974245724903, 'high': 0.7026185112340316, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 500}；fallback {'ok': 500}；每句模型调用 4.0
