# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcocog_umd_test

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcocog_umd_test；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.540775381917543（95% CI {'low': 0.52289539835218, 'high': 0.5577741456938329, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5527644937215493（95% CI {'low': 0.5385086091021838, 'high': 0.5669659763225052, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
