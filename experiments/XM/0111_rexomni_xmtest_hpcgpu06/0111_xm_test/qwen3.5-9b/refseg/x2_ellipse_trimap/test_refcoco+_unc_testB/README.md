# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcoco+_unc_testB

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 29e26667c7810ef2…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco+_unc_testB；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5572029858240195（95% CI {'low': 0.5377689630751497, 'high': 0.5767331715822549, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5831903766944868（95% CI {'low': 0.5674102874394288, 'high': 0.5983018160153974, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
