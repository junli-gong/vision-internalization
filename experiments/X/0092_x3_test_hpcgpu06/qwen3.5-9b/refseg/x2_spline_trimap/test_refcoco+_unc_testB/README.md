# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcoco+_unc_testB

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco+_unc_testB；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6657579037374486（95% CI {'low': 0.6401194899631022, 'high': 0.6884370889187981, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6790074158050895（95% CI {'low': 0.6608314171353576, 'high': 0.697227806883319, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
