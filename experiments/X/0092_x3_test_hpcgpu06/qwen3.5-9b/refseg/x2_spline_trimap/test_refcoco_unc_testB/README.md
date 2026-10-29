# X-2 · qwen3.5-9b · refseg · x2_spline_trimap · test_refcoco_unc_testB

- 定义：{"kind": "spline", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": false, "ablation": null, "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco_unc_testB；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7148221479026677（95% CI {'low': 0.6908143793098013, 'high': 0.7375769352872359, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.731071512016418（95% CI {'low': 0.7141693615004783, 'high': 0.7467209371586051, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 2.0
