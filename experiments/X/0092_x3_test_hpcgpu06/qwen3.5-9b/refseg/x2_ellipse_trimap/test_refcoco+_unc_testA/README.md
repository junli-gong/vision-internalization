# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcoco+_unc_testA

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest 837bb679cfc172ec…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco+_unc_testA；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5449194295291911（95% CI {'low': 0.5300354987821155, 'high': 0.5610622717530299, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5639815531662041（95% CI {'low': 0.5521755327008054, 'high': 0.5758154633494605, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
