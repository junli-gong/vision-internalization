# X-2 · qwen3.5-9b · refseg · x2_ellipse_trimap · test_refcoco+_unc_testB

- 定义：{"kind": "ellipse", "stages": ["propose"], "multi_round": false, "diagnostic": false, "ablation": "ellipse", "mode": "trimap"}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest a49cf4e5a81fd8f7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/test_refcoco+_unc_testB；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.5597597865388554（95% CI {'low': 0.5398184367552215, 'high': 0.5788281339362134, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.5849767796181659（95% CI {'low': 0.5673841183609718, 'high': 0.6002679387929497, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 1000}；fallback {'ok': 1000}；每句模型调用 1.0
