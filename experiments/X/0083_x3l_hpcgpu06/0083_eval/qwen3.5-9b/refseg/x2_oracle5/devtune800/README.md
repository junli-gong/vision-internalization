# X-2 · qwen3.5-9b · refseg · x2_oracle5 · devtune800

- 定义：{"kind": "oracle", "stages": ["propose", "spline"], "multi_round": false, "diagnostic": true, "ablation": null, "k": 5}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0083_x3l_hpcgpu06/0083_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest eb200b245cc620c7…）
- 轨迹：<out>/qwen3.5-9b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.7499200501452087（95% CI {'low': 0.7281496672960543, 'high': 0.7692445096339006, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.749613400447862（95% CI {'low': 0.7349459399896212, 'high': 0.762661607057027, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
