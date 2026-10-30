# X-2 · qwen3.5-4b · refseg · x2_select · devtune800

- 定义：{"kind": "select", "stages": ["propose", "select4"], "multi_round": false, "diagnostic": false, "ablation": null}
- 检查点：/home/mail/chenyuma/visint/repo/experiments/X/0098_x2_4b_hpcgpu07/0098_train/w2c/qwen3.5-4b/seed0/ckpt/final/model（manifest 1ce5dc51c6fdc04f…）
- 轨迹：<out>/qwen3.5-4b/refseg/x2_trace/devtune800；运行配置 configs/x2_run.json（c92ad838a9e69823…）
- cIoU：0.6323276018825293（95% CI {'low': 0.6108527557290708, 'high': 0.6540057555296575, 'level': 0.95, 'n_resamples': 1000}）
- gIoU：0.6466224179722017（95% CI {'low': 0.629314057827681, 'high': 0.6620850667032919, 'level': 0.95, 'n_resamples': 1000}）
- 失败 {'ok': 800}；fallback {'ok': 800}；每句模型调用 2.0
