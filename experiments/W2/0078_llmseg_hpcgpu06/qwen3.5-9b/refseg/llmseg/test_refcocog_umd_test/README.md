# W2 外部基线 LLMSeg · qwen3.5-9b · llmseg · test_refcocog_umd_test

- 官方代码：https://github.com/ajhamdi/llmseg-code@f2191e6605e4（third_party/llmseg，未改动；PROVENANCE sha256 6be8e7382736e8f0…）
- 管线：{'iterations': 8, 'topology': False, 'refine_box': False, 'postprocess': True, 'max_instances': 200, 'main_config': True, 'converge': {'eps': 0.005, 'patience': 3}, 'calls_bound': 18}
- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b；官方模型名 qwen3.5-9b）；思考 False；generate_batch
- 子集：300 句（sha256 8c8a1d134ab324e0…）；合批 {'scheduler': 'lockstep（全部在跑的管线都在等模型时成一轮；按 (阶段, 上限) 分组、组内 plan_sub_batches）', 'slots': 150, 'batch_size': 32, 'max_batch_tokens': 48000, 'chunk_size': 150}

## 读数

- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））cIoU：0.2745，95% CI [0.2304, 0.3149]，n = 300
- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））gIoU：0.2317，95% CI [0.2043, 0.2575]，n = 300
  - 每图计算量：{'n_model_calls': 13.52, 'n_rounds': 7.8933333333333335, 'prompt_tokens': 19571.536666666667, 'generated_tokens': 3637.0633333333335, 'visual_tokens': 4221.1, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））cIoU：0.3954，95% CI [0.3585, 0.4304]，n = 300
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））gIoU：0.3755，95% CI [0.3483, 0.4034]，n = 300
  - 每图计算量：{'n_model_calls': 1.9533333333333334, 'n_rounds': 1.9533333333333334, 'prompt_tokens': 2701.6566666666668, 'generated_tokens': 886.1233333333333, 'visual_tokens': 610.1833333333333, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}

## 调用与格式

- status {'failed': 16, 'ok': 284}；每图调用 {'n': 300, 'min': 1, 'p05': 2.0, 'p50': 16.0, 'p95': 18.0, 'max': 18, 'mean': 13.52, 'n_distinct': 16, 'top': [[18, 129], [10, 32], [12, 28], [16, 20], [14, 19], [2, 16], [1, 14], [8, 9], [17, 8], [7, 7]]}
- 截断 {'n': 78, 'of': 4056, 'rate': 0.019230769230769232}；格式失败 {'n': 93, 'of': 4056, 'rate': 0.022928994082840236}；坐标 > 1000 112 / 103294（5 次调用）
