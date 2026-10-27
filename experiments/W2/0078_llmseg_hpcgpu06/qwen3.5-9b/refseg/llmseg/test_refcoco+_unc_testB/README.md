# W2 外部基线 LLMSeg · qwen3.5-9b · llmseg · test_refcoco+_unc_testB

- 官方代码：https://github.com/ajhamdi/llmseg-code@f2191e6605e4（third_party/llmseg，未改动；PROVENANCE sha256 6be8e7382736e8f0…）
- 管线：{'iterations': 8, 'topology': False, 'refine_box': False, 'postprocess': True, 'max_instances': 200, 'main_config': True, 'converge': {'eps': 0.005, 'patience': 3}, 'calls_bound': 18}
- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b；官方模型名 qwen3.5-9b）；思考 False；generate_batch
- 子集：300 句（sha256 4b0ea42c6e566929…）；合批 {'scheduler': 'lockstep（全部在跑的管线都在等模型时成一轮；按 (阶段, 上限) 分组、组内 plan_sub_batches）', 'slots': 150, 'batch_size': 32, 'max_batch_tokens': 48000, 'chunk_size': 150}

## 读数

- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））cIoU：0.2942，95% CI [0.2559, 0.3318]，n = 300
- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））gIoU：0.2696，95% CI [0.2415, 0.2969]，n = 300
  - 每图计算量：{'n_model_calls': 13.466666666666667, 'n_rounds': 7.8966666666666665, 'prompt_tokens': 19391.68666666667, 'generated_tokens': 3544.8, 'visual_tokens': 4174.986666666667, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））cIoU：0.3926，95% CI [0.3566, 0.4283]，n = 300
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））gIoU：0.3992，95% CI [0.3693, 0.4264]，n = 300
  - 每图计算量：{'n_model_calls': 1.97, 'n_rounds': 1.97, 'prompt_tokens': 2737.576666666667, 'generated_tokens': 814.4933333333333, 'visual_tokens': 610.92, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}

## 调用与格式

- status {'failed': 6, 'ok': 294}；每图调用 {'n': 300, 'min': 1, 'p05': 4.850000000000003, 'p50': 14.0, 'p95': 18.0, 'max': 18, 'mean': 13.466666666666667, 'n_distinct': 15, 'top': [[18, 112], [10, 48], [12, 39], [14, 23], [16, 15], [8, 11], [17, 10], [1, 9], [5, 8], [2, 6]]}
- 截断 {'n': 52, 'of': 4040, 'rate': 0.01287128712871287}；格式失败 {'n': 81, 'of': 4040, 'rate': 0.02004950495049505}；坐标 > 1000 226 / 97378（7 次调用）
