# W2 外部基线 LLMSeg · qwen3.5-9b · llmseg · test_refcoco_unc_testA

- 官方代码：https://github.com/ajhamdi/llmseg-code@f2191e6605e4（third_party/llmseg，未改动；PROVENANCE sha256 6be8e7382736e8f0…）
- 管线：{'iterations': 8, 'topology': False, 'refine_box': False, 'postprocess': True, 'max_instances': 200, 'main_config': True, 'converge': {'eps': 0.005, 'patience': 3}, 'calls_bound': 18}
- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b；官方模型名 qwen3.5-9b）；思考 False；generate_batch
- 子集：300 句（sha256 eeb97846d6ba0af4…）；合批 {'scheduler': 'lockstep（全部在跑的管线都在等模型时成一轮；按 (阶段, 上限) 分组、组内 plan_sub_batches）', 'slots': 150, 'batch_size': 32, 'max_batch_tokens': 48000, 'chunk_size': 150}

## 读数

- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））cIoU：0.2463，95% CI [0.2143, 0.2774]，n = 300
- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））gIoU：0.2354，95% CI [0.2106, 0.2603]，n = 300
  - 每图计算量：{'n_model_calls': 13.926666666666666, 'n_rounds': 8.08, 'prompt_tokens': 19783.336666666666, 'generated_tokens': 3530.51, 'visual_tokens': 4336.233333333334, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））cIoU：0.3871，95% CI [0.3602, 0.4149]，n = 300
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））gIoU：0.3817，95% CI [0.3580, 0.4049]，n = 300
  - 每图计算量：{'n_model_calls': 1.9533333333333334, 'n_rounds': 1.9533333333333334, 'prompt_tokens': 2669.47, 'generated_tokens': 833.8133333333334, 'visual_tokens': 607.7266666666667, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}

## 调用与格式

- status {'failed': 10, 'ok': 290}；每图调用 {'n': 300, 'min': 1, 'p05': 2.0, 'p50': 16.0, 'p95': 18.0, 'max': 18, 'mean': 13.926666666666666, 'n_distinct': 15, 'top': [[18, 132], [10, 34], [16, 25], [14, 20], [12, 19], [8, 16], [1, 14], [17, 13], [2, 10], [11, 6]]}
- 截断 {'n': 72, 'of': 4178, 'rate': 0.01723312589755864}；格式失败 {'n': 75, 'of': 4178, 'rate': 0.01795117280995692}；坐标 > 1000 319 / 105704（3 次调用）
