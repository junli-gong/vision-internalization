# W2 外部基线 LLMSeg · qwen3.5-9b · llmseg · test_refcoco+_unc_testA

- 官方代码：https://github.com/ajhamdi/llmseg-code@f2191e6605e4（third_party/llmseg，未改动；PROVENANCE sha256 6be8e7382736e8f0…）
- 管线：{'iterations': 8, 'topology': False, 'refine_box': False, 'postprocess': True, 'max_instances': 200, 'main_config': True, 'converge': {'eps': 0.005, 'patience': 3}, 'calls_bound': 18}
- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b；官方模型名 qwen3.5-9b）；思考 False；generate_batch
- 子集：300 句（sha256 6447673051b83fc0…）；合批 {'scheduler': 'lockstep（全部在跑的管线都在等模型时成一轮；按 (阶段, 上限) 分组、组内 plan_sub_batches）', 'slots': 150, 'batch_size': 32, 'max_batch_tokens': 48000, 'chunk_size': 150}

## 读数

- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））cIoU：0.2340，95% CI [0.2021, 0.2712]，n = 300
- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））gIoU：0.2173，95% CI [0.1940, 0.2400]，n = 300
  - 每图计算量：{'n_model_calls': 14.0, 'n_rounds': 8.116666666666667, 'prompt_tokens': 20007.84, 'generated_tokens': 3747.346666666667, 'visual_tokens': 4369.033333333334, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））cIoU：0.3664，95% CI [0.3341, 0.4006]，n = 300
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））gIoU：0.3520，95% CI [0.3284, 0.3758]，n = 300
  - 每图计算量：{'n_model_calls': 1.96, 'n_rounds': 1.96, 'prompt_tokens': 2696.4733333333334, 'generated_tokens': 839.44, 'visual_tokens': 610.7866666666666, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}

## 调用与格式

- status {'failed': 11, 'ok': 289}；每图调用 {'n': 300, 'min': 1, 'p05': 2.0, 'p50': 16.0, 'p95': 18.0, 'max': 18, 'mean': 14.0, 'n_distinct': 16, 'top': [[18, 130], [10, 32], [12, 31], [14, 23], [16, 19], [17, 13], [1, 12], [2, 11], [8, 7], [13, 5]]}
- 截断 {'n': 82, 'of': 4200, 'rate': 0.019523809523809523}；格式失败 {'n': 89, 'of': 4200, 'rate': 0.02119047619047619}；坐标 > 1000 270 / 109373（3 次调用）
