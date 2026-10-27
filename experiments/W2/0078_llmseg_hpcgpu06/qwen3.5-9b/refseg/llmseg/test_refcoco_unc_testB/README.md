# W2 外部基线 LLMSeg · qwen3.5-9b · llmseg · test_refcoco_unc_testB

- 官方代码：https://github.com/ajhamdi/llmseg-code@f2191e6605e4（third_party/llmseg，未改动；PROVENANCE sha256 6be8e7382736e8f0…）
- 管线：{'iterations': 8, 'topology': False, 'refine_box': False, 'postprocess': True, 'max_instances': 200, 'main_config': True, 'converge': {'eps': 0.005, 'patience': 3}, 'calls_bound': 18}
- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b；官方模型名 qwen3.5-9b）；思考 False；generate_batch
- 子集：300 句（sha256 9b35912cbe696756…）；合批 {'scheduler': 'lockstep（全部在跑的管线都在等模型时成一轮；按 (阶段, 上限) 分组、组内 plan_sub_batches）', 'slots': 150, 'batch_size': 32, 'max_batch_tokens': 48000, 'chunk_size': 150}

## 读数

- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））cIoU：0.3201，95% CI [0.2804, 0.3620]，n = 300
- rec（修正后（官方 rec_*；主配置 T 轮修正，标题读数））gIoU：0.2940，95% CI [0.2649, 0.3240]，n = 300
  - 每图计算量：{'n_model_calls': 13.983333333333333, 'n_rounds': 8.17, 'prompt_tokens': 20082.966666666667, 'generated_tokens': 3750.41, 'visual_tokens': 4354.233333333334, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））cIoU：0.4528，95% CI [0.4176, 0.4871]，n = 300
- seg（修正前（官方 seg_*：阶段 2 之后，阶段 1、2 两次调用））gIoU：0.4489，95% CI [0.4202, 0.4757]，n = 300
  - 每图计算量：{'n_model_calls': 1.9766666666666666, 'n_rounds': 1.9766666666666666, 'prompt_tokens': 2733.5666666666666, 'generated_tokens': 834.6166666666667, 'visual_tokens': 614.9133333333333, 'rule': '每图均值；墙钟见 compute.json / compute_stage2.json（logs/timing.jsonl）'}

## 调用与格式

- status {'failed': 6, 'ok': 294}；每图调用 {'n': 300, 'min': 1, 'p05': 5.0, 'p50': 16.0, 'p95': 18.0, 'max': 18, 'mean': 13.983333333333333, 'n_distinct': 14, 'top': [[18, 130], [12, 37], [10, 29], [14, 24], [16, 19], [5, 12], [8, 10], [1, 7], [11, 7], [2, 6]]}
- 截断 {'n': 64, 'of': 4195, 'rate': 0.015256257449344458}；格式失败 {'n': 78, 'of': 4195, 'rate': 0.018593563766388557}；坐标 > 1000 115 / 110054（5 次调用）
