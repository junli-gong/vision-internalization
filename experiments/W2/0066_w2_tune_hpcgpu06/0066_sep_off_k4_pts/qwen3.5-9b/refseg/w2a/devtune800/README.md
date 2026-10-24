# W2 · qwen3.5-9b · refseg · w2a · devtune800

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 配置：/home/mail/chenyuma/visint/repo/experiments/W2/0066_w2_tune_hpcgpu06/cfg/sep_off_k4_pts.json sha256 792f9b2dbfa96ac1…（frozen = False）；提出 mode = box_points（<out>/qwen3.5-9b/refseg/propose_box_points/devtune800）
- 候选：<out>/qwen3.5-9b/refseg/candidates_box_points/devtune800（src.w2.candidates，K = 4）
- 挑选：generate_batch，思考 False（预算 2048），作答 64；渲染 src.w2.render（layout = separate）；批 {'batch_size': 16, 'chunk_size': 64, 'max_batch_tokens': 48000}

## 标题指标

- cIoU：0.0816，95% CI [0.0587, 0.1056]，n = 800
- gIoU：0.0692，95% CI [0.0556, 0.0849]，n = 800

## 挑选

- 有候选 94 / 800；失败 {'ambiguous': 33, 'malformed': 4, 'multiple_boxes': 560, 'no_box': 2, 'ok': 94, 'truncated': 107}
- 交的是最优候选 0.2872；交 1 号 0.9574；"都不对" 0.1170；解析失败 0.7660
- 最优候选（oracle）在有候选的样本上的 cIoU：0.5779
- 各候选的 cIoU：{'1': ('gc_default', '0.5399'), '2': ('gc_pts', '0.5267'), '3': ('gc_default+lcc+fill', '0.5419'), '4': ('ws_pts', '0.4028')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
