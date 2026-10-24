# W2 · qwen3.5-9b · refseg · w2a · devtune800

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 配置：/home/mail/chenyuma/visint/repo/experiments/W2/0066_w2_tune_hpcgpu06/cfg/sep_off_k4.json sha256 e695bbf01b61886e…（frozen = False）；提出 mode = box（<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune800）
- 候选：<out>/qwen3.5-9b/refseg/candidates/devtune800（src.w2.candidates，K = 4）
- 挑选：generate_batch，思考 False（预算 2048），作答 64；渲染 src.w2.render（layout = separate）；批 {'batch_size': 16, 'chunk_size': 64, 'max_batch_tokens': 48000}

## 标题指标

- cIoU：0.5721，95% CI [0.5470, 0.5984]，n = 800
- gIoU：0.5525，95% CI [0.5301, 0.5728]，n = 800

## 挑选

- 有候选 786 / 800；失败 {'multiple_boxes': 13, 'ok': 786, 'truncated': 1}
- 交的是最优候选 0.2061；交 1 号 0.9580；"都不对" 0.0712；解析失败 0.8524
- 最优候选（oracle）在有候选的样本上的 cIoU：0.6520
- 各候选的 cIoU：{'1': ('gc_default', '0.5848'), '2': ('gc_soft_box_minus10', '0.5709'), '3': ('gc_default+lcc+fill', '0.5850'), '4': ('ws_pts', '0.5438')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
