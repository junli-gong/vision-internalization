# W2 · qwen3.5-9b · refseg · oracle_pick · devtune800

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 配置：configs/w2.json sha256 a4c013c344c83f18…（frozen = False）；提出 mode = box（<out>/qwen3.5-9b/refexp_box/baseline_e1/devtune800）
- 候选：<out>/qwen3.5-9b/refseg/candidates/devtune800（src.w2.candidates，K = 8）

## 标题指标

- cIoU：0.6470，95% CI [0.6240, 0.6700]，n = 800
- gIoU：0.6586，95% CI [0.6418, 0.6748]，n = 800

## 挑选

- 有候选 786 / 800；失败 {'multiple_boxes': 13, 'ok': 786, 'truncated': 1}
- 交的是最优候选 1.0000；交 1 号 0.1692；"都不对" 0.0000；解析失败 0.0000
- 最优候选（oracle）在有候选的样本上的 cIoU：0.6606
- 各候选的 cIoU：{'1': ('gc_default', '0.5848'), '2': ('gc_box_plus10', '0.5305'), '3': ('gc_soft_box_minus10', '0.5709'), '4': ('gc_default+lcc+fill', '0.5850'), '5': ('slic_s30_t50', '0.4901'), '6': ('slic_s100_t50', '0.5036'), '7': ('ws_pts', '0.5438'), '8': ('box', '0.5103')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
