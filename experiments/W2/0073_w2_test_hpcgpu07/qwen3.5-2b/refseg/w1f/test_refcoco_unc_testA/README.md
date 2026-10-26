# W2 · qwen3.5-2b · refseg · w1f · test_refcoco_unc_testA

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（适配器 qwen3.5-2b）
- 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True）；提出 mode = box（<out>/qwen3.5-2b/refexp_box/baseline_e1/test_refcoco_unc_testA）
- 候选：<out>/qwen3.5-2b/refseg/candidates/test_refcoco_unc_testA（src.w2.candidates，K = 4）

## 标题指标

- cIoU：0.5507，95% CI [0.5288, 0.5720]，n = 1000
- gIoU：0.5334，95% CI [0.5169, 0.5503]，n = 1000

## 挑选

- 有候选 999 / 1000；失败 {'multiple_boxes': 1, 'ok': 999}
- 交的是最优候选 0.1481；交 1 号 1.0000；"都不对" 0.0000；解析失败 0.0000
- 最优候选（oracle）在有候选的样本上的 cIoU：0.6192
- 各候选的 cIoU：{'1': ('gc_default', '0.5519'), '2': ('gc_soft_box_minus10', '0.5468'), '3': ('gc_default+lcc+fill', '0.5566'), '4': ('ws_pts', '0.5289')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
