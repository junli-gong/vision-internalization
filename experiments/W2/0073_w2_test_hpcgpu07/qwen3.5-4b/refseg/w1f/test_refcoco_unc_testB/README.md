# W2 · qwen3.5-4b · refseg · w1f · test_refcoco_unc_testB

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（适配器 qwen3.5-4b）
- 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True）；提出 mode = box（<out>/qwen3.5-4b/refexp_box/baseline_e1/test_refcoco_unc_testB）
- 候选：<out>/qwen3.5-4b/refseg/candidates/test_refcoco_unc_testB（src.w2.candidates，K = 4）

## 标题指标

- cIoU：0.5581，95% CI [0.5339, 0.5821]，n = 1000
- gIoU：0.5374，95% CI [0.5175, 0.5574]，n = 1000

## 挑选

- 有候选 973 / 1000；失败 {'multiple_boxes': 23, 'no_box': 1, 'ok': 973, 'truncated': 3}
- 交的是最优候选 0.1429；交 1 号 1.0000；"都不对" 0.0000；解析失败 0.0000
- 最优候选（oracle）在有候选的样本上的 cIoU：0.6526
- 各候选的 cIoU：{'1': ('gc_default', '0.5759'), '2': ('gc_soft_box_minus10', '0.5746'), '3': ('gc_default+lcc+fill', '0.5758'), '4': ('ws_pts', '0.5582')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
