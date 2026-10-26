# W2 · qwen3.5-2b · refseg · w2a · test_refcoco+_unc_testB

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（适配器 qwen3.5-2b）
- 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True）；提出 mode = box（<out>/qwen3.5-2b/refexp_box/baseline_e1/test_refcoco+_unc_testB）
- 候选：<out>/qwen3.5-2b/refseg/candidates/test_refcoco+_unc_testB（src.w2.candidates，K = 4）
- 挑选：chat_turn_batch，思考 True（预算 2048），作答 64；渲染 src.w2.render（layout = separate）；批 {'batch_size': 16, 'chunk_size': 64, 'max_batch_tokens': 48000}

## 标题指标

- cIoU：0.4668，95% CI [0.4412, 0.4926]，n = 1000
- gIoU：0.4802，95% CI [0.4596, 0.5004]，n = 1000

## 挑选

- 有候选 971 / 1000；失败 {'multiple_boxes': 22, 'ok': 971, 'truncated': 7}
- 交的是最优候选 0.3347；交 1 号 0.5541；"都不对" 0.0062；解析失败 0.4542
- 最优候选（oracle）在有候选的样本上的 cIoU：0.5519
- 各候选的 cIoU：{'1': ('gc_default', '0.4893'), '2': ('gc_soft_box_minus10', '0.4912'), '3': ('gc_default+lcc+fill', '0.4921'), '4': ('ws_pts', '0.4647')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
