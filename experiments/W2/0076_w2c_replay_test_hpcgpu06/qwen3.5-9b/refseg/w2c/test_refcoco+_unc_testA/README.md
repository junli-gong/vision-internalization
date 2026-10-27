# W2 · qwen3.5-9b · refseg · w2c · test_refcoco+_unc_testA

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True）；提出 mode = box（<out>/qwen3.5-9b/refexp_box/propose_w2c/test_refcoco+_unc_testA）
- 候选：<out>/qwen3.5-9b/refseg/candidates_w2c/test_refcoco+_unc_testA（src.w2.candidates，K = 4）
- 检查点：/home/mail/chenyuma/visint/repo/experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest sha256 ecad7667b4588f48…；训练 {'variant': 'w2c', 'seed': 0, 'identity_sha256': '1494de31793fd35618e05dc634cafaeff69b115fd116a942528fb32b118ee4bc', 'w2_config_sha256': 'c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29', 'config_sha256': 'a775861041c6bf4ebb8c05a7d02505ef50407727b6c2c91e072e885e76b0f920', 'complete': True, 'steps': 2000}）
- 挑选：generate_batch，思考 False（预算 2048），作答 64；渲染 src.w2.render（layout = separate）；批 {'batch_size': 16, 'chunk_size': 64, 'max_batch_tokens': 48000}

## 标题指标

- cIoU：0.6173，95% CI [0.5986, 0.6357]，n = 1000
- gIoU：0.6390，95% CI [0.6260, 0.6512]，n = 1000

## 挑选

- 有候选 1000 / 1000；失败 {'ok': 1000}
- 交的是最优候选 0.6390；交 1 号 0.1210；"都不对" 0.0000；解析失败 0.0000
- 最优候选（oracle）在有候选的样本上的 cIoU：0.6321
- 各候选的 cIoU：{'1': ('gc_default', '0.5536'), '2': ('gc_soft_box_minus10', '0.5540'), '3': ('gc_default+lcc+fill', '0.5539'), '4': ('ws_pts', '0.5373')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
