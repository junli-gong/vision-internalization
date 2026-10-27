# W2 · qwen3.5-9b · refseg · w2c · explore_val_refcoco_unc_val

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（适配器 qwen3.5-9b）
- 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True）；提出 mode = box（<out>/qwen3.5-9b/refexp_box/propose_w2c/explore_val_refcoco_unc_val）
- 候选：<out>/qwen3.5-9b/refseg/candidates_w2c/explore_val_refcoco_unc_val（src.w2.candidates，K = 4）
- 检查点：/home/mail/chenyuma/visint/repo/experiments/W2/0069_w2c_hpcgpu06/0069_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（manifest sha256 c605408fbb92ccea…；训练 {'variant': 'w2c', 'seed': 0, 'identity_sha256': '61fd03751553cfc571a998f13295f96d7cc8fe9899a4f947682cfee93452e506', 'w2_config_sha256': 'c4d083489783039d46718e19ede810370bd4b9272ef94e99f406104c2e0f9b29', 'config_sha256': '709b24aec5ba9e0d0449e47d3391bdca5f79dc2673ae497c1b907582962b668f', 'complete': True, 'steps': 2000}）
- 挑选：generate_batch，思考 False（预算 2048），作答 64；渲染 src.w2.render（layout = separate）；批 {'batch_size': 16, 'chunk_size': 64, 'max_batch_tokens': 48000}

## 标题指标

- cIoU：0.6488，95% CI [0.6186, 0.6789]，n = 400
- gIoU：0.6643，95% CI [0.6428, 0.6848]，n = 400

## 挑选

- 有候选 400 / 400；失败 {'ok': 400}
- 交的是最优候选 0.5775；交 1 号 0.1125；"都不对" 0.0000；解析失败 0.0000
- 最优候选（oracle）在有候选的样本上的 cIoU：0.6610
- 各候选的 cIoU：{'1': ('gc_default', '0.5965'), '2': ('gc_soft_box_minus10', '0.5845'), '3': ('gc_default+lcc+fill', '0.5965'), '4': ('ws_pts', '0.5713')}

## 文件

- predictions.jsonl（逐条：提出、各候选的 IoU、挑选原文、挑中的编号、fallback、最终 mask 的 RLE、计分）；metrics.json；compute.json；config.yaml；meta.json；logs/（timing.jsonl、run_*.json）。
