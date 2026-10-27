# W2-c 训练 · qwen3.5-9b · seed 0

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b；bf16）；图像口径 pixel_budget:300
- 监督：只有文本——提出（E1 refexp_box 提示 + 原图 → GT 框，与 C3 的框监督逐字相同，order image_first）与挑选（挑选图 + 挑选提示 → 编号，order image_first）；关思考；不加任何新 token
- 训练：2000 步 × 16 条（提出 8 + 挑选 8；微批 4 × 累积 4，每个微批只含一种样本，先后 ['propose', 'select']）；损失 = 1 × 提出的逐 token 平均 + 1 × 挑选的逐 token 平均
- LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002，warmup 60 步 + 余弦（同 C3）
- X-2（DEV-0069，目标来自真值）：每步另加 12 条，spline 6 条（损失权重 1，按本流的监督 token 数平均）；correct 6 条（损失权重 1，按本流的监督 token 数平均）；参数 {'spline': {'n': 16, 'min_box_iou': 0.5}, 'correct': {'m': 2, 'min_box_iou': None, 'start_salt': 'x2_correct_start_v1', 'round2_salt': 'x2_correct_round2_v1', 'round2_ratio': 0.5, 'round2_apply': {'rule': 'grabcut', 'soft_lock': True}}}；数据 manifest sha256 afd98f6b388c500a…，池 {'spline': 13987, 'correct': 23470}
- 回放（DEV-0065，自蒸馏）：每步另加 5 条，replay_general 3 条（general，损失权重 0.5，按本流的监督 token 数平均）；replay_count 2 条（count，损失权重 0.5，按本流的监督 token 数平均）（回放的权重合计 1，提出、挑选各 1）；数据 manifest sha256 1a68646e415d6ad4…，由 qwen3.5-9b@c20223623576 贪心、关思考生成；池 {'replay_general': 1237, 'replay_count': 2000}（截断的不用）
- W2 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True；K = 4，layout = separate）
- 数据：16 个分片，16000 句（提出池 16000，挑选池 15691；提出失败 309）；标签分布 {'1': 2791, '2': 3582, '3': 3915, '4': 5403}；manifest sha256 22fd7767342d6f4f5cf6ccfb003329b9042e8087cbf3870f452d819f0f38bb83
- 第一步损失 5.073545134464512，最后一步损失 2.0537806928501157（2000 步）；本次合计 {'propose': {'n_samples': 16000, 'n_supervised_tokens': 747016}, 'select': {'n_samples': 16000, 'n_supervised_tokens': 32000}, 'spline': {'n_samples': 12000, 'n_supervised_tokens': 1910271}, 'correct': {'n_samples': 12000, 'n_supervised_tokens': 660066}, 'replay_general': {'n_samples': 6000, 'n_supervised_tokens': 1280624}, 'replay_count': {'n_samples': 4000, 'n_supervised_tokens': 8336}}
- 检查点：ckpt/final/model（346305781 字节；只有 LoRA、tokens.json、manifest，与 C3 同格式）

## 文件

- train_log.jsonl：每步的确定性记录（loss、loss_by_stream、by_stream、select_number_acc……）；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；config.yaml：运行身份；meta.json：git、主机、包版本、数据分片的 sha256。
