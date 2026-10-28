# W2-c 训练 · qwen3.5-9b · seed 0

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5；适配器 qwen3.5-9b；bf16）；图像口径 pixel_budget:300
- 监督：只有文本——提出（E1 refexp_box 提示 + 原图 → GT 框，与 C3 的框监督逐字相同，order image_first）与挑选（挑选图 + 挑选提示 → 编号，order image_first）；关思考；不加任何新 token
- 训练：2000 步 × 16 条（提出 8 + 挑选 8；微批 4 × 累积 4，每个微批只含一种样本，先后 ['propose', 'select']）；损失 = 1 × 提出的逐 token 平均 + 1 × 挑选的逐 token 平均
- LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002，warmup 60 步 + 余弦（同 C3）
- X-2（DEV-0069，目标来自真值）：每步另加 10 条，spline 10 条（损失权重 1，按本流的监督 token 数平均）；参数 {'spline': {'n': 16, 'min_box_iou': 0.5}}；数据 manifest sha256 afd98f6b388c500a…，池 {'spline': 13987}；没有修正流（DEV-0070，X-3：只读 X-2 数据里的样条行）
- 回放（DEV-0065，自蒸馏）：每步另加 7 条，replay_general 4 条（general，损失权重 1，按本流的监督 token 数平均）；replay_count 3 条（count，损失权重 1，按本流的监督 token 数平均）（回放的权重合计 2，提出、挑选各 1）；数据 manifest sha256 1a68646e415d6ad4…，由 qwen3.5-9b@c20223623576 贪心、关思考生成；池 {'replay_general': 1237, 'replay_count': 2000}（截断的不用）
- W2 配置：configs/w2.json sha256 c4d083489783039d…（frozen = True；K = 4，layout = separate）
- 数据：16 个分片，16000 句（提出池 16000，挑选池 15691；提出失败 309）；标签分布 {'1': 2791, '2': 3582, '3': 3915, '4': 5403}；manifest sha256 22fd7767342d6f4f5cf6ccfb003329b9042e8087cbf3870f452d819f0f38bb83
- 第一步损失 4.565245242249451，最后一步损失 1.3837151073659568（2000 步）；本次合计 {'propose': {'n_samples': 16000, 'n_supervised_tokens': 747016}, 'select': {'n_samples': 16000, 'n_supervised_tokens': 32000}, 'spline': {'n_samples': 20000, 'n_supervised_tokens': 3184056}, 'replay_general': {'n_samples': 8000, 'n_supervised_tokens': 1707126}, 'replay_count': {'n_samples': 6000, 'n_supervised_tokens': 12504}}
- 检查点：ckpt/final/model（346305781 字节；只有 LoRA、tokens.json、manifest，与 C3 同格式）

## 文件

- train_log.jsonl：每步的确定性记录（loss、loss_by_stream、by_stream、select_number_acc……）；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；config.yaml：运行身份；meta.json：git、主机、包版本、数据分片的 sha256。
