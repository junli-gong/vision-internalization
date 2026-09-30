| 测量 | batch | 峰值 allocated GiB | 峰值 reserved GiB | 备注 |
|---|---|---|---|---|
| 仅加载权重（bf16） | — | 9.508 | 9.602 | |
| 推理前向，1089 视觉 token（seq 1106，{"max_soft_tokens": 1120}），logits_to_keep=1 | 1 | 10.011 | 10.123 | logits [1, 1, 262144]；墙钟 0.584 s |
| 推理前向，1089 视觉 token（seq 1106，{"max_soft_tokens": 1120}），full_logits | 1 | 10.686 | 11.205 | logits [1, 1106, 262144]；墙钟 0.079 s |
| 推理前向，1089 视觉 token（seq 1106，{"max_soft_tokens": 1120}），logits_to_keep=1 | 16 | 16.625 | 17.912 | logits [16, 1, 262144]；墙钟 0.92 s |
| 推理前向，1089 视觉 token（seq 1106，{"max_soft_tokens": 1120}），full_logits | 16 | 27.414 | 35.193 | logits [16, 1106, 262144]；墙钟 0.922 s |
| LoRA r=32（language_model 内全部 nn.Linear），梯度检查点，前向+反向第 1 次，全序列 LM loss（seq 1122） | 8 | 46.576 | 49.029 | 可训练参数 52,666,368；LoRA 模块 276；墙钟 1.722 s；loss 20.829252 |
| LoRA r=32（language_model 内全部 nn.Linear），梯度检查点，前向+反向第 2 次，全序列 LM loss（seq 1122） | 8 | 46.607 | 49.207 | 可训练参数 52,666,368；LoRA 模块 276；墙钟 1.222 s；loss 20.829252 |
| HF generate，greedy，思考关，max_new_tokens=64 | 16 | 16.626 | 17.883 | 3 次：7.593, 7.612, 7.604 样本/s（墙钟 2.107, 2.102, 2.104 s）；new_tokens 矩阵 [16, 64]；prompt 1106 token |