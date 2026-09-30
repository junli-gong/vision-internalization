| 测量 | batch | 峰值 allocated GiB | 峰值 reserved GiB | 备注 |
|---|---|---|---|---|
| 仅加载权重（bf16） | — | 4.376 | 4.67 | |
| 推理前向，1280 视觉 token（seq 1297，{"crop_to_patches": true, "max_patches": 4}），logits_to_keep=1 | 1 | 4.567 | 4.674 | logits [1, 1, 151936]；墙钟 0.498 s |
| 推理前向，1280 视觉 token（seq 1297，{"crop_to_patches": true, "max_patches": 4}），full_logits | 1 | 4.807 | 5.041 | logits [1, 1297, 151936]；墙钟 0.021 s |
| 推理前向，1280 视觉 token（seq 1297，{"crop_to_patches": true, "max_patches": 4}），logits_to_keep=1 | 16 | 6.801 | 7.445 | logits [16, 1, 151936]；墙钟 0.271 s |
| 推理前向，1280 视觉 token（seq 1297，{"crop_to_patches": true, "max_patches": 4}），full_logits | 16 | 10.629 | 13.318 | logits [16, 1297, 151936]；墙钟 0.218 s |
| LoRA r=32（language_model 内全部 nn.Linear），梯度检查点，前向+反向第 1 次，全序列 LM loss（seq 1313） | 8 | 28.79 | 30.252 | 可训练参数 34,865,152；LoRA 模块 196；墙钟 2.83 s；loss 20.498777 |
| LoRA r=32（language_model 内全部 nn.Linear），梯度检查点，前向+反向第 2 次，全序列 LM loss（seq 1313） | 8 | 28.821 | 30.293 | 可训练参数 34,865,152；LoRA 模块 196；墙钟 0.627 s；loss 20.498777 |
| HF generate，greedy，思考关，max_new_tokens=64 | 16 | 7.936 | 8.553 | 3 次：18.806, 18.771, 18.888 样本/s（墙钟 0.851, 0.852, 0.847 s）；new_tokens 矩阵 [16, 64]；prompt 1297 token |