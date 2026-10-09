# E3a 训练 · c2 · qwen3.5-2b · seed 2

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b；bf16）；图像口径 pixel_budget:300
- 变体：c2；参数组 ['readout']；LoRA None
- 训练：4000 步 × 16 条（微批 4 × 累积 4），warmup 120 步 + 余弦；混合 {'refseg': 1, 'depth': 1, 'semseg': 1, 'normals': 1}；任务权重 {'refseg': 1.0, 'depth': 1.0, 'semseg': 1.0, 'normals': 1.0}
- 读出：解码器（含混合与类别投影）3806464 个参数（上限 50000000）；线性头 {'refseg': 65, 'depth': 65, 'normals': 195}（每个上限 1000）；来源 ['vision_features', 'llm_input_embeddings:class_names']
- 第一步损失 3.468180298805237，最后一步损失 0.7598741799592972（4000 步）
- 混合权重：训练前 None；训练后 None
- 检查点：/dev/shm/visint/ckpt_0036/c2/qwen3.5-2b/seed2/final（15235827 字节；model/ + readout/）

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本。
