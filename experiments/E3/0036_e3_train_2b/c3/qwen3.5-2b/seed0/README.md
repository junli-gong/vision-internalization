# E3b 训练 · c3 · qwen3.5-2b · seed 0

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b；bf16）
- 监督：text（C3：稀疏文本）；图像口径 pixel_budget:300
- 训练：4000 步 × 16 条（微批 4 × 累积 4），LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002（token 行 0.0002，fp32_master_rows），warmup 120 步 + 余弦
- 混合：{'refseg': 1, 'depth': 1}；修正对比例：None
- 第一步损失 0.9480499780147267，最后一步损失 0.6608927490890667（4000 步）
- 检查点：/dev/shm/visint/ckpt_0036/c3/qwen3.5-2b/seed0/final/model（134613268 字节；只有 LoRA、token 行、manifest）

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：初始化时的 token 表（含行哈希）。
