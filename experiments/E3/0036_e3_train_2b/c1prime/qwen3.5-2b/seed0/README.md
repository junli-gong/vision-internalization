# E3b 训练 · c1prime · qwen3.5-2b · seed 0

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b；bf16）
- 监督：grid（GT 网格 + 修正对；C1′：冻结 LLM，只训练网格 token 的行）；图像口径 pixel_budget:300
- 训练：4000 步 × 16 条（微批 4 × 累积 4），不注入 LoRA，AdamW 只有 token 行一组：lr 0.0002（fp32_master_rows），warmup 120 步 + 余弦
- 混合：{'refseg': 1, 'depth': 1}；修正对比例：0.5
- 第一步损失 8.413949885955095，最后一步损失 0.34729447877279807（4000 步）
- 检查点：/dev/shm/visint/ckpt_0036/c1prime/qwen3.5-2b/seed0/final/model（291750 字节；只有 token 行、manifest）

## token 布局

- mask：{'<|e3_m0|>': 248077, '<|e3_m1|>': 248078}；深度 248079–248142（连续：True）；行分隔 248143；停止 248046

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：初始化时的 token 表（含行哈希）。
