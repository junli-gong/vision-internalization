# E5 训练 · 起点 base_warmup · qwen3.5-2b · n64000 · seed 0

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16）；图像口径 pixel_budget:300
- 起点：base_warmup（from_checkpoint；起点检查点 manifest 58ba382e9cd0b5bf；起点 LoRA 合并进基座后注入新 LoRA）
- 数据：train 划分，数据量档 64000，任务 abcde（每类 12800 条，共 64000 条），数据种子 0；清单 sha256 69c71b092b7bd645…
- 训练：2 个 epoch = 8000 步；每步 16 条（微批 4 × 累积 4），LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002（token 行 0.0002，fp32_master_rows），warmup 240 步 + 余弦；受限交叉熵（允许集合见 config.yaml 的 tokens.e5）
- 第一步损失 0.21119821308464404，最后一步损失 0.04112274746006124（8000 步）
- 配种子：match（起点的种子 0，--allow-seed-mismatch False）
- 检查点的选择：取最后一步（这一档不做 val 选择）
- 检查点：/home/mail/chenyuma/visint/ckpt/hpcgpu06/tmp0051/base_warmup/qwen3.5-2b/n64000/seed0/final（134908500 字节；新 LoRA、网格 token 行、manifest；起点的引用在 e5_stack.json）

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：开训时的 token 表（含行哈希）。
