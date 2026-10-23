# E5 训练 · 起点 e3b_c3 · qwen3.5-2b · n4000 · seed 1

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16）；图像口径 pixel_budget:300
- 起点：e3b_c3（from_checkpoint；起点检查点 manifest 7c72a5007e461cca；起点 LoRA 合并进基座后注入新 LoRA）
- 数据：train 划分，数据量档 4000，任务 abcde（每类 800 条，共 4000 条），数据种子 1；清单 sha256 952e3f03314f61a6…
- 训练：步数下限 4000（按 2 个 epoch 是 500 步）= 4000 步，共 64000 条（16 个 epoch）；每步 16 条（微批 4 × 累积 4），LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002（token 行 0.0002，fp32_master_rows），warmup 120 步 + 余弦；受限交叉熵（允许集合见 config.yaml 的 tokens.e5）
- 第一步损失 0.10645974889157962，最后一步损失 0.00041451286824108186（4000 步）
- 配种子：match（起点的种子 1，--allow-seed-mismatch False）
- 检查点的选择（val 划分，每类 64 条，每 200 步一次；规则 mean_headline_v1；结构强制解码（D83-3））：选中第 4000 步（分数 0.4240）；val 曲线 200: 0.0031，400: 0.0031，600: 0.0339，800: 0.0514，1000: 0.0799，1200: 0.1423，1400: 0.1957，1600: 0.2115，1800: 0.3082，2000: 0.2825，2200: 0.3374，2400: 0.3565，2600: 0.3515，2800: 0.3983，3000: 0.4057，3200: 0.4111，3400: 0.4181，3600: 0.4224，3800: 0.4210，4000: 0.4240（全部读数在 val_selection.jsonl）
- 检查点：/home/mail/chenyuma/visint/ckpt/hpcgpu07/tmp0057/e3b_c3/qwen3.5-2b/n4000/seed1/final（134914920 字节；新 LoRA、网格 token 行、manifest；起点的引用在 e5_stack.json）

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：开训时的 token 表（含行哈希）。
