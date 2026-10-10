# E3b 训练 · c3_warmup · qwen3.5-2b · seed 2

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b；bf16）
- 监督：warmup（网格格式预热：纯文字给坐标 → 网格，不给图像；C3+预热：先把起点 C3 的 LoRA 合并进 base，再在其上预热）；图像口径 pixel_budget:300
- 训练：至多 2000 步 × 16 条（微批 4 × 累积 4；实际 200 步），LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002（token 行 0.0002，fp32_master_rows），warmup 60 步 + 余弦
- 混合：{'refseg': 1, 'depth': 1}；修正对比例：None
- 第一步损失 4.8765869140625，最后一步损失 0.029823692993048576（200 步）
- 检查点：/dev/shm/visint/ckpt_0040/c3_warmup/qwen3.5-2b/seed2/final/model（269459964 字节；C3 与预热的 LoRA 按秩拼接成一个（r 64、alpha 128.0）、预热之后的 token 行、manifest）

## C3+预热的起点

- 起点：变体 c3、种子 2，manifest sha256 5f9da78e2bcf0609134f3f652d6ef7e62e2dd45235571636f06bc7794659dcf7
- 起点的 LoRA 在 torch.manual_seed 之前合并进 base，再加网格 token、注入预热的 LoRA；预热的数据、留出集、停止规则与基座上的相同
- 存法：A = [A_父; A_子]（lora_A.weight [r, in] 沿第 0 维），B = [B_父, B_子]（lora_B.weight [out, r] 沿第 1 维），R = r_父 + r_子，lora_alpha = (alpha / r) × R（缩放不变）：合并后的增量 scaling · B A = ΔW_父 + ΔW_子。浮点上与依次合并只差舍入（不保证逐位相同）；拼接后核对：check_rank_concat 通过（['parent', 'child']，372 个张量逐位相同）

## 停止规则（网格格式预热）

- 规则：每 100 步在留出集（200 条）上受限解码，格式错误率连续 2 次 < 0.01 就停；上限 2000 步
- 结果：stop_rule_met = True，停在第 200 步（format_error_below_threshold），共 2 次读数；最后一次格式错误率 0.0
- 逐次读数：warmup_eval.jsonl

## token 布局

- mask：{'<|e3_m0|>': 248077, '<|e3_m1|>': 248078}；深度 248079–248142（连续：True）；行分隔 248143；停止 248046

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：初始化时的 token 表（含行哈希）。
