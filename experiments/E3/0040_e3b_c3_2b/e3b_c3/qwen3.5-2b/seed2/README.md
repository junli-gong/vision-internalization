# E3b 训练 · e3b_c3 · qwen3.5-2b · seed 2

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；适配器 qwen3.5-2b；bf16）
- 监督：grid+text（E3b+C3：C3 的文本样本流 + E3b 的网格样本流（GT 网格 + 修正对））；图像口径 pixel_budget:300
- 训练：8000 步 × 16 条（文本 8 + 网格 8；微批 4 × 累积 4，每个微批只含一种监督：{'text': 2, 'grid': 2}，先后 ['text', 'grid']；日程长度是 C3、E3b 的 2 倍），LoRA r 32 / alpha 64 / dropout 0.05，AdamW lr 0.0002（token 行 0.0002，fp32_master_rows），warmup 240 步 + 余弦
- 混合：{'refseg': 1, 'depth': 1}；修正对比例：0.5
- 第一步损失 9.276670475211938，最后一步损失 0.9161335909580643（8000 步）
- 检查点：/dev/shm/visint/ckpt_0040/e3b_c3/qwen3.5-2b/seed2/final/model（134903931 字节；只有 LoRA、token 行、manifest）

## E3b+C3 的两条样本流

- 第 t 步：文本 = c3 同一种子的样本流第 8t … 8t + 7 个槽位；网格 = e3b 同一种子的样本流第 8t … 8t + 7 个槽位（含修正对）。全程两条流各 64000 条 = c3、e3b 单独训练的全部样本，顺序相同
- 两条流的槽位与样本的对应只由种子与 k 决定，所以同一步里同一张图以两种监督各出现一次（不另换数据种子）
- 损失：每步 = 1 × 文本的逐 token 平均 + 1 × 网格的逐 token 平均（事先固定，D83-2）；train_log 的 loss 是这个和，loss_by_task 的 text、grid 两项是两种各自的平均
- 学习率：warmup 240 步 + 余弦，按 8000 步排，日程长度是 c3、e3b（4000 步）的 2 倍
- 本次合计（train_log 的 by_supervision）：{'text': {'n_samples': 63998, 'n_supervised_tokens': 6876338}, 'grid': {'n_samples': 64000, 'n_supervised_tokens': 56853087}}

## token 布局

- mask：{'<|e3_m0|>': 248077, '<|e3_m1|>': 248078}；深度 248079–248142（连续：True）；行分隔 248143；停止 248046

## 文件

- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；
- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：初始化时的 token 表（含行哈希）。
