# E1 事后读数 · 去偏字母读数 · qwen3.5-4b · spatial_cvbench · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/c3/seed0，manifest sha256 b3020ee5928d0dac0e3b25ded412c6c4ba16186677c8952a2447d79bf3835768，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 2588；读数 7144 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.4182；k 的分布：{'2': 1817, '4': 482, '5': 152, '6': 137}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.8250，95% CI [0.8099, 0.8396]，n = 2588

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.7987，95% CI [0.7832, 0.8145]，n = 2588
- 去偏 − 原字母：0.0263，配对 95% CI [0.0158, 0.0375]；答案变了 271 条（错→对 148，对→错 80）
- 正确选项的平均概率：去偏 0.7432，原字母 0.7532

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 2588 | 0.7987 | [0.7832, 0.8145] | A | 0.5100 |
| 1 | 2588 | 0.7913 | [0.7755, 0.8076] | A | 0.5174 |
| 2 | 771 | 0.5888 | [0.5538, 0.6226] | A | 0.3217 |
| 3 | 771 | 0.5927 | [0.5577, 0.6278] | A | 0.3346 |
| 4 | 289 | 0.3945 | [0.3390, 0.4499] | A | 0.3114 |
| 5 | 137 | 0.2117 | [0.1460, 0.2774] | A | 0.4672 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
