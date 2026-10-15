# E1 事后读数 · 去偏字母读数 · qwen3.5-4b · spatial_cvbench · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed1，manifest sha256 7e0b95ee67a7700345f319e860c8104c28285bc96f8c3bc88ccefbdc10cf8e71，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 2588；读数 7144 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.4182；k 的分布：{'2': 1817, '4': 482, '5': 152, '6': 137}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.7697，95% CI [0.7535, 0.7855]，n = 2588

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.7562，95% CI [0.7392, 0.7740]，n = 2588
- 去偏 − 原字母：0.0135，配对 95% CI [0.0008, 0.0259]；答案变了 345 条（错→对 161，对→错 126）
- 正确选项的平均概率：去偏 0.7154，原字母 0.7222

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 2588 | 0.7562 | [0.7392, 0.7740] | A | 0.5298 |
| 1 | 2588 | 0.7454 | [0.7284, 0.7620] | A | 0.5402 |
| 2 | 771 | 0.4825 | [0.4488, 0.5149] | A | 0.3658 |
| 3 | 771 | 0.5071 | [0.4734, 0.5409] | A | 0.3567 |
| 4 | 289 | 0.2734 | [0.2215, 0.3218] | A | 0.3979 |
| 5 | 137 | 0.2190 | [0.1460, 0.2847] | A | 0.4818 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
