# E1 事后读数 · 去偏字母读数 · qwen3.5-0.8b · spatial_cvbench · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed1，manifest sha256 6d1edd0da17be1a41a9a4acb8f75c0f2ef06e0d05707dadb36e4dc8ef81fd3f9，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 2588；读数 7144 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.4182；k 的分布：{'2': 1817, '4': 482, '5': 152, '6': 137}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.4042，95% CI [0.3856, 0.4227]，n = 2588

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.4185，95% CI [0.3999, 0.4366]，n = 2588
- 去偏 − 原字母：-0.0143，配对 95% CI [-0.0367, 0.0081]；答案变了 1206 条（错→对 429，对→错 466）
- 正确选项的平均概率：去偏 0.4189，原字母 0.4200

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 2588 | 0.4185 | [0.3999, 0.4366] | B | 0.6113 |
| 1 | 2588 | 0.4169 | [0.3980, 0.4374] | B | 0.5189 |
| 2 | 771 | 0.2166 | [0.1881, 0.2464] | B | 0.6991 |
| 3 | 771 | 0.2335 | [0.2049, 0.2620] | B | 0.7147 |
| 4 | 289 | 0.1938 | [0.1453, 0.2388] | B | 0.6125 |
| 5 | 137 | 0.1533 | [0.0949, 0.2117] | A | 0.6788 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
