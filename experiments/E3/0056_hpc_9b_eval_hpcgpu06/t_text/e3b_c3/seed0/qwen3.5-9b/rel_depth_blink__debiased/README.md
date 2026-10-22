# E1 事后读数 · 去偏字母读数 · qwen3.5-9b · rel_depth_blink · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed0，manifest sha256 02f416c93689a72da0f7c941fbfaf3d2320e6afe4448e38248710630fc9de3f6，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 112；读数 224 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.5000；k 的分布：{'2': 112}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.9464，95% CI [0.9018, 0.9821]，n = 112

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.8661，95% CI [0.8036, 0.9286]，n = 112
- 去偏 − 原字母：0.0804，配对 95% CI [0.0268, 0.1429]；答案变了 11 条（错→对 10，对→错 1）
- 正确选项的平均概率：去偏 0.7968，原字母 0.8218

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 112 | 0.8661 | [0.8036, 0.9286] | A | 0.6339 |
| 1 | 112 | 0.8304 | [0.7589, 0.8929] | A | 0.6161 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
