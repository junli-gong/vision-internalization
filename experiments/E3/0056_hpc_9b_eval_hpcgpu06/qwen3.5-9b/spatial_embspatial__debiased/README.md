# E1 事后读数 · 去偏字母读数 · qwen3.5-9b · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.7992，95% CI [0.7863, 0.8123]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.7716，95% CI [0.7588, 0.7847]，n = 3590
- 去偏 − 原字母：0.0276，配对 95% CI [0.0178, 0.0376]；答案变了 398 条（错→对 222，对→错 123）
- 正确选项的平均概率：去偏 0.7416，原字母 0.7442

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.7716 | [0.7588, 0.7847] | D | 0.3117 |
| 1 | 3590 | 0.7688 | [0.7552, 0.7827] | D | 0.3117 |
| 2 | 3590 | 0.7691 | [0.7557, 0.7827] | D | 0.3084 |
| 3 | 3590 | 0.7688 | [0.7563, 0.7822] | D | 0.2989 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
