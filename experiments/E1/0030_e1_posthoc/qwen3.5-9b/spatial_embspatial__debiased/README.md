# E1 事后读数 · 去偏字母读数 · qwen3.5-9b · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.8000，95% CI [0.7875, 0.8134]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.7724，95% CI [0.7596, 0.7855]，n = 3590
- 去偏 − 原字母：0.0276，配对 95% CI [0.0181, 0.0371]；答案变了 390 条（错→对 217，对→错 118）
- 正确选项的平均概率：去偏 0.7416，原字母 0.7443

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.7724 | [0.7596, 0.7855] | D | 0.3092 |
| 1 | 3590 | 0.7682 | [0.7549, 0.7825] | D | 0.3134 |
| 2 | 3590 | 0.7710 | [0.7574, 0.7847] | D | 0.3081 |
| 3 | 3590 | 0.7669 | [0.7546, 0.7805] | D | 0.3011 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
