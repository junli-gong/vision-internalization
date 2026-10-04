# E1 事后读数 · 去偏字母读数 · qwen3.5-4b · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.7961，95% CI [0.7833, 0.8089]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.7376，95% CI [0.7226, 0.7518]，n = 3590
- 去偏 − 原字母：0.0585，配对 95% CI [0.0473, 0.0699]；答案变了 529 条（错→对 318，对→错 108）
- 正确选项的平均概率：去偏 0.6783，原字母 0.6743

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.7376 | [0.7226, 0.7518] | A | 0.3370 |
| 1 | 3590 | 0.7440 | [0.7304, 0.7574] | A | 0.3326 |
| 2 | 3590 | 0.7343 | [0.7206, 0.7485] | A | 0.3192 |
| 3 | 3590 | 0.7490 | [0.7345, 0.7627] | A | 0.3365 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
