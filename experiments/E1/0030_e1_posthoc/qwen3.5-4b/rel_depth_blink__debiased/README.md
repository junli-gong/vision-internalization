# E1 事后读数 · 去偏字母读数 · qwen3.5-4b · rel_depth_blink · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 样本：E1 的 eval 子集，n = 112；读数 224 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc
- 随机水平（逐题 1/k 的平均）：0.5000；k 的分布：{'2': 112}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.8304，95% CI [0.7589, 0.8929]，n = 112

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.5179，95% CI [0.4283, 0.6161]，n = 112
- 去偏 − 原字母：0.3125，配对 95% CI [0.2143, 0.4018]；答案变了 41 条（错→对 38，对→错 3）
- 正确选项的平均概率：去偏 0.5151，原字母 0.5639

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 112 | 0.5179 | [0.4283, 0.6161] | A | 1.0000 |
| 1 | 112 | 0.4821 | [0.3839, 0.5717] | A | 1.0000 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
