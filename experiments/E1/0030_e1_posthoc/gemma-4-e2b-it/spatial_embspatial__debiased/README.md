# E1 事后读数 · 去偏字母读数 · gemma-4-e2b-it · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7（gemma4）；图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.3446，95% CI [0.3304, 0.3610]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.2585，95% CI [0.2440, 0.2730]，n = 3590
- 去偏 − 原字母：0.0861，配对 95% CI [0.0646, 0.1075]；答案变了 2570 条（错→对 880，对→错 571）
- 正确选项的平均概率：去偏 0.2557，原字母 0.2641

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.2585 | [0.2440, 0.2730] | B | 0.5905 |
| 1 | 3590 | 0.2393 | [0.2259, 0.2529] | B | 0.5886 |
| 2 | 3590 | 0.2490 | [0.2348, 0.2621] | B | 0.5841 |
| 3 | 3590 | 0.2660 | [0.2515, 0.2808] | B | 0.5844 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
