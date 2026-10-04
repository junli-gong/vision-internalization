# E1 事后读数 · 去偏字母读数 · gemma-4-e4b-it · rel_depth_nyu · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2（gemma4）；图像口径：{'tag': 'default', 'type': 'SoftTokenBudget', 'max_soft_tokens': 280}
- 样本：E1 的 eval 子集，n = 3270；读数 6540 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc
- 随机水平（逐题 1/k 的平均）：0.5000；k 的分布：{'2': 3270}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.4945，95% CI [0.4768, 0.5110]，n = 3270

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.5361，95% CI [0.5199, 0.5547]，n = 3270
- 去偏 − 原字母：-0.0416，配对 95% CI [-0.0743, -0.0128]；答案变了 2520 条（错→对 1192，对→错 1328）
- 正确选项的平均概率：去偏 0.5062，原字母 0.5211

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3270 | 0.5361 | [0.5199, 0.5547] | A | 0.7976 |
| 1 | 3270 | 0.4976 | [0.4795, 0.5144] | A | 0.9920 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
