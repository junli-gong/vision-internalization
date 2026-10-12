# E1 事后读数 · 去偏字母读数 · qwen3.5-0.8b · spatial_cvbench · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/c3/seed0，manifest sha256 fae595c013022f99f115e87a398286cb37d8106899d1bfabc388aaa9103f5f8c，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 2588；读数 7144 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.4182；k 的分布：{'2': 1817, '4': 482, '5': 152, '6': 137}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.6781，95% CI [0.6600, 0.6963]，n = 2588

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.5046，95% CI [0.4853, 0.5240]，n = 2588
- 去偏 − 原字母：0.1735，配对 95% CI [0.1507, 0.1959]；答案变了 1125 条（错→对 708，对→错 259）
- 正确选项的平均概率：去偏 0.5056，原字母 0.5107

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 2588 | 0.5046 | [0.4853, 0.5240] | A | 0.8601 |
| 1 | 2588 | 0.4961 | [0.4768, 0.5155] | A | 0.8628 |
| 2 | 771 | 0.3437 | [0.3113, 0.3749] | A | 0.7380 |
| 3 | 771 | 0.3658 | [0.3320, 0.3982] | A | 0.7004 |
| 4 | 289 | 0.2284 | [0.1799, 0.2768] | A | 0.8097 |
| 5 | 137 | 0.1460 | [0.0876, 0.2044] | A | 0.8321 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
