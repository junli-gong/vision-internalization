# E1 事后读数 · 去偏字母读数 · qwen3.5-0.8b · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/c3/seed1，manifest sha256 431afcd7fb9dba78e45c9b6b86d97f4643f5c90d3a401c023f8d9bc70e1847ae，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.5499，95% CI [0.5340, 0.5652]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.4092，95% CI [0.3928, 0.4242]，n = 3590
- 去偏 − 原字母：0.1407，配对 95% CI [0.1237, 0.1588]；答案变了 1590 条（错→对 822，对→错 317）
- 正确选项的平均概率：去偏 0.3755，原字母 0.3782

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.4092 | [0.3928, 0.4242] | A | 0.6148 |
| 1 | 3590 | 0.4070 | [0.3925, 0.4226] | A | 0.5883 |
| 2 | 3590 | 0.4095 | [0.3941, 0.4254] | A | 0.5933 |
| 3 | 3590 | 0.4173 | [0.4019, 0.4345] | A | 0.5986 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
