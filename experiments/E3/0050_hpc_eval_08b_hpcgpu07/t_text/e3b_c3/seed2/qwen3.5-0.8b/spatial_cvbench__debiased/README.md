# E1 事后读数 · 去偏字母读数 · qwen3.5-0.8b · spatial_cvbench · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed2，manifest sha256 784f17c8ff4a0263b3132dcaeb01ca90c0208b6c1703d1922acf5b6ef48b0fbf，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 2588；读数 7144 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.4182；k 的分布：{'2': 1817, '4': 482, '5': 152, '6': 137}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.5560，95% CI [0.5367, 0.5746]，n = 2588

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.4347，95% CI [0.4162, 0.4532]，n = 2588
- 去偏 − 原字母：0.1213，配对 95% CI [0.0970, 0.1453]；答案变了 1274 条（错→对 666，对→错 352）
- 正确选项的平均概率：去偏 0.4436，原字母 0.4452

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 2588 | 0.4347 | [0.4162, 0.4532] | A | 0.9204 |
| 1 | 2588 | 0.4258 | [0.4057, 0.4451] | A | 0.9316 |
| 2 | 771 | 0.2348 | [0.2062, 0.2633] | A | 0.7795 |
| 3 | 771 | 0.2529 | [0.2231, 0.2815] | A | 0.7678 |
| 4 | 289 | 0.2215 | [0.1730, 0.2734] | A | 0.8581 |
| 5 | 137 | 0.1460 | [0.0876, 0.2117] | A | 0.8467 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
