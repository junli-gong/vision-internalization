# E1 事后读数 · 去偏字母读数 · qwen3.5-0.8b · spatial_cvbench · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed1，manifest sha256 c650ce685326451887368b4441411dae859f69f3405fe9ef94be79a95146d760，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 2588；读数 7144 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.4182；k 的分布：{'2': 1817, '4': 482, '5': 152, '6': 137}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.5286，95% CI [0.5097, 0.5479]，n = 2588

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.4274，95% CI [0.4084, 0.4467]，n = 2588
- 去偏 − 原字母：0.1012，配对 95% CI [0.0792, 0.1225]；答案变了 1117 条（错→对 552，对→错 290）
- 正确选项的平均概率：去偏 0.4313，原字母 0.4382

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 2588 | 0.4274 | [0.4084, 0.4467] | A | 0.9923 |
| 1 | 2588 | 0.4216 | [0.4014, 0.4409] | A | 0.9957 |
| 2 | 771 | 0.2114 | [0.1842, 0.2387] | A | 0.9935 |
| 3 | 771 | 0.2114 | [0.1842, 0.2399] | A | 0.9909 |
| 4 | 289 | 0.2145 | [0.1661, 0.2630] | A | 0.9862 |
| 5 | 137 | 0.1606 | [0.1022, 0.2190] | A | 0.9854 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
