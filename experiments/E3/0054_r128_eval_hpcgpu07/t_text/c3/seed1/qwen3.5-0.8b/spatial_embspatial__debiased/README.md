# E1 事后读数 · 去偏字母读数 · qwen3.5-0.8b · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/c3/seed1，manifest sha256 987e23a658d224acb42aea0cfec9f66fc377685740675b3492ad207f65ea879f，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.2702，95% CI [0.2560, 0.2847]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.2618，95% CI [0.2479, 0.2758]，n = 3590
- 去偏 − 原字母：0.0084，配对 95% CI [-0.0103, 0.0295]；答案变了 2706 条（错→对 747，对→错 717）
- 正确选项的平均概率：去偏 0.2507，原字母 0.2604

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.2618 | [0.2479, 0.2758] | A | 0.9964 |
| 1 | 3590 | 0.2510 | [0.2365, 0.2655] | A | 0.9958 |
| 2 | 3590 | 0.2345 | [0.2214, 0.2482] | A | 0.9955 |
| 3 | 3590 | 0.2524 | [0.2384, 0.2660] | A | 0.9964 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
