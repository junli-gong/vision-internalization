# E1 事后读数 · 去偏字母读数 · qwen3.5-2b · spatial_embspatial · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed0，manifest sha256 bf73188b8c88c7607de553d2a0256b8c9cfc1dcd5a8f6852032c95ee1fc70cd1，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 3590；读数 14360 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平（逐题 1/k 的平均）：0.2500；k 的分布：{'4': 3590}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.5474，95% CI [0.5309, 0.5638]，n = 3590

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.4799，95% CI [0.4643, 0.4961]，n = 3590
- 去偏 − 原字母：0.0674，配对 95% CI [0.0515, 0.0833]；答案变了 1257 条（错→对 542，对→错 300）
- 正确选项的平均概率：去偏 0.3907，原字母 0.3943

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3590 | 0.4799 | [0.4643, 0.4961] | A | 0.5593 |
| 1 | 3590 | 0.4721 | [0.4557, 0.4891] | A | 0.5604 |
| 2 | 3590 | 0.4593 | [0.4435, 0.4763] | A | 0.5641 |
| 3 | 3590 | 0.4682 | [0.4526, 0.4844] | A | 0.5627 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
