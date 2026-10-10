# E1 事后读数 · 对调标记的去偏读数 · qwen3.5-2b · rel_depth_nyu · letter_swap

事后修订（D72-2，N26 → 问题 1 选 (a)；DEV-0036）。规则见 configs/e1_posthoc.json 的 letter_swap：每题读两张图（original、labels_swapped：图上 A、B 两个标签的字形互换，位置与环不变；提示不变），字母概率按物理点取平均。

- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5）；图像口径：{'tag': 'default', 'type': 'PixelBudget', 'pixel_budget_tokens': 300, 'min_pixels': 307200, 'max_pixels': 307200}
- 训练后的检查点：标签 t_text/e3b_c3/seed0，manifest sha256 bf73188b8c88c7607de553d2a0256b8c9cfc1dcd5a8f6852032c95ee1fc70cd1，load_trained(merge=True)（合并核对 {'criteria_version': 'v1', 'n': 200, 'checks': ['merge']}）
- 样本：E1 的 eval 子集，n = 3270；读数 6540 次（两张图，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
- 随机水平：0.5000

## 标题：对调标记的去偏读数（严格读数）

- accuracy（all）：0.8768，95% CI [0.8654, 0.8878]，n = 3270

## 对照：原字母读数（同一次运行 original 那张图，批组成同主评测）

- accuracy（all）：0.7820，95% CI [0.7676, 0.7960]，n = 3270
- 对调去偏 − 原字母：0.0948，配对 95% CI [0.0807, 0.1083]；答案变了 566 条（错→对 438，对→错 128）

## 两张图各自的读数

| 图 | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| original | 3270 | 0.7820 | [0.7676, 0.7960] | A | 0.6379 |
| labels_swapped | 3270 | 0.7749 | [0.7608, 0.7893] | A | 0.6523 |

## 文件

- readings.jsonl：每次读数（样本 × 图；另记图名与像素 sha256）；predictions.jsonl：每条样本；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
