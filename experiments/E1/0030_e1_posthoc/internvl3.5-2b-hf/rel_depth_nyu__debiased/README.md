# E1 事后读数 · 去偏字母读数 · internvl3.5-2b-hf · rel_depth_nyu · debiased

事后修订（D66-3，N22 → (b)；E1 预注册 §10.1，写于运行之前）。规则见 configs/e1_posthoc.json 的 debiased_letter。

- 模型：OpenGVLab/InternVL3_5-2B-HF@3f301ffcf3dcbb47893afae6650ea3e78d96fb6d（internvl3.5）；图像口径：{'tag': 'default', 'type': 'TilePolicy', 'crop_to_patches': False, 'max_patches': 1}
- 样本：E1 的 eval 子集，n = 3270；读数 6540 次（k 种循环移位，order_major 分批，批大小 64）
- 提示配置 sha256：f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452；事后配置 sha256：e02971cd232a57891e193f581eb2e09de5af164440f60409e8ba11383471b8cc
- 随机水平（逐题 1/k 的平均）：0.5000；k 的分布：{'2': 3270}

## 标题：去偏读数（严格读数）

- accuracy（all）：0.5853，95% CI [0.5676, 0.6021]，n = 3270

## 对照：原字母读数（同一次运行的 s = 0，批组成同主评测）

- accuracy（all）：0.5251，95% CI [0.5076, 0.5416]，n = 3270
- 去偏 − 原字母：0.0602，配对 95% CI [0.0419, 0.0780]；答案变了 1019 条（错→对 608，对→错 411）
- 正确选项的平均概率：去偏 0.5121，原字母 0.5253

## 每种顺序单独的读数

| 顺序 s | n | 准确率 | 95% CI | 最常见字母 | 占比 |
|---|---|---|---|---|---|
| 0 | 3270 | 0.5251 | [0.5076, 0.5416] | B | 0.9639 |
| 1 | 3270 | 0.4997 | [0.4835, 0.5177] | B | 0.9979 |

## 文件

- readings.jsonl：每次读数（样本 × 顺序）；predictions.jsonl：每条样本（由 readings.jsonl 算出）；metrics.json：全部指标；
- compute.json：计算记账（每次读数记一次调用、生成 1 个 token）；config.yaml：运行身份；meta.json；logs/。
