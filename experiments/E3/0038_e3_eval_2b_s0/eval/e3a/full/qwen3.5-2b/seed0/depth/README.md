# E3a 评测 · full · qwen3.5-2b · seed 0 · depth

- 检查点：model/manifest sha256 7fa48b7d1e5f5e9c049e702d32c2513db466189f105291448ff6a0d39593bf37，readout.json sha256 3d1085d774ba978eb62cd10c024a0faf457ee47b37f4d6ed56873a60e4933ab4
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 4f06d8a06cfd3234…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | abs_rel | d1 |
|---|---|---|
| guided | 0.0988 [0.0956, 0.1019] | 0.9117 [0.9048, 0.9195] |
| bilinear | 0.0991 [0.0959, 0.1022] | 0.9111 [0.9042, 0.9189] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | abs_rel | d1 | 配对差 |
|---|---|---|---|
| shuffle | 0.3908 [0.3761, 0.4059] | 0.3932 [0.3805, 0.4069] | abs_rel -0.2920 [-0.3070, -0.2777]; d1 0.5184 [0.5042, 0.5321] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
