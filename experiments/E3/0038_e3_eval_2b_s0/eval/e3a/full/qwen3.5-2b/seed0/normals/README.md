# E3a 评测 · full · qwen3.5-2b · seed 0 · normals

- 检查点：model/manifest sha256 7fa48b7d1e5f5e9c049e702d32c2513db466189f105291448ff6a0d39593bf37，readout.json sha256 3d1085d774ba978eb62cd10c024a0faf457ee47b37f4d6ed56873a60e4933ab4
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 797e05f6ec08c68e…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | mean | median | within_11.25 |
|---|---|---|---|
| guided | 26.5682 [26.1970, 26.9165] | 21.2355 [20.7758, 21.6459] | 0.2682 [0.2605, 0.2761] |
| bilinear | 26.5791 [26.2066, 26.9270] | 21.1996 [20.7378, 21.6116] | 0.2693 [0.2616, 0.2773] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | mean | median | within_11.25 | 配对差 |
|---|---|---|---|---|
| shuffle | 44.8713 [44.3615, 45.3735] | 42.0515 [41.3346, 42.7826] | 0.1021 [0.0975, 0.1071] | mean -18.3032 [-18.8692, -17.7665]; within_11.25 0.1660 [0.1582, 0.1743] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
