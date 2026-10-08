# E3a 评测 · c2 · qwen3.5-2b · seed 0 · normals

- 检查点：model/manifest sha256 23696c4de83fc38d57db3903d2e9f7f93d55796f525d638ca01b8a9edb997e04，readout.json sha256 b63c271e030ed75c724b14a1c0a62e4ace8f79abe26ba22298f3784d514038e8
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 797e05f6ec08c68e…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | mean | median | within_11.25 |
|---|---|---|---|
| guided | 27.1335 [26.7908, 27.4605] | 22.1590 [21.7171, 22.5709] | 0.2558 [0.2489, 0.2632] |
| bilinear | 27.1589 [26.8163, 27.4868] | 22.1363 [21.6969, 22.5453] | 0.2565 [0.2497, 0.2638] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | mean | median | within_11.25 | 配对差 |
|---|---|---|---|---|
| shuffle | 44.6358 [44.1390, 45.1138] | 41.8755 [41.1916, 42.5464] | 0.1084 [0.1035, 0.1130] | mean -17.5023 [-18.0823, -16.9991]; within_11.25 0.1474 [0.1409, 0.1551] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
