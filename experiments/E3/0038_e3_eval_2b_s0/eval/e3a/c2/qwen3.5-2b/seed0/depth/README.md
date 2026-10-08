# E3a 评测 · c2 · qwen3.5-2b · seed 0 · depth

- 检查点：model/manifest sha256 23696c4de83fc38d57db3903d2e9f7f93d55796f525d638ca01b8a9edb997e04，readout.json sha256 b63c271e030ed75c724b14a1c0a62e4ace8f79abe26ba22298f3784d514038e8
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 4f06d8a06cfd3234…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | abs_rel | d1 |
|---|---|---|
| guided | 0.1019 [0.0985, 0.1050] | 0.9065 [0.8989, 0.9146] |
| bilinear | 0.1021 [0.0988, 0.1053] | 0.9059 [0.8983, 0.9140] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | abs_rel | d1 | 配对差 |
|---|---|---|---|
| shuffle | 0.3892 [0.3743, 0.4048] | 0.3954 [0.3825, 0.4093] | abs_rel -0.2873 [-0.3026, -0.2731]; d1 0.5111 [0.4964, 0.5243] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
