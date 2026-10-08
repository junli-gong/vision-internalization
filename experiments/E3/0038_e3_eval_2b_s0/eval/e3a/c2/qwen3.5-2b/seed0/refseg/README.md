# E3a 评测 · c2 · qwen3.5-2b · seed 0 · refseg

- 检查点：model/manifest sha256 23696c4de83fc38d57db3903d2e9f7f93d55796f525d638ca01b8a9edb997e04，readout.json sha256 b63c271e030ed75c724b14a1c0a62e4ace8f79abe26ba22298f3784d514038e8
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 10634（id 指纹 5297041f81e4ac69…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | cIoU | gIoU | BIoU |
|---|---|---|---|
| guided | 0.2306 [0.2261, 0.2350] | 0.2083 [0.2046, 0.2118] | 0.0922 [0.0904, 0.0940] |
| bilinear | 0.2339 [0.2294, 0.2383] | 0.2119 [0.2082, 0.2153] | 0.0954 [0.0936, 0.0971] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | cIoU | gIoU | BIoU | 配对差 |
|---|---|---|---|---|
| shuffle | 0.0842 [0.0822, 0.0863] | 0.0720 [0.0703, 0.0737] | 0.0340 [0.0333, 0.0347] | gIoU 0.1363 [0.1327, 0.1398]; cIoU 0.1464 [0.1419, 0.1508]; BIoU 0.0582 [0.0564, 0.0600] |
| no_instruction | 0.2306 [0.2261, 0.2350] | 0.2083 [0.2046, 0.2118] | 0.0922 [0.0904, 0.0940] | gIoU 0.0000 [0.0000, 0.0000]; cIoU 0.0000 [0.0000, 0.0000]; BIoU 0.0000 [0.0000, 0.0000] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
