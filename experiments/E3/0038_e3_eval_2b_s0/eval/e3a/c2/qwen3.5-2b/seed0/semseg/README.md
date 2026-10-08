# E3a 评测 · c2 · qwen3.5-2b · seed 0 · semseg

- 检查点：model/manifest sha256 23696c4de83fc38d57db3903d2e9f7f93d55796f525d638ca01b8a9edb997e04，readout.json sha256 b63c271e030ed75c724b14a1c0a62e4ace8f79abe26ba22298f3784d514038e8
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 2000（id 指纹 ade78eeb8ee61ac8…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | miou | pixel_acc |
|---|---|---|
| guided | 0.1735 [0.1681, 0.1795] | 0.7119 [0.7034, 0.7202] |
| bilinear | 0.1728 [0.1674, 0.1786] | 0.7092 [0.7008, 0.7176] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | miou | pixel_acc | 配对差 |
|---|---|---|---|
| shuffle | 0.0212 [0.0197, 0.0225] | 0.2347 [0.2256, 0.2439] | miou 0.1523 [0.1470, 0.1583]; pixel_acc 0.4772 [0.4673, 0.4879] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
