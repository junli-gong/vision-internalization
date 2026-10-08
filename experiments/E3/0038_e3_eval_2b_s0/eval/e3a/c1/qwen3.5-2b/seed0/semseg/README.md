# E3a 评测 · c1 · qwen3.5-2b · seed 0 · semseg

- 检查点：model/manifest sha256 157606aac2f5f947958778c800eb507f93df9113700928cfd04dd96e8316e733，readout.json sha256 638c60c11a5d9e9b9d8e7027e7f96e6220f3c610d77cf2c66e61621db17c2603
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 2000（id 指纹 ade78eeb8ee61ac8…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | miou | pixel_acc |
|---|---|---|
| guided | 0.0331 [0.0320, 0.0343] | 0.5173 [0.5078, 0.5270] |
| bilinear | 0.0330 [0.0318, 0.0342] | 0.5160 [0.5065, 0.5256] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | miou | pixel_acc | 配对差 |
|---|---|---|---|
| shuffle | 0.0128 [0.0121, 0.0135] | 0.2811 [0.2704, 0.2923] | miou 0.0203 [0.0190, 0.0215]; pixel_acc 0.2362 [0.2245, 0.2477] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
