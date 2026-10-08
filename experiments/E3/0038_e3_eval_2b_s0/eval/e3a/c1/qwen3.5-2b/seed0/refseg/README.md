# E3a 评测 · c1 · qwen3.5-2b · seed 0 · refseg

- 检查点：model/manifest sha256 157606aac2f5f947958778c800eb507f93df9113700928cfd04dd96e8316e733，readout.json sha256 638c60c11a5d9e9b9d8e7027e7f96e6220f3c610d77cf2c66e61621db17c2603
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 10634（id 指纹 5297041f81e4ac69…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | cIoU | gIoU | BIoU |
|---|---|---|---|
| guided | 0.1316 [0.1279, 0.1354] | 0.1161 [0.1128, 0.1194] | 0.0384 [0.0374, 0.0395] |
| bilinear | 0.1330 [0.1293, 0.1367] | 0.1174 [0.1141, 0.1207] | 0.0382 [0.0372, 0.0392] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | cIoU | gIoU | BIoU | 配对差 |
|---|---|---|---|---|
| shuffle | 0.0620 [0.0599, 0.0642] | 0.0544 [0.0525, 0.0564] | 0.0229 [0.0222, 0.0236] | gIoU 0.0616 [0.0582, 0.0652]; cIoU 0.0696 [0.0655, 0.0739]; BIoU 0.0155 [0.0143, 0.0167] |
| no_instruction | 0.1114 [0.1079, 0.1149] | 0.0971 [0.0943, 0.1000] | 0.0331 [0.0322, 0.0340] | gIoU 0.0189 [0.0176, 0.0202]; cIoU 0.0202 [0.0189, 0.0215]; BIoU 0.0053 [0.0048, 0.0059] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
