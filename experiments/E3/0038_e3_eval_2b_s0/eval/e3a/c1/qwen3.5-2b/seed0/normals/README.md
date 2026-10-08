# E3a 评测 · c1 · qwen3.5-2b · seed 0 · normals

- 检查点：model/manifest sha256 157606aac2f5f947958778c800eb507f93df9113700928cfd04dd96e8316e733，readout.json sha256 638c60c11a5d9e9b9d8e7027e7f96e6220f3c610d77cf2c66e61621db17c2603
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 797e05f6ec08c68e…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | mean | median | within_11.25 |
|---|---|---|---|
| guided | 33.2484 [32.8735, 33.6027] | 29.9237 [29.4524, 30.4285] | 0.1516 [0.1465, 0.1575] |
| bilinear | 33.2516 [32.8735, 33.6053] | 29.9070 [29.4337, 30.4134] | 0.1524 [0.1473, 0.1583] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | mean | median | within_11.25 | 配对差 |
|---|---|---|---|---|
| shuffle | 42.1159 [41.6133, 42.6020] | 39.8304 [39.1878, 40.4866] | 0.0935 [0.0895, 0.0976] | mean -8.8675 [-9.2715, -8.4396]; within_11.25 0.0581 [0.0530, 0.0636] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
