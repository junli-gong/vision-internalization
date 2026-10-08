# E3a 评测 · c1 · qwen3.5-2b · seed 0 · depth

- 检查点：model/manifest sha256 157606aac2f5f947958778c800eb507f93df9113700928cfd04dd96e8316e733，readout.json sha256 638c60c11a5d9e9b9d8e7027e7f96e6220f3c610d77cf2c66e61621db17c2603
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 654（id 指纹 4f06d8a06cfd3234…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | abs_rel | d1 |
|---|---|---|
| guided | 0.1415 [0.1372, 0.1458] | 0.8137 [0.8044, 0.8239] |
| bilinear | 0.1417 [0.1373, 0.1460] | 0.8133 [0.8040, 0.8236] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | abs_rel | d1 | 配对差 |
|---|---|---|---|
| shuffle | 0.3764 [0.3609, 0.3920] | 0.4051 [0.3921, 0.4186] | abs_rel -0.2349 [-0.2494, -0.2206]; d1 0.4086 [0.3943, 0.4225] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
