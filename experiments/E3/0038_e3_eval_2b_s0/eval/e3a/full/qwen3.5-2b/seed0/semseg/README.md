# E3a 评测 · full · qwen3.5-2b · seed 0 · semseg

- 检查点：model/manifest sha256 7fa48b7d1e5f5e9c049e702d32c2513db466189f105291448ff6a0d39593bf37，readout.json sha256 3d1085d774ba978eb62cd10c024a0faf457ee47b37f4d6ed56873a60e4933ab4
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 2000（id 指纹 ade78eeb8ee61ac8…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | miou | pixel_acc |
|---|---|---|
| guided | 0.0624 [0.0604, 0.0643] | 0.6168 [0.6078, 0.6262] |
| bilinear | 0.0621 [0.0601, 0.0640] | 0.6153 [0.6064, 0.6246] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | miou | pixel_acc | 配对差 |
|---|---|---|---|
| shuffle | 0.0156 [0.0148, 0.0164] | 0.2556 [0.2459, 0.2653] | miou 0.0468 [0.0448, 0.0489]; pixel_acc 0.3612 [0.3503, 0.3725] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
