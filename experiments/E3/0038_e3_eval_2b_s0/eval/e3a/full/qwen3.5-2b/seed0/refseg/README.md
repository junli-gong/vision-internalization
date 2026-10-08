# E3a 评测 · full · qwen3.5-2b · seed 0 · refseg

- 检查点：model/manifest sha256 7fa48b7d1e5f5e9c049e702d32c2513db466189f105291448ff6a0d39593bf37，readout.json sha256 3d1085d774ba978eb62cd10c024a0faf457ee47b37f4d6ed56873a60e4933ab4
- 模型：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc（qwen3.5；bf16；tiny = False）
- 评测集：n = 10634（id 指纹 5297041f81e4ac69…）；批大小 32
- 上采样：默认 guided；另报 bilinear

## 指标（95% CI）

| 上采样 | cIoU | gIoU | BIoU |
|---|---|---|---|
| guided | 0.1968 [0.1924, 0.2014] | 0.1609 [0.1576, 0.1642] | 0.0539 [0.0526, 0.0550] |
| bilinear | 0.1983 [0.1939, 0.2030] | 0.1626 [0.1593, 0.1659] | 0.0538 [0.0526, 0.0550] |

## 泄漏检查（guided；配对差 = 正常 − 检查，95% CI）

| 检查 | cIoU | gIoU | BIoU | 配对差 |
|---|---|---|---|---|
| shuffle | 0.0782 [0.0760, 0.0804] | 0.0666 [0.0649, 0.0685] | 0.0282 [0.0275, 0.0288] | gIoU 0.0943 [0.0906, 0.0979]; cIoU 0.1186 [0.1142, 0.1235]; BIoU 0.0257 [0.0244, 0.0270] |
| no_instruction | 0.1948 [0.1907, 0.1994] | 0.1610 [0.1579, 0.1643] | 0.0538 [0.0526, 0.0549] | gIoU -0.0001 [-0.0010, 0.0007]; cIoU 0.0020 [0.0009, 0.0032]; BIoU 0.0001 [-0.0004, 0.0005] |

- 判定规则留给 E3 预注册；这里只给读数（规则见 src/readout/evaluate.py 的模块说明）

## 文件

- scores.jsonl、arrays/：逐样本统计（配对比较用 src/readout/evaluate.paired_compare）；metrics.json：全部汇总；
- compute*.json：记账（含墙钟）；config.yaml：运行身份；meta.json；logs/：墙钟与每次调用的记录。
