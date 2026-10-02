#### 每种设置：同一进程里与第 1 遍不同的条数（第 4 批，64 条；每遍一列）

| 设置 | pid | 环境 | deterministic | 遍数 | 每遍与第 1 遍不同的条数 | 不同的位置 | 每遍 generate 墙钟 s：中位 / 最小 / 最大 | 模型加载 s |
|---|---|---|---|---|---|---|---|---|
| S1 | 1009874 | （同 RUN-0024） | False | 20 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 无 | 1.391 / 1.382 / 1.568 | 5.8 |
| S2a | 1010271 | （同 RUN-0024） | False | 3 | [0, 0, 0] | 无 | 1.414 / 1.411 / 1.597 | 5.22 |
| S2b | 1010551 | （同 RUN-0024） | False | 3 | [0, 0, 0] | 无 | 1.381 / 1.379 / 1.568 | 5.24 |
| S2c | 1010748 | （同 RUN-0024） | False | 3 | [0, 0, 0] | 无 | 1.387 / 1.386 / 1.573 | 10.34 |
| S3 | 1010943 | CUBLAS_WORKSPACE_CONFIG=:4096:8 | True | 20 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 无 | 1.553 / 1.549 / 1.734 | 5.1 |
| S4 | 1011190 | CUBLAS_WORKSPACE_CONFIG=:4096:8, FLA_CACHE_MODE=default | True | 20 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] | 无 | 1.551 / 1.545 / 1.731 | 4.67 |

#### 跨进程与跨设置：各进程第 1 遍的 64 条 token 序列（与 S1 第 1 遍比较；sha256 是 64 行 token id 的 JSON）

| 设置 | 第 1 遍 sha256（前 16 位） | 与 S1 第 1 遍不同的条数 | 不同的 id |
|---|---|---|---|
| S1 | 5a220fe5b20a3d19 | 0 | 无 |
| S2a | 5a220fe5b20a3d19 | 0 | 无 |
| S2b | 5a220fe5b20a3d19 | 0 | 无 |
| S2c | 5a220fe5b20a3d19 | 0 | 无 |
| S3 | 5a220fe5b20a3d19 | 0 | 无 |
| S4 | 5a220fe5b20a3d19 | 0 | 无 |

#### 与 RUN-0024 的对照：第 4 批（第 192–255 条）的 raw_text

| 比较 | 不同的条数（/64） | 不同的 id |
|---|---|---|
| S1 第 1 遍 vs RUN-0024 A1 | 1 | refcoco+_unc/val/7909 |
| S1 第 1 遍 vs RUN-0024 A2 | 0 | 无 |
| S1 第 1 遍 vs RUN-0024 B | 41 | refcoco_unc/val/136108, refcoco_unc/val/137635, refcoco_unc/val/137938, refcoco_unc/val/139675, refcoco_unc/val/140960, refcoco+_unc/val/1162, refcoco+_unc/val/2207, refcoco+_unc/val/2718, refcoco+_unc/val/6472, refcoco+_unc/val/7186, refcoco+_unc/val/7543, refcoco+_unc/val/7909, refcoco+_unc/val/8452, refcoco+_unc/val/12233, refcoco+_unc/val/17563, refcoco+_unc/val/17675, refcoco+_unc/val/18811, refcoco+_unc/val/19023, refcoco+_unc/val/19471, refcoco+_unc/val/19848, refcoco+_unc/val/21625, refcoco+_unc/val/22732, refcoco+_unc/val/24209, refcoco+_unc/val/24441, refcoco+_unc/val/25737, refcoco+_unc/val/26021, refcoco+_unc/val/26884, refcoco+_unc/val/26949, refcoco+_unc/val/27165, refcoco+_unc/val/27984, refcoco+_unc/val/30576, refcoco+_unc/val/31296, refcoco+_unc/val/31298, refcoco+_unc/val/31688, refcoco+_unc/val/32538, refcoco+_unc/val/34159, refcoco+_unc/val/34919, refcoco+_unc/val/35493, refcoco+_unc/val/35767, refcoco+_unc/val/37793, refcoco+_unc/val/37798 |

`refcoco+_unc/val/7909` 的 raw_text：

| 来源 | raw_text |
|---|---|
| RUN-0024 A1 | ````json\n[\n\t{\"bbox_2d\": [506, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| RUN-0024 A2 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| RUN-0024 B（同 id） | ````json\n[\n\t{\"bbox_2d\": [508, 495, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S1 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S2a 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S2b 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S2c 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S3 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |
| 本次 S4 第 1 遍 | ````json\n[\n\t{\"bbox_2d\": [508, 496, 631, 988], \"label\": \"white jacket\"}\n]\n```<|im_end|>` |

#### 完整开发子集（run_e1.py --acceptance --task refexp_box，600 条；A1 与 A2，另有 B）

| 设置 | A1 vs A2 文件逐字节相同 | A1 vs A2 不同的条数 | A1 sha256（前 16 位） | A1 = RUN-0024 A1 | A1 = RUN-0024 A2 | 与 RUN-0024 A1 不同的条数 | 墙钟 s：A1 / A2 / B（compute.json） | all_pass |
|---|---|---|---|---|---|---|---|---|
| S1 | True | 0 | 320edc1d26647052 | False | True | 1 | 14.8 / 14.1 / 16.2 | True |
| S3 | True | 0 | 320edc1d26647052 | False | True | 1 | 16.2 / 15.9 / 18.1 | True |

#### 第 1 个不同位置的前两名 token 与 logit 差

本次所有设置里都没有出现与第 1 遍不同的条，没有可记录的分歧位置。

#### warn_only 列出的非确定性算子（torch.use_deterministic_algorithms(True, warn_only=True) 期间的 warnings，去重）

- S3（0 条与确定性相关；全部 warnings 去重后 8 条）：无
- S4（0 条与确定性相关；全部 warnings 去重后 8 条）：无
- full_S3（0 条与确定性相关；全部去重后 8 条；deterministic_algorithms=True，rc=0）：无
