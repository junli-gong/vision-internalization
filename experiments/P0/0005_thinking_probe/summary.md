# RUN-0005 summary（由 p9_summary.py 从 records.jsonl 生成）

records: 240 条；模型：internvl3.5-2b-hf, qwen3.5-0.8b, gemma-4-e2b-it, qwen3.5-2b, internvl3.5-4b-hf, gemma-4-e4b-it, qwen3.5-4b, qwen3.5-9b

## 表 1：模型 × 设置（每格应为 n=12）

| 模型 | 设置 | n | 开思考 | 闭合 | 写满预算 | 以停止 token 结束 | 思考 token 中位数（仅开思考） | 思考 token 最大（仅开思考） | 回答 token 中位数 | tokens/s 均值 | 墙钟合计 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| internvl3.5-2b-hf | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 9 | 58.54 | 14.0 |
| internvl3.5-2b-hf | think_greedy | 12 | 8 | 3 | 5 | 7 | 4095 | 4095 | 4.5 | 67.45 | 301.7 |
| internvl3.5-2b-hf | think_sampled | 12 | 9 | 5 | 4 | 8 | 757 | 4095 | 12 | 77.69 | 272.4 |
| qwen3.5-0.8b | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 25 | 62.57 | 12.2 |
| qwen3.5-0.8b | think_greedy | 12 | 12 | 10 | 2 | 10 | 226.5 | 4096 | 2 | 82.63 | 154.4 |
| qwen3.5-0.8b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||
| gemma-4-e2b-it | nothink_greedy | 12 | 0 | 0 | 1 | 11 | — | — | 21.5 | 43.41 | 17.2 |
| gemma-4-e2b-it | think_greedy | 12 | 11 | 11 | 0 | 12 | 303 | 1132 | 13.5 | 54.55 | 108.3 |
| gemma-4-e2b-it | think_sampled | 12 | 11 | 11 | 0 | 12 | 273 | 1070 | 13.5 | 52.99 | 103.1 |
| qwen3.5-2b | nothink_greedy | 12 | 0 | 0 | 1 | 11 | — | — | 22 | 60.36 | 18.2 |
| qwen3.5-2b | think_greedy | 12 | 12 | 9 | 3 | 9 | 311.5 | 4096 | 2 | 83.26 | 185.3 |
| qwen3.5-2b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||
| internvl3.5-4b-hf | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 1.5 | 45.84 | 16.0 |
| internvl3.5-4b-hf | think_greedy | 12 | 11 | 5 | 6 | 6 | 4095 | 4095 | 1 | 71.09 | 342.7 |
| internvl3.5-4b-hf | think_sampled | 12 | 11 | 7 | 5 | 7 | 1222 | 4095 | 2 | 70.22 | 418.7 |
| gemma-4-e4b-it | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 1 | 33.74 | 19.3 |
| gemma-4-e4b-it | think_greedy | 12 | 11 | 11 | 0 | 12 | 312 | 761 | 1.5 | 42.86 | 104.9 |
| gemma-4-e4b-it | think_sampled | 12 | 10 | 10 | 0 | 12 | 423.5 | 775 | 1.5 | 41 | 118.3 |
| qwen3.5-4b | nothink_greedy | 12 | 0 | 0 | 2 | 10 | — | — | 2.5 | 44.02 | 25.0 |
| qwen3.5-4b | think_greedy | 12 | 12 | 10 | 2 | 10 | 232.5 | 4096 | 2 | 60.46 | 196.7 |
| qwen3.5-4b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||
| qwen3.5-9b | nothink_greedy | 12 | 0 | 0 | 2 | 10 | — | — | 11.5 | 45.32 | 22.1 |
| qwen3.5-9b | think_greedy | 12 | 12 | 11 | 1 | 11 | 296.5 | 4096 | 2.5 | 61.16 | 211.4 |
| qwen3.5-9b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||

## 表 2：think_greedy 下没开思考的提示 id

| 模型 | 没开思考的提示 | 个数 |
|---|---|---|
| internvl3.5-2b-hf | A1, A5, B1, B3 | 4 |
| qwen3.5-0.8b | （无） | 0 |
| gemma-4-e2b-it | B4 | 1 |
| qwen3.5-2b | （无） | 0 |
| internvl3.5-4b-hf | B1 | 1 |
| gemma-4-e4b-it | A1 | 1 |
| qwen3.5-4b | （无） | 0 |
| qwen3.5-9b | （无） | 0 |

## 表 3：nothink_greedy 下输出里是否出现任何思考标记

| 模型 | 出现思考标记的提示（open/close/其他标记计数） | 个数 / n |
|---|---|---|
| internvl3.5-2b-hf | （无） | 0 / 12 |
| qwen3.5-0.8b | （无） | 0 / 12 |
| gemma-4-e2b-it | （无） | 0 / 12 |
| qwen3.5-2b | （无） | 0 / 12 |
| internvl3.5-4b-hf | （无） | 0 / 12 |
| gemma-4-e4b-it | （无） | 0 / 12 |
| qwen3.5-4b | （无） | 0 / 12 |
| qwen3.5-9b | （无） | 0 / 12 |

## 附：每条记录一行（模型、提示、设置、生成 token、停止、开/闭、思考/回答 token、墙钟）

| 模型 | 提示 | 设置 | 生成 | 停止 | 满预算 | 开 | 闭 | 思考 | 回答 | unclosed | 墙钟 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| internvl3.5-2b-hf | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.653 |
| internvl3.5-2b-hf | A1 | think_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.092 |
| internvl3.5-2b-hf | A1 | think_sampled | 2 | True | False | False | False | 0 | 1 | False | 0.076 |
| internvl3.5-2b-hf | A2 | nothink_greedy | 10 | True | False | False | False | 0 | 9 | False | 0.31 |
| internvl3.5-2b-hf | A2 | think_greedy | 198 | True | False | True | True | 179 | 16 | False | 6.938 |
| internvl3.5-2b-hf | A2 | think_sampled | 159 | True | False | True | True | 140 | 16 | False | 1.751 |
| internvl3.5-2b-hf | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.059 |
| internvl3.5-2b-hf | A3 | think_greedy | 415 | True | False | True | True | 390 | 22 | False | 9.602 |
| internvl3.5-2b-hf | A3 | think_sampled | 646 | True | False | True | True | 607 | 36 | False | 12.78 |
| internvl3.5-2b-hf | A4 | nothink_greedy | 316 | True | False | False | False | 0 | 315 | False | 6.064 |
| internvl3.5-2b-hf | A4 | think_greedy | 1969 | True | False | True | True | 1881 | 85 | False | 55.038 |
| internvl3.5-2b-hf | A4 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 96.667 |
| internvl3.5-2b-hf | A5 | nothink_greedy | 10 | True | False | False | False | 0 | 9 | False | 0.148 |
| internvl3.5-2b-hf | A5 | think_greedy | 10 | True | False | False | False | 0 | 9 | False | 0.148 |
| internvl3.5-2b-hf | A5 | think_sampled | 89 | True | False | False | False | 0 | 88 | False | 0.994 |
| internvl3.5-2b-hf | A6 | nothink_greedy | 27 | True | False | False | False | 0 | 26 | False | 0.328 |
| internvl3.5-2b-hf | A6 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 45.064 |
| internvl3.5-2b-hf | A6 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 45.507 |
| internvl3.5-2b-hf | B1 | nothink_greedy | 342 | True | False | False | False | 0 | 341 | False | 3.902 |
| internvl3.5-2b-hf | B1 | think_greedy | 323 | True | False | False | False | 0 | 322 | False | 3.504 |
| internvl3.5-2b-hf | B1 | think_sampled | 727 | True | False | True | True | 266 | 458 | False | 7.994 |
| internvl3.5-2b-hf | B2 | nothink_greedy | 10 | True | False | False | False | 0 | 9 | False | 0.118 |
| internvl3.5-2b-hf | B2 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 44.648 |
| internvl3.5-2b-hf | B2 | think_sampled | 793 | True | False | True | True | 757 | 33 | False | 8.706 |
| internvl3.5-2b-hf | B3 | nothink_greedy | 3 | True | False | False | False | 0 | 2 | False | 0.073 |
| internvl3.5-2b-hf | B3 | think_greedy | 9 | True | False | False | False | 0 | 8 | False | 0.134 |
| internvl3.5-2b-hf | B3 | think_sampled | 9 | True | False | False | False | 0 | 8 | False | 0.108 |
| internvl3.5-2b-hf | B4 | nothink_greedy | 60 | True | False | False | False | 0 | 59 | False | 0.681 |
| internvl3.5-2b-hf | B4 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 44.496 |
| internvl3.5-2b-hf | B4 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 45.487 |
| internvl3.5-2b-hf | B5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.061 |
| internvl3.5-2b-hf | B5 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 43.183 |
| internvl3.5-2b-hf | B5 | think_sampled | 658 | True | False | True | True | 614 | 41 | False | 6.756 |
| internvl3.5-2b-hf | B6 | nothink_greedy | 155 | True | False | False | False | 0 | 154 | False | 1.597 |
| internvl3.5-2b-hf | B6 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 48.842 |
| internvl3.5-2b-hf | B6 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 45.589 |
| qwen3.5-0.8b | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.682 |
| qwen3.5-0.8b | A1 | think_greedy | 144 | True | False | True | True | 140 | 2 | False | 1.756 |
| qwen3.5-0.8b | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.041 |
| qwen3.5-0.8b | A2 | think_greedy | 165 | True | False | True | True | 161 | 2 | False | 2.017 |
| qwen3.5-0.8b | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.045 |
| qwen3.5-0.8b | A3 | think_greedy | 115 | True | False | True | True | 111 | 2 | False | 1.436 |
| qwen3.5-0.8b | A4 | nothink_greedy | 45 | True | False | False | False | 0 | 44 | False | 0.568 |
| qwen3.5-0.8b | A4 | think_greedy | 103 | True | False | True | True | 81 | 20 | False | 1.256 |
| qwen3.5-0.8b | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.047 |
| qwen3.5-0.8b | A5 | think_greedy | 224 | True | False | True | True | 220 | 2 | False | 2.789 |
| qwen3.5-0.8b | A6 | nothink_greedy | 159 | True | False | False | False | 0 | 158 | False | 1.917 |
| qwen3.5-0.8b | A6 | think_greedy | 2023 | True | False | True | True | 1916 | 105 | False | 24.367 |
| qwen3.5-0.8b | B1 | nothink_greedy | 432 | True | False | False | False | 0 | 431 | False | 5.178 |
| qwen3.5-0.8b | B1 | think_greedy | 1330 | True | False | True | True | 912 | 416 | False | 15.836 |
| qwen3.5-0.8b | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.042 |
| qwen3.5-0.8b | B2 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 48.763 |
| qwen3.5-0.8b | B3 | nothink_greedy | 9 | True | False | False | False | 0 | 8 | False | 0.126 |
| qwen3.5-0.8b | B3 | think_greedy | 238 | True | False | True | True | 233 | 3 | False | 2.844 |
| qwen3.5-0.8b | B4 | nothink_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.53 |
| qwen3.5-0.8b | B4 | think_greedy | 93 | True | False | True | True | 74 | 17 | False | 1.127 |
| qwen3.5-0.8b | B5 | nothink_greedy | 76 | True | False | False | False | 0 | 75 | False | 0.924 |
| qwen3.5-0.8b | B5 | think_greedy | 301 | True | False | True | True | 297 | 2 | False | 3.585 |
| qwen3.5-0.8b | B6 | nothink_greedy | 172 | True | False | False | False | 0 | 171 | False | 2.06 |
| qwen3.5-0.8b | B6 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 48.656 |
| gemma-4-e2b-it | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.645 |
| gemma-4-e2b-it | A1 | think_greedy | 202 | True | False | True | True | 198 | 1 | False | 3.804 |
| gemma-4-e2b-it | A1 | think_sampled | 210 | True | False | True | True | 206 | 1 | False | 3.985 |
| gemma-4-e2b-it | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.057 |
| gemma-4-e2b-it | A2 | think_greedy | 246 | True | False | True | True | 242 | 1 | False | 4.503 |
| gemma-4-e2b-it | A2 | think_sampled | 235 | True | False | True | True | 231 | 1 | False | 4.423 |
| gemma-4-e2b-it | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.058 |
| gemma-4-e2b-it | A3 | think_greedy | 267 | True | False | True | True | 263 | 1 | False | 4.901 |
| gemma-4-e2b-it | A3 | think_sampled | 273 | True | False | True | True | 269 | 1 | False | 5.153 |
| gemma-4-e2b-it | A4 | nothink_greedy | 46 | True | False | False | False | 0 | 45 | False | 0.858 |
| gemma-4-e2b-it | A4 | think_greedy | 143 | True | False | True | True | 95 | 45 | False | 2.613 |
| gemma-4-e2b-it | A4 | think_sampled | 77 | True | False | True | True | 29 | 45 | False | 1.461 |
| gemma-4-e2b-it | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.058 |
| gemma-4-e2b-it | A5 | think_greedy | 307 | True | False | True | True | 303 | 1 | False | 5.676 |
| gemma-4-e2b-it | A5 | think_sampled | 277 | True | False | True | True | 273 | 1 | False | 5.266 |
| gemma-4-e2b-it | A6 | nothink_greedy | 130 | True | False | False | False | 0 | 129 | False | 2.375 |
| gemma-4-e2b-it | A6 | think_greedy | 997 | True | False | True | True | 914 | 80 | False | 18.256 |
| gemma-4-e2b-it | A6 | think_sampled | 713 | True | False | True | True | 633 | 77 | False | 13.721 |
| gemma-4-e2b-it | B1 | nothink_greedy | 512 | False | True | False | False | 0 | 512 | False | 9.525 |
| gemma-4-e2b-it | B1 | think_greedy | 1205 | True | False | True | True | 698 | 504 | False | 22.382 |
| gemma-4-e2b-it | B1 | think_sampled | 1146 | True | False | True | True | 692 | 451 | False | 22.142 |
| gemma-4-e2b-it | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.058 |
| gemma-4-e2b-it | B2 | think_greedy | 598 | True | False | True | True | 594 | 1 | False | 11.186 |
| gemma-4-e2b-it | B2 | think_sampled | 663 | True | False | True | True | 659 | 1 | False | 12.573 |
| gemma-4-e2b-it | B3 | nothink_greedy | 9 | True | False | False | False | 0 | 8 | False | 0.179 |
| gemma-4-e2b-it | B3 | think_greedy | 177 | True | False | True | True | 173 | 1 | False | 3.128 |
| gemma-4-e2b-it | B3 | think_sampled | 172 | True | False | True | True | 168 | 1 | False | 3.13 |
| gemma-4-e2b-it | B4 | nothink_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.779 |
| gemma-4-e2b-it | B4 | think_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.778 |
| gemma-4-e2b-it | B4 | think_sampled | 43 | True | False | False | False | 0 | 42 | False | 0.802 |
| gemma-4-e2b-it | B5 | nothink_greedy | 36 | True | False | False | False | 0 | 35 | False | 0.655 |
| gemma-4-e2b-it | B5 | think_greedy | 1161 | True | False | True | True | 1132 | 26 | False | 21.284 |
| gemma-4-e2b-it | B5 | think_sampled | 1099 | True | False | True | True | 1070 | 26 | False | 20.606 |
| gemma-4-e2b-it | B6 | nothink_greedy | 108 | True | False | False | False | 0 | 107 | False | 1.925 |
| gemma-4-e2b-it | B6 | think_greedy | 544 | True | False | True | True | 456 | 85 | False | 9.837 |
| gemma-4-e2b-it | B6 | think_sampled | 524 | True | False | True | True | 429 | 92 | False | 9.802 |
| qwen3.5-2b | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.664 |
| qwen3.5-2b | A1 | think_greedy | 152 | True | False | True | True | 148 | 2 | False | 1.83 |
| qwen3.5-2b | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.044 |
| qwen3.5-2b | A2 | think_greedy | 218 | True | False | True | True | 214 | 2 | False | 2.616 |
| qwen3.5-2b | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.045 |
| qwen3.5-2b | A3 | think_greedy | 116 | True | False | True | True | 112 | 2 | False | 1.404 |
| qwen3.5-2b | A4 | nothink_greedy | 45 | True | False | False | False | 0 | 44 | False | 0.56 |
| qwen3.5-2b | A4 | think_greedy | 134 | True | False | True | True | 86 | 46 | False | 1.618 |
| qwen3.5-2b | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.047 |
| qwen3.5-2b | A5 | think_greedy | 409 | True | False | True | True | 405 | 2 | False | 4.9 |
| qwen3.5-2b | A6 | nothink_greedy | 512 | False | True | False | False | 0 | 512 | False | 6.13 |
| qwen3.5-2b | A6 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 48.936 |
| qwen3.5-2b | B1 | nothink_greedy | 474 | True | False | False | False | 0 | 473 | False | 5.693 |
| qwen3.5-2b | B1 | think_greedy | 1437 | True | False | True | True | 886 | 549 | False | 17.214 |
| qwen3.5-2b | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.045 |
| qwen3.5-2b | B2 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 48.851 |
| qwen3.5-2b | B3 | nothink_greedy | 3 | True | False | False | False | 0 | 2 | False | 0.058 |
| qwen3.5-2b | B3 | think_greedy | 268 | True | False | True | True | 257 | 9 | False | 3.223 |
| qwen3.5-2b | B4 | nothink_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.536 |
| qwen3.5-2b | B4 | think_greedy | 119 | True | False | True | True | 75 | 42 | False | 1.447 |
| qwen3.5-2b | B5 | nothink_greedy | 105 | True | False | False | False | 0 | 104 | False | 1.272 |
| qwen3.5-2b | B5 | think_greedy | 370 | True | False | True | True | 366 | 2 | False | 4.428 |
| qwen3.5-2b | B6 | nothink_greedy | 259 | True | False | False | False | 0 | 258 | False | 3.108 |
| qwen3.5-2b | B6 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 48.851 |
| internvl3.5-4b-hf | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.633 |
| internvl3.5-4b-hf | A1 | think_greedy | 99 | True | False | True | True | 84 | 12 | False | 3.49 |
| internvl3.5-4b-hf | A1 | think_sampled | 4096 | False | True | True | True | 303 | 3791 | False | 142.864 |
| internvl3.5-4b-hf | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.035 |
| internvl3.5-4b-hf | A2 | think_greedy | 102 | True | False | True | True | 97 | 2 | False | 1.281 |
| internvl3.5-4b-hf | A2 | think_sampled | 156 | True | False | True | True | 151 | 2 | False | 1.992 |
| internvl3.5-4b-hf | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.085 |
| internvl3.5-4b-hf | A3 | think_greedy | 61 | True | False | True | True | 56 | 2 | False | 0.794 |
| internvl3.5-4b-hf | A3 | think_sampled | 66 | True | False | True | True | 61 | 2 | False | 0.85 |
| internvl3.5-4b-hf | A4 | nothink_greedy | 147 | True | False | False | False | 0 | 146 | False | 4.703 |
| internvl3.5-4b-hf | A4 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 51.858 |
| internvl3.5-4b-hf | A4 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 56.33 |
| internvl3.5-4b-hf | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.072 |
| internvl3.5-4b-hf | A5 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 55.53 |
| internvl3.5-4b-hf | A5 | think_sampled | 1241 | True | False | True | True | 1222 | 16 | False | 17.075 |
| internvl3.5-4b-hf | A6 | nothink_greedy | 23 | True | False | False | False | 0 | 22 | False | 0.352 |
| internvl3.5-4b-hf | A6 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 56.35 |
| internvl3.5-4b-hf | A6 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 55.817 |
| internvl3.5-4b-hf | B1 | nothink_greedy | 360 | True | False | False | False | 0 | 359 | False | 5.008 |
| internvl3.5-4b-hf | B1 | think_greedy | 290 | True | False | False | False | 0 | 289 | False | 3.878 |
| internvl3.5-4b-hf | B1 | think_sampled | 315 | True | False | False | False | 0 | 314 | False | 4.258 |
| internvl3.5-4b-hf | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.036 |
| internvl3.5-4b-hf | B2 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 54.969 |
| internvl3.5-4b-hf | B2 | think_sampled | 1636 | True | False | True | True | 1631 | 2 | False | 22.28 |
| internvl3.5-4b-hf | B3 | nothink_greedy | 3 | True | False | False | False | 0 | 2 | False | 0.08 |
| internvl3.5-4b-hf | B3 | think_greedy | 124 | True | False | True | True | 101 | 20 | False | 1.692 |
| internvl3.5-4b-hf | B3 | think_sampled | 102 | True | False | True | True | 79 | 20 | False | 1.362 |
| internvl3.5-4b-hf | B4 | nothink_greedy | 204 | True | False | False | False | 0 | 203 | False | 2.751 |
| internvl3.5-4b-hf | B4 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 55.252 |
| internvl3.5-4b-hf | B4 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 56.764 |
| internvl3.5-4b-hf | B5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.066 |
| internvl3.5-4b-hf | B5 | think_greedy | 132 | True | False | True | True | 87 | 42 | False | 1.819 |
| internvl3.5-4b-hf | B5 | think_sampled | 172 | True | False | True | True | 119 | 50 | False | 2.366 |
| internvl3.5-4b-hf | B6 | nothink_greedy | 159 | True | False | False | False | 0 | 158 | False | 2.182 |
| internvl3.5-4b-hf | B6 | think_greedy | 4096 | False | True | True | False | 4095 | 0 | True | 55.753 |
| internvl3.5-4b-hf | B6 | think_sampled | 4096 | False | True | True | False | 4095 | 0 | True | 56.761 |
| gemma-4-e4b-it | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.629 |
| gemma-4-e4b-it | A1 | think_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.07 |
| gemma-4-e4b-it | A1 | think_sampled | 2 | True | False | False | False | 0 | 1 | False | 0.099 |
| gemma-4-e4b-it | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.065 |
| gemma-4-e4b-it | A2 | think_greedy | 197 | True | False | True | True | 193 | 1 | False | 4.454 |
| gemma-4-e4b-it | A2 | think_sampled | 171 | True | False | True | True | 167 | 1 | False | 3.965 |
| gemma-4-e4b-it | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.068 |
| gemma-4-e4b-it | A3 | think_greedy | 218 | True | False | True | True | 214 | 1 | False | 4.947 |
| gemma-4-e4b-it | A3 | think_sampled | 245 | True | False | True | True | 241 | 1 | False | 5.709 |
| gemma-4-e4b-it | A4 | nothink_greedy | 46 | True | False | False | False | 0 | 45 | False | 1.06 |
| gemma-4-e4b-it | A4 | think_greedy | 90 | True | False | True | True | 42 | 45 | False | 2.045 |
| gemma-4-e4b-it | A4 | think_sampled | 407 | True | False | True | True | 359 | 45 | False | 9.643 |
| gemma-4-e4b-it | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.067 |
| gemma-4-e4b-it | A5 | think_greedy | 333 | True | False | True | True | 329 | 1 | False | 7.441 |
| gemma-4-e4b-it | A5 | think_sampled | 659 | True | False | True | True | 655 | 1 | False | 15.764 |
| gemma-4-e4b-it | A6 | nothink_greedy | 195 | True | False | False | False | 0 | 194 | False | 4.377 |
| gemma-4-e4b-it | A6 | think_greedy | 850 | True | False | True | True | 761 | 86 | False | 19.423 |
| gemma-4-e4b-it | A6 | think_sampled | 637 | True | False | True | True | 555 | 79 | False | 14.921 |
| gemma-4-e4b-it | B1 | nothink_greedy | 448 | True | False | False | False | 0 | 447 | False | 10.379 |
| gemma-4-e4b-it | B1 | think_greedy | 1214 | True | False | True | True | 760 | 451 | False | 28.02 |
| gemma-4-e4b-it | B1 | think_sampled | 1231 | True | False | True | True | 775 | 453 | False | 28.781 |
| gemma-4-e4b-it | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.066 |
| gemma-4-e4b-it | B2 | think_greedy | 515 | True | False | True | True | 511 | 1 | False | 11.658 |
| gemma-4-e4b-it | B2 | think_sampled | 331 | True | False | True | True | 327 | 1 | False | 7.647 |
| gemma-4-e4b-it | B3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.065 |
| gemma-4-e4b-it | B3 | think_greedy | 214 | True | False | True | True | 209 | 2 | False | 4.844 |
| gemma-4-e4b-it | B3 | think_sampled | 186 | True | False | True | True | 181 | 2 | False | 4.279 |
| gemma-4-e4b-it | B4 | nothink_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.966 |
| gemma-4-e4b-it | B4 | think_greedy | 58 | True | False | True | True | 36 | 19 | False | 1.295 |
| gemma-4-e4b-it | B4 | think_sampled | 43 | True | False | False | False | 0 | 42 | False | 0.986 |
| gemma-4-e4b-it | B5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.066 |
| gemma-4-e4b-it | B5 | think_greedy | 316 | True | False | True | True | 312 | 1 | False | 7.071 |
| gemma-4-e4b-it | B5 | think_sampled | 492 | True | False | True | True | 488 | 1 | False | 11.421 |
| gemma-4-e4b-it | B6 | nothink_greedy | 66 | True | False | False | False | 0 | 65 | False | 1.467 |
| gemma-4-e4b-it | B6 | think_greedy | 597 | True | False | True | True | 531 | 63 | False | 13.659 |
| gemma-4-e4b-it | B6 | think_sampled | 647 | True | False | True | True | 557 | 87 | False | 15.077 |
| qwen3.5-4b | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.663 |
| qwen3.5-4b | A1 | think_greedy | 147 | True | False | True | True | 143 | 2 | False | 2.436 |
| qwen3.5-4b | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.057 |
| qwen3.5-4b | A2 | think_greedy | 207 | True | False | True | True | 203 | 2 | False | 3.526 |
| qwen3.5-4b | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.058 |
| qwen3.5-4b | A3 | think_greedy | 170 | True | False | True | True | 166 | 2 | False | 2.869 |
| qwen3.5-4b | A4 | nothink_greedy | 45 | True | False | False | False | 0 | 44 | False | 0.758 |
| qwen3.5-4b | A4 | think_greedy | 115 | True | False | True | True | 93 | 20 | False | 1.898 |
| qwen3.5-4b | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.061 |
| qwen3.5-4b | A5 | think_greedy | 652 | True | False | True | True | 648 | 2 | False | 10.988 |
| qwen3.5-4b | A6 | nothink_greedy | 340 | True | False | False | False | 0 | 339 | False | 5.776 |
| qwen3.5-4b | A6 | think_greedy | 519 | True | False | True | True | 492 | 25 | False | 8.534 |
| qwen3.5-4b | B1 | nothink_greedy | 512 | False | True | False | False | 0 | 512 | False | 8.323 |
| qwen3.5-4b | B1 | think_greedy | 1381 | True | False | True | True | 933 | 446 | False | 22.46 |
| qwen3.5-4b | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.056 |
| qwen3.5-4b | B2 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 67.397 |
| qwen3.5-4b | B3 | nothink_greedy | 3 | True | False | False | False | 0 | 2 | False | 0.073 |
| qwen3.5-4b | B3 | think_greedy | 200 | True | False | True | True | 195 | 3 | False | 3.275 |
| qwen3.5-4b | B4 | nothink_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.725 |
| qwen3.5-4b | B4 | think_greedy | 130 | True | False | True | True | 85 | 43 | False | 2.141 |
| qwen3.5-4b | B5 | nothink_greedy | 4 | True | False | False | False | 0 | 3 | False | 0.088 |
| qwen3.5-4b | B5 | think_greedy | 266 | True | False | True | True | 262 | 2 | False | 4.355 |
| qwen3.5-4b | B6 | nothink_greedy | 512 | False | True | False | False | 0 | 512 | False | 8.345 |
| qwen3.5-4b | B6 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 66.794 |
| qwen3.5-9b | A1 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 1.134 |
| qwen3.5-9b | A1 | think_greedy | 335 | True | False | True | True | 331 | 2 | False | 5.536 |
| qwen3.5-9b | A2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.056 |
| qwen3.5-9b | A2 | think_greedy | 174 | True | False | True | True | 170 | 2 | False | 2.868 |
| qwen3.5-9b | A3 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.057 |
| qwen3.5-9b | A3 | think_greedy | 145 | True | False | True | True | 141 | 2 | False | 2.393 |
| qwen3.5-9b | A4 | nothink_greedy | 45 | True | False | False | False | 0 | 44 | False | 0.756 |
| qwen3.5-9b | A4 | think_greedy | 139 | True | False | True | True | 91 | 46 | False | 2.269 |
| qwen3.5-9b | A5 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.06 |
| qwen3.5-9b | A5 | think_greedy | 1820 | True | False | True | True | 1816 | 2 | False | 29.579 |
| qwen3.5-9b | A6 | nothink_greedy | 133 | True | False | False | False | 0 | 132 | False | 2.18 |
| qwen3.5-9b | A6 | think_greedy | 460 | True | False | True | True | 414 | 44 | False | 7.483 |
| qwen3.5-9b | B1 | nothink_greedy | 512 | False | True | False | False | 0 | 512 | False | 8.339 |
| qwen3.5-9b | B1 | think_greedy | 1453 | True | False | True | True | 1052 | 399 | False | 23.63 |
| qwen3.5-9b | B2 | nothink_greedy | 2 | True | False | False | False | 0 | 1 | False | 0.056 |
| qwen3.5-9b | B2 | think_greedy | 4096 | False | True | True | False | 4096 | 0 | True | 66.79 |
| qwen3.5-9b | B3 | nothink_greedy | 3 | True | False | False | False | 0 | 2 | False | 0.073 |
| qwen3.5-9b | B3 | think_greedy | 227 | True | False | True | True | 222 | 3 | False | 3.705 |
| qwen3.5-9b | B4 | nothink_greedy | 43 | True | False | False | False | 0 | 42 | False | 0.726 |
| qwen3.5-9b | B4 | think_greedy | 142 | True | False | True | True | 97 | 43 | False | 2.338 |
| qwen3.5-9b | B5 | nothink_greedy | 22 | True | False | False | False | 0 | 21 | False | 0.38 |
| qwen3.5-9b | B5 | think_greedy | 266 | True | False | True | True | 262 | 2 | False | 4.331 |
| qwen3.5-9b | B6 | nothink_greedy | 512 | False | True | False | False | 0 | 512 | False | 8.3 |
| qwen3.5-9b | B6 | think_greedy | 3727 | True | False | True | True | 3602 | 123 | False | 60.434 |
