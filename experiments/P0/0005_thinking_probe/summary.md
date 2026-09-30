# RUN-0005 summary（由 p9_summary.py 从 records.jsonl 生成）

records: 96 条；模型：internvl3.5-2b-hf, qwen3.5-0.8b, gemma-4-e2b-it

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

## 表 2：think_greedy 下没开思考的提示 id

| 模型 | 没开思考的提示 | 个数 |
|---|---|---|
| internvl3.5-2b-hf | A1, A5, B1, B3 | 4 |
| qwen3.5-0.8b | （无） | 0 |
| gemma-4-e2b-it | B4 | 1 |

## 表 3：nothink_greedy 下输出里是否出现任何思考标记

| 模型 | 出现思考标记的提示（open/close/其他标记计数） | 个数 / n |
|---|---|---|
| internvl3.5-2b-hf | （无） | 0 / 12 |
| qwen3.5-0.8b | （无） | 0 / 12 |
| gemma-4-e2b-it | （无） | 0 / 12 |

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
