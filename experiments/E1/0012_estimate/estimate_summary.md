#### 估算汇总（主口径；推荐批大小 = 可行的批大小里样本/秒最高者；GPU 小时 = n / 样本每秒，由 run_e1 外推）

| 模型 | 任务 | 推荐批大小 | 样本/秒 | n(eval) | GPU h(eval) | 思考实测批大小 | 思考样本/秒 | n(think) | GPU h(think) | n(8192) | GPU h(8192, 上界) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 32 | 20.44 | 56857 | 0.773 | 8 | 4.52 | 200 | 0.012 | 50 | 0.006 |
| qwen3.5-0.8b | refexp_poly | 32 | 12.79 | 8000 | 0.174 | 8 | 0.14 | 200 | 0.383 | 50 | 0.192 |
| qwen3.5-0.8b | grid16 | 32 | 2.47 | 500 | 0.056 | 8 | 0.15 | 200 | 0.359 | 50 | 0.179 |
| qwen3.5-0.8b | count | 16 | 91.80 | 442 | 0.001 | 8 | 0.15 | 200 | 0.360 | 50 | 0.180 |
| qwen3.5-0.8b | ovd | 32 | 4.88 | 963 | 0.055 | 8 | 0.16 | 200 | 0.358 | 50 | 0.179 |
| qwen3.5-0.8b | rel_depth_nyu | 32 | 36.77 | 3270 | 0.025 | 8 | 0.16 | 200 | 0.340 | 50 | 0.170 |
| qwen3.5-0.8b | rel_depth_blink | 32 | 14.35 | 112 | 0.002 | 8 | 0.15 | 112 | 0.201 | 50 | 0.180 |
| qwen3.5-0.8b | spatial_cvbench | 32 | 21.56 | 2588 | 0.033 | 8 | 0.15 | 200 | 0.359 | 50 | 0.179 |
| qwen3.5-0.8b | spatial_embspatial | 32 | 28.25 | 3590 | 0.035 | 8 | 0.15 | 200 | 0.358 | 50 | 0.179 |
| qwen3.5-0.8b | **合计** | | | | **1.154** | | | | **2.731** | | **1.445** |
| qwen3.5-2b | refexp_box | 32 | 23.76 | 56857 | 0.665 | 8 | 3.79 | 200 | 0.015 | 50 | 0.007 |
| qwen3.5-2b | refexp_poly | 32 | 15.53 | 8000 | 0.143 | 8 | 0.15 | 200 | 0.374 | 50 | 0.187 |
| qwen3.5-2b | grid16 | 32 | 2.20 | 500 | 0.063 | 8 | 0.14 | 200 | 0.399 | 50 | 0.200 |
| qwen3.5-2b | count | 32 | 141.66 | 442 | 0.001 | 8 | 0.21 | 200 | 0.262 | 50 | 0.131 |
| qwen3.5-2b | ovd | 32 | 4.34 | 963 | 0.062 | 8 | 0.21 | 200 | 0.262 | 50 | 0.131 |
| qwen3.5-2b | rel_depth_nyu | 32 | 32.11 | 3270 | 0.028 | 8 | 0.16 | 200 | 0.357 | 50 | 0.178 |
| qwen3.5-2b | rel_depth_blink | 32 | 13.29 | 112 | 0.002 | 8 | 0.15 | 112 | 0.210 | 50 | 0.187 |
| qwen3.5-2b | spatial_cvbench | 32 | 25.55 | 2588 | 0.028 | 8 | 0.15 | 200 | 0.375 | 50 | 0.188 |
| qwen3.5-2b | spatial_embspatial | 32 | 25.54 | 3590 | 0.039 | 8 | 0.15 | 200 | 0.374 | 50 | 0.187 |
| qwen3.5-2b | **合计** | | | | **1.031** | | | | **2.628** | | **1.396** |
| gemma-4-e2b-it | refexp_box | 32 | 10.72 | 56857 | 1.474 | 8 | 0.52 | 200 | 0.107 | 50 | 0.054 |
| gemma-4-e2b-it | refexp_poly | 32 | 4.41 | 8000 | 0.504 | 8 | 1.84 | 200 | 0.030 | 50 | 0.015 |
| gemma-4-e2b-it | grid16 | 32 | 5.61 | 500 | 0.025 | 8 | 0.42 | 200 | 0.131 | 50 | 0.066 |
| gemma-4-e2b-it | count | 32 | 78.52 | 442 | 0.002 | 8 | 0.70 | 200 | 0.080 | 50 | 0.040 |
| gemma-4-e2b-it | ovd | 32 | 4.32 | 963 | 0.062 | 8 | 0.37 | 200 | 0.151 | 50 | 0.075 |
| gemma-4-e2b-it | rel_depth_nyu | 32 | 21.49 | 3270 | 0.042 | 8 | 0.43 | 200 | 0.129 | 50 | 0.064 |
| gemma-4-e2b-it | rel_depth_blink | 32 | 8.29 | 112 | 0.004 | 8 | 0.30 | 112 | 0.102 | 50 | 0.091 |
| gemma-4-e2b-it | spatial_cvbench | 32 | 16.14 | 2588 | 0.045 | 8 | 0.37 | 200 | 0.150 | 50 | 0.075 |
| gemma-4-e2b-it | spatial_embspatial | 32 | 16.51 | 3590 | 0.060 | 8 | 0.20 | 200 | 0.271 | 50 | 0.136 |
| gemma-4-e2b-it | **合计** | | | | **2.217** | | | | **1.152** | | **0.616** |
| internvl3.5-2b-hf | refexp_box | 32 | 30.51 | 56857 | 0.518 | 8 | 0.05 | 200 | 1.100 | 50 | 0.550 |
| internvl3.5-2b-hf | refexp_poly | 32 | 1.07 | 8000 | 2.068 | 8 | 0.13 | 200 | 0.425 | 50 | 0.213 |
| internvl3.5-2b-hf | grid16 | 32 | 8.98 | 500 | 0.015 | 8 | 0.13 | 200 | 0.418 | 50 | 0.209 |
| internvl3.5-2b-hf | count | 16 | 16.87 | 442 | 0.007 | 8 | 0.14 | 200 | 0.408 | 50 | 0.204 |
| internvl3.5-2b-hf | ovd | 32 | 2.83 | 963 | 0.094 | 8 | 0.06 | 200 | 0.880 | 50 | 0.440 |
| internvl3.5-2b-hf | rel_depth_nyu | 32 | 21.65 | 3270 | 0.042 | 8 | 0.19 | 200 | 0.296 | 50 | 0.148 |
| internvl3.5-2b-hf | rel_depth_blink | 32 | 17.14 | 112 | 0.002 | 8 | 0.19 | 112 | 0.166 | 50 | 0.148 |
| internvl3.5-2b-hf | spatial_cvbench | 16 | 16.25 | 2588 | 0.044 | 8 | 0.14 | 200 | 0.409 | 50 | 0.204 |
| internvl3.5-2b-hf | spatial_embspatial | 32 | 22.46 | 3590 | 0.044 | 8 | 0.14 | 200 | 0.411 | 50 | 0.205 |
| internvl3.5-2b-hf | **合计** | | | | **2.835** | | | | **4.514** | | **2.322** |

#### 每个模型的合计（GPU 小时，主口径）

| 模型 | eval | think | think8192（上界） | eval+think+think8192 |
|---|---|---|---|---|
| qwen3.5-0.8b | 1.154 | 2.731 | 1.445 | 5.330 |
| qwen3.5-2b | 1.031 | 2.628 | 1.396 | 5.056 |
| gemma-4-e2b-it | 2.217 | 1.152 | 0.616 | 3.986 |
| internvl3.5-2b-hf | 2.835 | 4.514 | 2.322 | 9.671 |
| 已测 4 个模型之和 | 7.238 | 11.025 | 5.779 | 24.042 |

未测的模型：qwen3.5-4b, gemma-4-e4b-it, internvl3.5-4b-hf, qwen3.5-9b

#### 附加口径（只跑 refexp_box、ovd；只外推 extra 子集）

| 模型 | 口径 | 任务 | 推荐批大小 | 样本/秒 | n(extra) | GPU h(extra) | 主口径同任务样本/秒 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | qwen:1024 | refexp_box | 32 | 20.84 | 500 | 0.007 | 20.44 |
| qwen3.5-0.8b | qwen:1024 | ovd | 32 | 3.23 | 100 | 0.009 | 4.88 |
| qwen3.5-0.8b | qwen:1024 | **合计** | | | | **0.015** | |
| internvl3.5-2b-hf | internvl:native | refexp_box | 32 | 10.18 | 500 | 0.014 | 30.51 |
| internvl3.5-2b-hf | internvl:native | ovd | 32 | 1.06 | 100 | 0.026 | 2.83 |
| internvl3.5-2b-hf | internvl:native | **合计** | | | | **0.040** | |

#### 逐批大小明细（关思考；峰值显存 = torch.cuda.max_memory_allocated）

| 模型 | 口径 | 任务 | 批大小 | ok | 样本/秒 | token/秒 | 平均生成 token | 截断率 | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | default | refexp_box | 1 | True | 1.75 | 81.2 | 46.4 | 0.016 | 1.80 |
| qwen3.5-0.8b | default | refexp_box | 8 | True | 10.12 | 469.9 | 46.4 | 0.016 | 2.23 |
| qwen3.5-0.8b | default | refexp_box | 16 | True | 16.43 | 762.0 | 46.4 | 0.016 | 2.73 |
| qwen3.5-0.8b | default | refexp_box | 32 | True | 20.44 | 948.4 | 46.4 | 0.016 | 3.71 |
| qwen3.5-0.8b | default | refexp_poly | 1 | True | 1.55 | 82.0 | 53.1 | 0.000 | 1.81 |
| qwen3.5-0.8b | default | refexp_poly | 8 | True | 6.66 | 353.8 | 53.1 | 0.000 | 2.28 |
| qwen3.5-0.8b | default | refexp_poly | 16 | True | 8.38 | 460.0 | 54.9 | 0.000 | 2.80 |
| qwen3.5-0.8b | default | refexp_poly | 32 | True | 12.79 | 694.5 | 54.3 | 0.000 | 3.87 |
| qwen3.5-0.8b | default | grid16 | 1 | True | 0.08 | 86.8 | 1024.0 | 1.000 | 1.81 |
| qwen3.5-0.8b | default | grid16 | 8 | True | 0.63 | 641.2 | 1024.0 | 1.000 | 2.28 |
| qwen3.5-0.8b | default | grid16 | 16 | True | 1.25 | 1275.8 | 1024.0 | 1.000 | 2.82 |
| qwen3.5-0.8b | default | grid16 | 32 | True | 2.47 | 2525.1 | 1024.0 | 1.000 | 3.89 |
| qwen3.5-0.8b | default | count | 1 | True | 20.73 | 47.0 | 2.3 | 0.000 | 1.80 |
| qwen3.5-0.8b | default | count | 8 | True | 76.97 | 201.1 | 2.6 | 0.000 | 2.23 |
| qwen3.5-0.8b | default | count | 16 | True | 91.80 | 239.8 | 2.6 | 0.000 | 2.71 |
| qwen3.5-0.8b | default | count | 32 | True | 53.92 | 140.9 | 2.6 | 0.000 | 3.68 |
| qwen3.5-0.8b | default | ovd | 1 | True | 0.23 | 86.7 | 378.2 | 0.562 | 1.82 |
| qwen3.5-0.8b | default | ovd | 8 | True | 1.24 | 467.8 | 376.2 | 0.578 | 2.36 |
| qwen3.5-0.8b | default | ovd | 16 | True | 2.47 | 925.5 | 374.7 | 0.578 | 2.97 |
| qwen3.5-0.8b | default | ovd | 32 | True | 4.88 | 1827.0 | 374.7 | 0.578 | 4.20 |
| qwen3.5-0.8b | default | rel_depth_nyu | 1 | True | 1.33 | 85.1 | 64.0 | 1.000 | 1.80 |
| qwen3.5-0.8b | default | rel_depth_nyu | 8 | True | 10.07 | 644.6 | 64.0 | 1.000 | 2.22 |
| qwen3.5-0.8b | default | rel_depth_nyu | 16 | True | 19.50 | 1247.9 | 64.0 | 1.000 | 2.69 |
| qwen3.5-0.8b | default | rel_depth_nyu | 32 | True | 36.77 | 2353.0 | 64.0 | 1.000 | 3.64 |
| qwen3.5-0.8b | default | rel_depth_blink | 1 | True | 1.33 | 85.4 | 64.0 | 1.000 | 1.80 |
| qwen3.5-0.8b | default | rel_depth_blink | 8 | True | 7.24 | 463.4 | 64.0 | 1.000 | 2.25 |
| qwen3.5-0.8b | default | rel_depth_blink | 16 | True | 14.29 | 914.3 | 64.0 | 1.000 | 2.50 |
| qwen3.5-0.8b | default | rel_depth_blink | 32 | True | 14.35 | 918.1 | 64.0 | 1.000 | 2.50 |
| qwen3.5-0.8b | default | spatial_cvbench | 1 | True | 1.33 | 85.1 | 63.8 | 0.960 | 1.81 |
| qwen3.5-0.8b | default | spatial_cvbench | 8 | True | 8.65 | 551.4 | 63.8 | 0.940 | 2.26 |
| qwen3.5-0.8b | default | spatial_cvbench | 16 | True | 14.87 | 950.8 | 64.0 | 0.940 | 2.78 |
| qwen3.5-0.8b | default | spatial_cvbench | 32 | True | 21.56 | 1375.1 | 63.8 | 0.940 | 3.81 |
| qwen3.5-0.8b | default | spatial_embspatial | 1 | True | 1.42 | 84.8 | 59.9 | 0.920 | 1.81 |
| qwen3.5-0.8b | default | spatial_embspatial | 8 | True | 8.58 | 529.0 | 61.6 | 0.960 | 2.28 |
| qwen3.5-0.8b | default | spatial_embspatial | 16 | True | 14.72 | 907.0 | 61.6 | 0.960 | 2.81 |
| qwen3.5-0.8b | default | spatial_embspatial | 32 | True | 28.25 | 1740.9 | 61.6 | 0.960 | 3.88 |
| qwen3.5-2b | default | refexp_box | 1 | True | 1.67 | 78.2 | 46.9 | 0.016 | 4.53 |
| qwen3.5-2b | default | refexp_box | 8 | True | 9.56 | 448.1 | 46.9 | 0.016 | 4.99 |
| qwen3.5-2b | default | refexp_box | 16 | True | 15.37 | 720.5 | 46.9 | 0.016 | 5.53 |
| qwen3.5-2b | default | refexp_box | 32 | True | 23.76 | 1115.4 | 46.9 | 0.016 | 6.59 |
| qwen3.5-2b | default | refexp_poly | 1 | True | 1.87 | 78.1 | 41.8 | 0.000 | 4.53 |
| qwen3.5-2b | default | refexp_poly | 8 | True | 8.03 | 334.8 | 41.7 | 0.000 | 5.04 |
| qwen3.5-2b | default | refexp_poly | 16 | True | 11.19 | 469.1 | 41.9 | 0.000 | 5.61 |
| qwen3.5-2b | default | refexp_poly | 32 | True | 15.53 | 640.0 | 41.2 | 0.000 | 6.76 |
| qwen3.5-2b | default | grid16 | 1 | True | 0.08 | 77.8 | 1012.0 | 0.984 | 4.53 |
| qwen3.5-2b | default | grid16 | 8 | True | 0.56 | 567.9 | 1012.0 | 0.984 | 5.04 |
| qwen3.5-2b | default | grid16 | 16 | True | 1.11 | 1126.6 | 1012.0 | 0.984 | 5.63 |
| qwen3.5-2b | default | grid16 | 32 | True | 2.20 | 2205.9 | 1000.5 | 0.969 | 6.78 |
| qwen3.5-2b | default | count | 1 | True | 18.45 | 39.2 | 2.1 | 0.000 | 4.53 |
| qwen3.5-2b | default | count | 8 | True | 91.67 | 194.6 | 2.1 | 0.000 | 4.99 |
| qwen3.5-2b | default | count | 16 | True | 114.38 | 242.8 | 2.1 | 0.000 | 5.51 |
| qwen3.5-2b | default | count | 32 | True | 141.66 | 300.7 | 2.1 | 0.000 | 6.56 |
| qwen3.5-2b | default | ovd | 1 | True | 0.23 | 77.5 | 340.8 | 0.453 | 4.54 |
| qwen3.5-2b | default | ovd | 8 | True | 1.12 | 378.5 | 338.0 | 0.422 | 5.13 |
| qwen3.5-2b | default | ovd | 16 | True | 2.20 | 742.7 | 337.5 | 0.422 | 5.79 |
| qwen3.5-2b | default | ovd | 32 | True | 4.34 | 1481.0 | 341.1 | 0.438 | 7.12 |
| qwen3.5-2b | default | rel_depth_nyu | 1 | True | 1.18 | 75.4 | 64.0 | 1.000 | 4.53 |
| qwen3.5-2b | default | rel_depth_nyu | 8 | True | 8.86 | 567.1 | 64.0 | 1.000 | 4.97 |
| qwen3.5-2b | default | rel_depth_nyu | 16 | True | 16.90 | 1081.3 | 64.0 | 1.000 | 5.48 |
| qwen3.5-2b | default | rel_depth_nyu | 32 | True | 32.11 | 2055.2 | 64.0 | 1.000 | 6.50 |
| qwen3.5-2b | default | rel_depth_blink | 1 | True | 1.27 | 81.0 | 64.0 | 1.000 | 4.53 |
| qwen3.5-2b | default | rel_depth_blink | 8 | True | 6.83 | 437.3 | 64.0 | 1.000 | 5.01 |
| qwen3.5-2b | default | rel_depth_blink | 16 | True | 13.09 | 838.1 | 64.0 | 1.000 | 5.29 |
| qwen3.5-2b | default | rel_depth_blink | 32 | True | 13.29 | 850.4 | 64.0 | 1.000 | 5.29 |
| qwen3.5-2b | default | spatial_cvbench | 1 | True | 1.29 | 80.9 | 62.9 | 0.940 | 4.53 |
| qwen3.5-2b | default | spatial_cvbench | 8 | True | 8.12 | 511.2 | 62.9 | 0.940 | 5.02 |
| qwen3.5-2b | default | spatial_cvbench | 16 | True | 13.85 | 864.0 | 62.4 | 0.920 | 5.59 |
| qwen3.5-2b | default | spatial_cvbench | 32 | True | 25.55 | 1614.1 | 63.2 | 0.960 | 6.71 |
| qwen3.5-2b | default | spatial_embspatial | 1 | True | 1.41 | 80.5 | 57.2 | 0.860 | 4.53 |
| qwen3.5-2b | default | spatial_embspatial | 8 | True | 8.13 | 464.6 | 57.2 | 0.860 | 5.05 |
| qwen3.5-2b | default | spatial_embspatial | 16 | True | 13.75 | 773.6 | 56.3 | 0.840 | 5.62 |
| qwen3.5-2b | default | spatial_embspatial | 32 | True | 25.54 | 1459.9 | 57.2 | 0.860 | 6.77 |
| gemma-4-e2b-it | default | refexp_box | 1 | True | 1.19 | 50.9 | 42.7 | 0.141 | 10.33 |
| gemma-4-e2b-it | default | refexp_box | 8 | True | 3.87 | 163.2 | 42.2 | 0.125 | 10.92 |
| gemma-4-e2b-it | default | refexp_box | 16 | True | 6.02 | 253.3 | 42.1 | 0.125 | 11.59 |
| gemma-4-e2b-it | default | refexp_box | 32 | True | 10.72 | 449.6 | 42.0 | 0.125 | 12.92 |
| gemma-4-e2b-it | default | refexp_poly | 1 | True | 0.70 | 52.5 | 74.5 | 0.000 | 10.33 |
| gemma-4-e2b-it | default | refexp_poly | 8 | True | 2.45 | 187.5 | 76.6 | 0.000 | 10.93 |
| gemma-4-e2b-it | default | refexp_poly | 16 | True | 2.80 | 215.9 | 77.2 | 0.000 | 11.62 |
| gemma-4-e2b-it | default | refexp_poly | 32 | True | 4.41 | 343.4 | 77.9 | 0.000 | 12.98 |
| gemma-4-e2b-it | default | grid16 | 1 | True | 0.21 | 51.0 | 245.5 | 0.000 | 10.33 |
| gemma-4-e2b-it | default | grid16 | 8 | True | 1.49 | 372.1 | 249.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | default | grid16 | 16 | True | 2.90 | 710.2 | 244.5 | 0.000 | 11.62 |
| gemma-4-e2b-it | default | grid16 | 32 | True | 5.61 | 1396.9 | 249.0 | 0.000 | 12.98 |
| gemma-4-e2b-it | default | count | 1 | True | 16.88 | 35.5 | 2.1 | 0.000 | 10.33 |
| gemma-4-e2b-it | default | count | 8 | True | 57.27 | 120.4 | 2.1 | 0.000 | 10.92 |
| gemma-4-e2b-it | default | count | 16 | True | 69.48 | 146.1 | 2.1 | 0.000 | 11.59 |
| gemma-4-e2b-it | default | count | 32 | True | 78.52 | 165.1 | 2.1 | 0.000 | 12.93 |
| gemma-4-e2b-it | default | ovd | 1 | True | 0.37 | 52.9 | 144.7 | 0.000 | 10.33 |
| gemma-4-e2b-it | default | ovd | 8 | True | 1.38 | 205.0 | 148.2 | 0.000 | 10.95 |
| gemma-4-e2b-it | default | ovd | 16 | True | 2.56 | 386.9 | 150.9 | 0.000 | 11.66 |
| gemma-4-e2b-it | default | ovd | 32 | True | 4.32 | 640.1 | 148.2 | 0.000 | 13.06 |
| gemma-4-e2b-it | default | rel_depth_nyu | 1 | True | 0.83 | 52.9 | 64.0 | 1.000 | 10.33 |
| gemma-4-e2b-it | default | rel_depth_nyu | 8 | True | 6.41 | 410.1 | 64.0 | 1.000 | 10.92 |
| gemma-4-e2b-it | default | rel_depth_nyu | 16 | True | 12.01 | 768.9 | 64.0 | 1.000 | 11.60 |
| gemma-4-e2b-it | default | rel_depth_nyu | 32 | True | 21.49 | 1375.4 | 64.0 | 1.000 | 12.95 |
| gemma-4-e2b-it | default | rel_depth_blink | 1 | True | 0.80 | 51.2 | 64.0 | 1.000 | 10.33 |
| gemma-4-e2b-it | default | rel_depth_blink | 8 | True | 4.21 | 269.4 | 64.0 | 1.000 | 10.92 |
| gemma-4-e2b-it | default | rel_depth_blink | 16 | True | 8.09 | 518.1 | 64.0 | 1.000 | 11.26 |
| gemma-4-e2b-it | default | rel_depth_blink | 32 | True | 8.29 | 530.5 | 64.0 | 1.000 | 11.26 |
| gemma-4-e2b-it | default | spatial_cvbench | 1 | True | 0.90 | 50.6 | 56.5 | 0.700 | 10.33 |
| gemma-4-e2b-it | default | spatial_cvbench | 8 | True | 5.16 | 291.1 | 56.5 | 0.720 | 10.93 |
| gemma-4-e2b-it | default | spatial_cvbench | 16 | True | 9.13 | 515.3 | 56.5 | 0.700 | 11.61 |
| gemma-4-e2b-it | default | spatial_cvbench | 32 | True | 16.14 | 911.1 | 56.5 | 0.720 | 12.96 |
| gemma-4-e2b-it | default | spatial_embspatial | 1 | True | 0.85 | 53.7 | 63.4 | 0.980 | 10.33 |
| gemma-4-e2b-it | default | spatial_embspatial | 8 | True | 5.34 | 339.0 | 63.4 | 0.980 | 10.93 |
| gemma-4-e2b-it | default | spatial_embspatial | 16 | True | 8.92 | 566.0 | 63.4 | 0.980 | 11.62 |
| gemma-4-e2b-it | default | spatial_embspatial | 32 | True | 16.51 | 1047.3 | 63.4 | 0.980 | 12.98 |
| internvl3.5-2b-hf | default | refexp_box | 1 | True | 2.98 | 84.0 | 28.1 | 0.000 | 4.79 |
| internvl3.5-2b-hf | default | refexp_box | 8 | True | 14.04 | 395.3 | 28.2 | 0.000 | 5.16 |
| internvl3.5-2b-hf | default | refexp_box | 16 | True | 21.65 | 609.3 | 28.1 | 0.000 | 5.59 |
| internvl3.5-2b-hf | default | refexp_box | 32 | True | 30.51 | 859.1 | 28.2 | 0.000 | 6.43 |
| internvl3.5-2b-hf | default | refexp_poly | 1 | True | 0.31 | 81.3 | 262.6 | 0.031 | 4.89 |
| internvl3.5-2b-hf | default | refexp_poly | 8 | True | 0.95 | 261.7 | 276.2 | 0.047 | 6.10 |
| internvl3.5-2b-hf | default | refexp_poly | 16 | True | 1.03 | 280.2 | 271.1 | 0.016 | 7.47 |
| internvl3.5-2b-hf | default | refexp_poly | 32 | True | 1.07 | 294.9 | 274.4 | 0.031 | 10.23 |
| internvl3.5-2b-hf | default | grid16 | 1 | True | 0.37 | 91.9 | 251.7 | 0.000 | 4.84 |
| internvl3.5-2b-hf | default | grid16 | 8 | True | 2.48 | 629.6 | 253.6 | 0.000 | 5.37 |
| internvl3.5-2b-hf | default | grid16 | 16 | True | 4.88 | 1234.8 | 252.8 | 0.000 | 6.00 |
| internvl3.5-2b-hf | default | grid16 | 32 | True | 8.98 | 2269.0 | 252.8 | 0.000 | 7.26 |
| internvl3.5-2b-hf | default | count | 1 | True | 3.13 | 87.7 | 28.0 | 0.286 | 4.78 |
| internvl3.5-2b-hf | default | count | 8 | True | 9.31 | 270.8 | 29.1 | 0.306 | 5.15 |
| internvl3.5-2b-hf | default | count | 16 | True | 16.87 | 490.5 | 29.1 | 0.306 | 5.56 |
| internvl3.5-2b-hf | default | count | 32 | True | 13.47 | 391.6 | 29.1 | 0.306 | 6.40 |
| internvl3.5-2b-hf | default | ovd | 1 | True | 0.61 | 92.1 | 151.0 | 0.047 | 4.85 |
| internvl3.5-2b-hf | default | ovd | 8 | True | 1.61 | 242.3 | 150.8 | 0.016 | 5.69 |
| internvl3.5-2b-hf | default | ovd | 16 | True | 1.99 | 332.1 | 167.2 | 0.062 | 6.63 |
| internvl3.5-2b-hf | default | ovd | 32 | True | 2.83 | 473.3 | 167.2 | 0.062 | 8.53 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 1 | True | 1.50 | 96.1 | 64.0 | 1.000 | 4.79 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 8 | True | 9.41 | 602.5 | 64.0 | 1.000 | 5.18 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 16 | True | 15.19 | 972.0 | 64.0 | 1.000 | 5.62 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 32 | True | 21.65 | 1385.8 | 64.0 | 1.000 | 6.51 |
| internvl3.5-2b-hf | default | rel_depth_blink | 1 | True | 1.47 | 94.3 | 64.0 | 1.000 | 4.79 |
| internvl3.5-2b-hf | default | rel_depth_blink | 8 | True | 4.18 | 267.8 | 64.0 | 1.000 | 5.18 |
| internvl3.5-2b-hf | default | rel_depth_blink | 16 | True | 5.43 | 347.6 | 64.0 | 1.000 | 5.40 |
| internvl3.5-2b-hf | default | rel_depth_blink | 32 | True | 17.14 | 1097.1 | 64.0 | 1.000 | 5.40 |
| internvl3.5-2b-hf | default | spatial_cvbench | 1 | True | 1.57 | 96.1 | 61.1 | 0.880 | 4.79 |
| internvl3.5-2b-hf | default | spatial_cvbench | 8 | True | 7.32 | 450.8 | 61.6 | 0.860 | 5.20 |
| internvl3.5-2b-hf | default | spatial_cvbench | 16 | True | 16.25 | 1002.2 | 61.7 | 0.880 | 5.67 |
| internvl3.5-2b-hf | default | spatial_cvbench | 32 | True | 15.62 | 963.7 | 61.7 | 0.880 | 6.62 |
| internvl3.5-2b-hf | default | spatial_embspatial | 1 | True | 2.08 | 92.4 | 44.3 | 0.620 | 4.80 |
| internvl3.5-2b-hf | default | spatial_embspatial | 8 | True | 9.70 | 413.8 | 42.7 | 0.560 | 5.24 |
| internvl3.5-2b-hf | default | spatial_embspatial | 16 | True | 19.10 | 813.7 | 42.6 | 0.560 | 5.74 |
| internvl3.5-2b-hf | default | spatial_embspatial | 32 | True | 22.46 | 959.1 | 42.7 | 0.580 | 6.76 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 1 | True | 1.72 | 80.0 | 46.5 | 0.016 | 1.90 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 8 | True | 9.47 | 440.3 | 46.5 | 0.016 | 2.98 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 16 | True | 14.74 | 685.3 | 46.5 | 0.016 | 4.23 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 32 | True | 20.84 | 968.7 | 46.5 | 0.016 | 6.70 |
| qwen3.5-0.8b | qwen:1024 | ovd | 1 | True | 0.23 | 84.4 | 368.2 | 0.531 | 1.90 |
| qwen3.5-0.8b | qwen:1024 | ovd | 8 | True | 1.20 | 439.7 | 367.1 | 0.516 | 3.07 |
| qwen3.5-0.8b | qwen:1024 | ovd | 16 | True | 1.94 | 727.1 | 373.9 | 0.562 | 4.40 |
| qwen3.5-0.8b | qwen:1024 | ovd | 32 | True | 3.23 | 1206.0 | 373.9 | 0.562 | 7.05 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 1 | True | 2.54 | 71.5 | 28.1 | 0.000 | 5.35 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 8 | True | 7.48 | 210.3 | 28.1 | 0.000 | 9.68 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 16 | True | 8.74 | 245.5 | 28.1 | 0.000 | 14.51 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 32 | True | 10.18 | 286.0 | 28.1 | 0.000 | 24.29 |
| internvl3.5-2b-hf | internvl:native | ovd | 1 | True | 0.45 | 80.3 | 180.3 | 0.094 | 5.37 |
| internvl3.5-2b-hf | internvl:native | ovd | 8 | True | 0.78 | 141.1 | 181.2 | 0.078 | 9.93 |
| internvl3.5-2b-hf | internvl:native | ovd | 16 | True | 0.92 | 171.3 | 187.1 | 0.109 | 15.10 |
| internvl3.5-2b-hf | internvl:native | ovd | 32 | True | 1.06 | 198.3 | 187.1 | 0.109 | 25.45 |

#### 开思考的实测（预算 4096；开发样本 ≤ 8 条）

| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 8 | 8 | True | 4.52 | 118.2 | 0.000 | 2.23 |
| qwen3.5-0.8b | refexp_poly | 8 | 8 | True | 0.14 | 2265.8 | 0.500 | 2.71 |
| qwen3.5-0.8b | grid16 | 8 | 8 | True | 0.15 | 3253.5 | 0.750 | 2.71 |
| qwen3.5-0.8b | count | 8 | 8 | True | 0.15 | 1482.4 | 0.250 | 2.70 |
| qwen3.5-0.8b | ovd | 8 | 8 | True | 0.16 | 1788.1 | 0.250 | 2.74 |
| qwen3.5-0.8b | rel_depth_nyu | 8 | 8 | True | 0.16 | 2731.2 | 0.500 | 2.44 |
| qwen3.5-0.8b | rel_depth_blink | 8 | 8 | True | 0.15 | 1421.8 | 0.250 | 2.71 |
| qwen3.5-0.8b | spatial_cvbench | 8 | 8 | True | 0.15 | 2446.5 | 0.500 | 2.71 |
| qwen3.5-0.8b | spatial_embspatial | 8 | 8 | True | 0.15 | 1288.0 | 0.250 | 2.72 |
| qwen3.5-2b | refexp_box | 8 | 8 | True | 3.79 | 126.5 | 0.000 | 4.99 |
| qwen3.5-2b | refexp_poly | 8 | 8 | True | 0.15 | 2271.0 | 0.500 | 5.43 |
| qwen3.5-2b | grid16 | 8 | 8 | True | 0.14 | 3758.9 | 0.875 | 5.44 |
| qwen3.5-2b | count | 8 | 8 | True | 0.21 | 749.4 | 0.000 | 5.19 |
| qwen3.5-2b | ovd | 8 | 8 | True | 0.21 | 1488.1 | 0.000 | 5.25 |
| qwen3.5-2b | rel_depth_nyu | 8 | 8 | True | 0.16 | 3168.5 | 0.750 | 5.16 |
| qwen3.5-2b | rel_depth_blink | 8 | 8 | True | 0.15 | 1905.8 | 0.250 | 5.43 |
| qwen3.5-2b | spatial_cvbench | 8 | 8 | True | 0.15 | 1330.6 | 0.125 | 5.43 |
| qwen3.5-2b | spatial_embspatial | 8 | 8 | True | 0.15 | 1337.2 | 0.250 | 5.44 |
| gemma-4-e2b-it | refexp_box | 8 | 8 | True | 0.52 | 560.4 | 0.000 | 10.92 |
| gemma-4-e2b-it | refexp_poly | 8 | 8 | True | 1.84 | 180.5 | 0.000 | 10.93 |
| gemma-4-e2b-it | grid16 | 8 | 8 | True | 0.42 | 676.1 | 0.000 | 10.93 |
| gemma-4-e2b-it | count | 8 | 8 | True | 0.70 | 323.8 | 0.000 | 10.92 |
| gemma-4-e2b-it | ovd | 8 | 8 | True | 0.37 | 412.1 | 0.000 | 10.95 |
| gemma-4-e2b-it | rel_depth_nyu | 8 | 8 | True | 0.43 | 583.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | rel_depth_blink | 8 | 8 | True | 0.30 | 707.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | spatial_cvbench | 8 | 8 | True | 0.37 | 497.1 | 0.000 | 10.92 |
| gemma-4-e2b-it | spatial_embspatial | 8 | 8 | True | 0.20 | 755.8 | 0.000 | 10.93 |
| internvl3.5-2b-hf | refexp_box | 8 | 8 | True | 0.05 | 3104.4 | 0.625 | 9.18 |
| internvl3.5-2b-hf | refexp_poly | 8 | 8 | True | 0.13 | 3443.6 | 0.625 | 9.24 |
| internvl3.5-2b-hf | grid16 | 8 | 8 | True | 0.13 | 4096.0 | 1.000 | 9.25 |
| internvl3.5-2b-hf | count | 8 | 8 | True | 0.14 | 3153.8 | 0.750 | 9.18 |
| internvl3.5-2b-hf | ovd | 8 | 8 | True | 0.06 | 3331.1 | 0.750 | 9.18 |
| internvl3.5-2b-hf | rel_depth_nyu | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | rel_depth_blink | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | spatial_cvbench | 8 | 8 | True | 0.14 | 3180.6 | 0.750 | 9.21 |
| internvl3.5-2b-hf | spatial_embspatial | 8 | 8 | True | 0.14 | 4096.0 | 1.000 | 9.25 |
