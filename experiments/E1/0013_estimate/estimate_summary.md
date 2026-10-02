#### 估算汇总（主口径；推荐批大小 = 可行的批大小里样本/秒最高者；GPU 小时 = n / 样本每秒，由 run_e1 外推）

| 模型 | 任务 | 推荐批大小 | 样本/秒 | n(eval) | GPU h(eval) | 思考实测批大小 | 思考样本/秒 | n(think) | GPU h(think) | n(8192) | GPU h(8192, 上界) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 32 | 17.08 | 56857 | 0.925 | 8 | 4.52 | 200 | 0.012 | 50 | 0.006 |
| qwen3.5-0.8b | refexp_poly | 32 | 12.50 | 8000 | 0.178 | 8 | 0.15 | 200 | 0.368 | 50 | 0.184 |
| qwen3.5-0.8b | grid16 | 32 | 2.25 | 500 | 0.062 | 8 | 0.15 | 200 | 0.374 | 50 | 0.187 |
| qwen3.5-0.8b | count | 32 | 128.81 | 442 | 0.001 | 8 | 0.14 | 200 | 0.393 | 50 | 0.196 |
| qwen3.5-0.8b | ovd | 32 | 1.17 | 963 | 0.229 | 8 | 0.15 | 200 | 0.368 | 50 | 0.184 |
| qwen3.5-0.8b | rel_depth_nyu | 32 | 3.28 | 3270 | 0.277 | 8 | 0.16 | 200 | 0.351 | 50 | 0.176 |
| qwen3.5-0.8b | rel_depth_blink | 8 | 1.29 | 112 | 0.024 | 8 | 0.15 | 112 | 0.205 | 50 | 0.183 |
| qwen3.5-0.8b | spatial_cvbench | 32 | 2.67 | 2588 | 0.270 | 8 | 0.14 | 200 | 0.392 | 50 | 0.196 |
| qwen3.5-0.8b | spatial_embspatial | 32 | 6.57 | 3590 | 0.152 | 8 | 0.15 | 200 | 0.375 | 50 | 0.187 |
| qwen3.5-0.8b | **合计** | | | | **2.116** | | | | **2.837** | | **1.499** |
| qwen3.5-2b | refexp_box | 32 | 17.57 | 56857 | 0.899 | 8 | 3.79 | 200 | 0.015 | 50 | 0.007 |
| qwen3.5-2b | refexp_poly | 32 | 15.33 | 8000 | 0.145 | 8 | 0.15 | 200 | 0.378 | 50 | 0.189 |
| qwen3.5-2b | grid16 | 32 | 2.27 | 500 | 0.061 | 8 | 0.14 | 200 | 0.397 | 50 | 0.198 |
| qwen3.5-2b | count | 32 | 143.00 | 442 | 0.001 | 8 | 0.22 | 200 | 0.248 | 50 | 0.124 |
| qwen3.5-2b | ovd | 32 | 1.15 | 963 | 0.233 | 8 | 0.21 | 200 | 0.262 | 50 | 0.131 |
| qwen3.5-2b | rel_depth_nyu | 32 | 4.68 | 3270 | 0.194 | 8 | 0.15 | 200 | 0.383 | 50 | 0.191 |
| qwen3.5-2b | rel_depth_blink | 32 | 1.31 | 112 | 0.024 | 8 | 0.14 | 112 | 0.224 | 50 | 0.200 |
| qwen3.5-2b | spatial_cvbench | 32 | 2.42 | 2588 | 0.298 | 8 | 0.14 | 200 | 0.400 | 50 | 0.200 |
| qwen3.5-2b | spatial_embspatial | 32 | 5.49 | 3590 | 0.182 | 8 | 0.14 | 200 | 0.398 | 50 | 0.199 |
| qwen3.5-2b | **合计** | | | | **2.037** | | | | **2.704** | | **1.440** |
| gemma-4-e2b-it | refexp_box | 32 | 6.03 | 56857 | 2.617 | 8 | 0.56 | 200 | 0.100 | 50 | 0.050 |
| gemma-4-e2b-it | refexp_poly | 32 | 4.45 | 8000 | 0.500 | 8 | 1.88 | 200 | 0.030 | 50 | 0.015 |
| gemma-4-e2b-it | grid16 | 32 | 6.07 | 500 | 0.023 | 8 | 0.46 | 200 | 0.122 | 50 | 0.061 |
| gemma-4-e2b-it | count | 32 | 77.69 | 442 | 0.002 | 8 | 0.66 | 200 | 0.085 | 50 | 0.042 |
| gemma-4-e2b-it | ovd | 32 | 4.08 | 963 | 0.066 | 8 | 0.35 | 200 | 0.160 | 50 | 0.080 |
| gemma-4-e2b-it | rel_depth_nyu | 32 | 2.38 | 3270 | 0.381 | 8 | 0.46 | 200 | 0.121 | 50 | 0.060 |
| gemma-4-e2b-it | rel_depth_blink | 32 | 1.33 | 112 | 0.023 | 8 | 0.33 | 112 | 0.096 | 50 | 0.085 |
| gemma-4-e2b-it | spatial_cvbench | 32 | 1.76 | 2588 | 0.408 | 8 | 0.37 | 200 | 0.152 | 50 | 0.076 |
| gemma-4-e2b-it | spatial_embspatial | 32 | 1.55 | 3590 | 0.643 | 8 | 0.20 | 200 | 0.279 | 50 | 0.139 |
| gemma-4-e2b-it | **合计** | | | | **4.663** | | | | **1.142** | | **0.609** |
| internvl3.5-2b-hf | refexp_box | 32 | 31.89 | 56857 | 0.495 | 8 | 0.05 | 200 | 1.093 | 50 | 0.547 |
| internvl3.5-2b-hf | refexp_poly | 32 | 1.07 | 8000 | 2.078 | 8 | 0.13 | 200 | 0.424 | 50 | 0.212 |
| internvl3.5-2b-hf | grid16 | 32 | 8.98 | 500 | 0.015 | 8 | 0.13 | 200 | 0.417 | 50 | 0.209 |
| internvl3.5-2b-hf | count | 16 | 4.83 | 442 | 0.025 | 8 | 0.13 | 200 | 0.412 | 50 | 0.206 |
| internvl3.5-2b-hf | ovd | 8 | 1.40 | 963 | 0.191 | 8 | 0.06 | 200 | 0.899 | 50 | 0.450 |
| internvl3.5-2b-hf | rel_depth_nyu | 32 | 6.40 | 3270 | 0.142 | 8 | 0.19 | 200 | 0.286 | 50 | 0.143 |
| internvl3.5-2b-hf | rel_depth_blink | 32 | 3.34 | 112 | 0.009 | 8 | 0.19 | 112 | 0.160 | 50 | 0.143 |
| internvl3.5-2b-hf | spatial_cvbench | 16 | 4.35 | 2588 | 0.165 | 8 | 0.14 | 200 | 0.407 | 50 | 0.204 |
| internvl3.5-2b-hf | spatial_embspatial | 32 | 6.06 | 3590 | 0.164 | 8 | 0.14 | 200 | 0.408 | 50 | 0.204 |
| internvl3.5-2b-hf | **合计** | | | | **3.286** | | | | **4.507** | | **2.316** |

#### 每个模型的合计（GPU 小时，主口径）

| 模型 | eval | think | think8192（上界） | eval+think+think8192 |
|---|---|---|---|---|
| qwen3.5-0.8b | 2.116 | 2.837 | 1.499 | 6.452 |
| qwen3.5-2b | 2.037 | 2.704 | 1.440 | 6.181 |
| gemma-4-e2b-it | 4.663 | 1.142 | 0.609 | 6.413 |
| internvl3.5-2b-hf | 3.286 | 4.507 | 2.316 | 10.110 |
| 已测 4 个模型之和 | 12.102 | 11.191 | 5.864 | 29.156 |

未测的模型：qwen3.5-4b, gemma-4-e4b-it, internvl3.5-4b-hf, qwen3.5-9b

#### 附加口径（只跑 refexp_box、ovd；只外推 extra 子集）

| 模型 | 口径 | 任务 | 推荐批大小 | 样本/秒 | n(extra) | GPU h(extra) | 主口径同任务样本/秒 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | qwen:1024 | refexp_box | 32 | 14.93 | 500 | 0.009 | 17.08 |
| qwen3.5-0.8b | qwen:1024 | ovd | 32 | 1.11 | 100 | 0.025 | 1.17 |
| qwen3.5-0.8b | qwen:1024 | **合计** | | | | **0.034** | |
| internvl3.5-2b-hf | internvl:native | refexp_box | 32 | 10.18 | 500 | 0.014 | 31.89 |
| internvl3.5-2b-hf | internvl:native | ovd | 8 | 0.37 | 100 | 0.075 | 1.40 |
| internvl3.5-2b-hf | internvl:native | **合计** | | | | **0.089** | |

#### 逐批大小明细（关思考；峰值显存 = torch.cuda.max_memory_allocated）

| 模型 | 口径 | 任务 | 批大小 | ok | 样本/秒 | token/秒 | 平均生成 token | 截断率 | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | default | refexp_box | 8 | True | 8.32 | 398.2 | 47.9 | 0.000 | 2.24 |
| qwen3.5-0.8b | default | refexp_box | 16 | True | 12.66 | 605.6 | 47.8 | 0.000 | 2.73 |
| qwen3.5-0.8b | default | refexp_box | 32 | True | 17.08 | 817.0 | 47.8 | 0.000 | 3.71 |
| qwen3.5-0.8b | default | refexp_poly | 8 | True | 6.71 | 356.6 | 53.1 | 0.000 | 2.28 |
| qwen3.5-0.8b | default | refexp_poly | 16 | True | 8.35 | 458.7 | 54.9 | 0.000 | 2.80 |
| qwen3.5-0.8b | default | refexp_poly | 32 | True | 12.50 | 678.7 | 54.3 | 0.000 | 3.87 |
| qwen3.5-0.8b | default | grid16 | 8 | True | 0.59 | 599.4 | 1024.0 | 1.000 | 2.28 |
| qwen3.5-0.8b | default | grid16 | 16 | True | 1.13 | 1159.5 | 1024.0 | 1.000 | 2.82 |
| qwen3.5-0.8b | default | grid16 | 32 | True | 2.25 | 2308.7 | 1024.0 | 1.000 | 3.89 |
| qwen3.5-0.8b | default | count | 8 | True | 51.55 | 134.7 | 2.6 | 0.000 | 2.23 |
| qwen3.5-0.8b | default | count | 16 | True | 86.62 | 226.3 | 2.6 | 0.000 | 2.71 |
| qwen3.5-0.8b | default | count | 32 | True | 128.81 | 336.5 | 2.6 | 0.000 | 3.68 |
| qwen3.5-0.8b | default | ovd | 8 | True | 0.29 | 365.9 | 1242.0 | 0.562 | 2.40 |
| qwen3.5-0.8b | default | ovd | 16 | True | 0.59 | 732.7 | 1240.6 | 0.562 | 3.05 |
| qwen3.5-0.8b | default | ovd | 32 | True | 1.17 | 1449.1 | 1240.6 | 0.562 | 4.36 |
| qwen3.5-0.8b | default | rel_depth_nyu | 8 | True | 1.22 | 334.5 | 273.7 | 0.000 | 2.22 |
| qwen3.5-0.8b | default | rel_depth_nyu | 16 | True | 2.20 | 578.1 | 263.1 | 0.031 | 2.69 |
| qwen3.5-0.8b | default | rel_depth_nyu | 32 | True | 3.28 | 875.0 | 266.5 | 0.031 | 3.64 |
| qwen3.5-0.8b | default | rel_depth_blink | 8 | True | 1.29 | 318.1 | 246.2 | 0.000 | 2.25 |
| qwen3.5-0.8b | default | rel_depth_blink | 16 | True | 0.94 | 309.8 | 329.0 | 0.000 | 2.50 |
| qwen3.5-0.8b | default | rel_depth_blink | 32 | True | 0.95 | 311.7 | 329.0 | 0.000 | 2.50 |
| qwen3.5-0.8b | default | spatial_cvbench | 8 | True | 1.29 | 222.8 | 173.2 | 0.040 | 2.26 |
| qwen3.5-0.8b | default | spatial_cvbench | 16 | True | 2.28 | 371.9 | 163.2 | 0.040 | 2.78 |
| qwen3.5-0.8b | default | spatial_cvbench | 32 | True | 2.67 | 398.4 | 149.4 | 0.020 | 3.82 |
| qwen3.5-0.8b | default | spatial_embspatial | 8 | True | 1.98 | 263.5 | 133.3 | 0.000 | 2.28 |
| qwen3.5-0.8b | default | spatial_embspatial | 16 | True | 3.46 | 453.7 | 131.2 | 0.000 | 2.81 |
| qwen3.5-0.8b | default | spatial_embspatial | 32 | True | 6.57 | 846.2 | 128.8 | 0.000 | 3.88 |
| qwen3.5-2b | default | refexp_box | 8 | True | 8.62 | 408.4 | 47.4 | 0.000 | 5.00 |
| qwen3.5-2b | default | refexp_box | 16 | True | 12.65 | 606.4 | 47.9 | 0.000 | 5.53 |
| qwen3.5-2b | default | refexp_box | 32 | True | 17.57 | 843.4 | 48.0 | 0.000 | 6.59 |
| qwen3.5-2b | default | refexp_poly | 8 | True | 7.86 | 328.0 | 41.7 | 0.000 | 5.04 |
| qwen3.5-2b | default | refexp_poly | 16 | True | 10.95 | 459.1 | 41.9 | 0.000 | 5.61 |
| qwen3.5-2b | default | refexp_poly | 32 | True | 15.33 | 631.6 | 41.2 | 0.000 | 6.76 |
| qwen3.5-2b | default | grid16 | 8 | True | 0.58 | 590.2 | 1012.0 | 0.984 | 5.04 |
| qwen3.5-2b | default | grid16 | 16 | True | 1.16 | 1178.9 | 1012.0 | 0.984 | 5.62 |
| qwen3.5-2b | default | grid16 | 32 | True | 2.27 | 2267.7 | 1000.5 | 0.969 | 6.78 |
| qwen3.5-2b | default | count | 8 | True | 63.19 | 134.1 | 2.1 | 0.000 | 4.99 |
| qwen3.5-2b | default | count | 16 | True | 116.51 | 247.3 | 2.1 | 0.000 | 5.51 |
| qwen3.5-2b | default | count | 32 | True | 143.00 | 303.5 | 2.1 | 0.000 | 6.56 |
| qwen3.5-2b | default | ovd | 8 | True | 0.30 | 257.1 | 870.8 | 0.328 | 5.13 |
| qwen3.5-2b | default | ovd | 16 | True | 0.59 | 519.4 | 887.5 | 0.344 | 5.79 |
| qwen3.5-2b | default | ovd | 32 | True | 1.15 | 1039.8 | 906.4 | 0.359 | 7.12 |
| qwen3.5-2b | default | rel_depth_nyu | 8 | True | 1.27 | 273.0 | 215.2 | 0.016 | 4.97 |
| qwen3.5-2b | default | rel_depth_nyu | 16 | True | 1.76 | 398.3 | 226.8 | 0.031 | 5.48 |
| qwen3.5-2b | default | rel_depth_nyu | 32 | True | 4.68 | 888.3 | 189.9 | 0.000 | 6.49 |
| qwen3.5-2b | default | rel_depth_blink | 8 | True | 0.60 | 170.2 | 281.7 | 0.083 | 5.01 |
| qwen3.5-2b | default | rel_depth_blink | 16 | True | 1.30 | 330.6 | 253.7 | 0.000 | 5.29 |
| qwen3.5-2b | default | rel_depth_blink | 32 | True | 1.31 | 331.5 | 253.7 | 0.000 | 5.29 |
| qwen3.5-2b | default | spatial_cvbench | 8 | True | 1.65 | 262.2 | 159.2 | 0.000 | 5.02 |
| qwen3.5-2b | default | spatial_cvbench | 16 | True | 2.04 | 372.4 | 182.6 | 0.040 | 5.58 |
| qwen3.5-2b | default | spatial_cvbench | 32 | True | 2.42 | 417.8 | 172.9 | 0.020 | 6.70 |
| qwen3.5-2b | default | spatial_embspatial | 8 | True | 1.56 | 223.8 | 143.4 | 0.020 | 5.04 |
| qwen3.5-2b | default | spatial_embspatial | 16 | True | 3.80 | 473.9 | 124.7 | 0.000 | 5.62 |
| qwen3.5-2b | default | spatial_embspatial | 32 | True | 5.49 | 714.6 | 130.3 | 0.000 | 6.77 |
| gemma-4-e2b-it | default | refexp_box | 8 | True | 2.70 | 147.1 | 54.4 | 0.047 | 10.92 |
| gemma-4-e2b-it | default | refexp_box | 16 | True | 4.00 | 219.5 | 54.8 | 0.047 | 11.59 |
| gemma-4-e2b-it | default | refexp_box | 32 | True | 6.03 | 329.2 | 54.5 | 0.047 | 12.92 |
| gemma-4-e2b-it | default | refexp_poly | 8 | True | 2.51 | 192.5 | 76.6 | 0.000 | 10.93 |
| gemma-4-e2b-it | default | refexp_poly | 16 | True | 2.84 | 219.1 | 77.2 | 0.000 | 11.62 |
| gemma-4-e2b-it | default | refexp_poly | 32 | True | 4.45 | 346.7 | 77.9 | 0.000 | 12.98 |
| gemma-4-e2b-it | default | grid16 | 8 | True | 1.60 | 399.6 | 249.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | default | grid16 | 16 | True | 3.14 | 766.6 | 244.5 | 0.000 | 11.62 |
| gemma-4-e2b-it | default | grid16 | 32 | True | 6.07 | 1510.9 | 249.0 | 0.000 | 12.98 |
| gemma-4-e2b-it | default | count | 8 | True | 52.88 | 111.2 | 2.1 | 0.000 | 10.92 |
| gemma-4-e2b-it | default | count | 16 | True | 68.25 | 143.5 | 2.1 | 0.000 | 11.59 |
| gemma-4-e2b-it | default | count | 32 | True | 77.69 | 163.3 | 2.1 | 0.000 | 12.93 |
| gemma-4-e2b-it | default | ovd | 8 | True | 1.33 | 196.3 | 148.2 | 0.000 | 10.95 |
| gemma-4-e2b-it | default | ovd | 16 | True | 2.46 | 371.4 | 150.9 | 0.000 | 11.66 |
| gemma-4-e2b-it | default | ovd | 32 | True | 4.08 | 604.2 | 148.2 | 0.000 | 13.06 |
| gemma-4-e2b-it | default | rel_depth_nyu | 8 | True | 0.84 | 256.3 | 306.8 | 0.000 | 10.92 |
| gemma-4-e2b-it | default | rel_depth_nyu | 16 | True | 1.60 | 466.8 | 292.3 | 0.000 | 11.61 |
| gemma-4-e2b-it | default | rel_depth_nyu | 32 | True | 2.38 | 696.2 | 292.3 | 0.000 | 12.95 |
| gemma-4-e2b-it | default | rel_depth_blink | 8 | True | 0.87 | 210.0 | 242.8 | 0.000 | 10.92 |
| gemma-4-e2b-it | default | rel_depth_blink | 16 | True | 1.32 | 358.2 | 270.5 | 0.000 | 11.26 |
| gemma-4-e2b-it | default | rel_depth_blink | 32 | True | 1.33 | 359.2 | 270.5 | 0.000 | 11.26 |
| gemma-4-e2b-it | default | spatial_cvbench | 8 | True | 0.89 | 166.9 | 188.5 | 0.000 | 10.93 |
| gemma-4-e2b-it | default | spatial_cvbench | 16 | True | 1.08 | 212.7 | 196.5 | 0.000 | 11.61 |
| gemma-4-e2b-it | default | spatial_cvbench | 32 | True | 1.76 | 362.4 | 205.8 | 0.000 | 12.96 |
| gemma-4-e2b-it | default | spatial_embspatial | 8 | True | 0.56 | 193.5 | 345.8 | 0.000 | 10.93 |
| gemma-4-e2b-it | default | spatial_embspatial | 16 | True | 0.80 | 275.9 | 346.4 | 0.020 | 11.62 |
| gemma-4-e2b-it | default | spatial_embspatial | 32 | True | 1.55 | 530.8 | 342.1 | 0.000 | 12.98 |
| internvl3.5-2b-hf | default | refexp_box | 8 | True | 14.84 | 417.8 | 28.2 | 0.000 | 5.16 |
| internvl3.5-2b-hf | default | refexp_box | 16 | True | 22.56 | 635.0 | 28.1 | 0.000 | 5.59 |
| internvl3.5-2b-hf | default | refexp_box | 32 | True | 31.89 | 897.9 | 28.2 | 0.000 | 6.44 |
| internvl3.5-2b-hf | default | refexp_poly | 8 | True | 0.95 | 261.2 | 276.2 | 0.047 | 6.10 |
| internvl3.5-2b-hf | default | refexp_poly | 16 | True | 1.03 | 279.6 | 271.1 | 0.016 | 7.47 |
| internvl3.5-2b-hf | default | refexp_poly | 32 | True | 1.07 | 293.4 | 274.4 | 0.031 | 10.23 |
| internvl3.5-2b-hf | default | grid16 | 8 | True | 2.48 | 629.3 | 253.6 | 0.000 | 5.37 |
| internvl3.5-2b-hf | default | grid16 | 16 | True | 4.90 | 1239.0 | 252.8 | 0.000 | 6.00 |
| internvl3.5-2b-hf | default | grid16 | 32 | True | 8.98 | 2270.0 | 252.8 | 0.000 | 7.26 |
| internvl3.5-2b-hf | default | count | 8 | True | 2.93 | 206.5 | 70.5 | 0.143 | 5.29 |
| internvl3.5-2b-hf | default | count | 16 | True | 4.83 | 331.5 | 68.6 | 0.143 | 5.85 |
| internvl3.5-2b-hf | default | count | 32 | True | 3.63 | 252.0 | 69.4 | 0.143 | 6.96 |
| internvl3.5-2b-hf | default | ovd | 8 | True | 1.40 | 215.0 | 153.3 | 0.000 | 5.83 |
| internvl3.5-2b-hf | default | ovd | 16 | True | 0.60 | 147.0 | 245.2 | 0.047 | 9.50 |
| internvl3.5-2b-hf | default | ovd | 32 | True | 0.67 | 163.5 | 245.2 | 0.047 | 14.27 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 8 | True | 2.76 | 436.9 | 158.6 | 0.000 | 5.28 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 16 | True | 4.23 | 703.3 | 166.3 | 0.000 | 5.90 |
| internvl3.5-2b-hf | default | rel_depth_nyu | 32 | True | 6.40 | 1063.8 | 166.3 | 0.000 | 7.08 |
| internvl3.5-2b-hf | default | rel_depth_blink | 8 | True | 1.25 | 241.6 | 193.6 | 0.000 | 5.35 |
| internvl3.5-2b-hf | default | rel_depth_blink | 16 | True | 1.01 | 206.3 | 204.1 | 0.000 | 5.71 |
| internvl3.5-2b-hf | default | rel_depth_blink | 32 | True | 3.34 | 680.9 | 204.1 | 0.000 | 5.71 |
| internvl3.5-2b-hf | default | spatial_cvbench | 8 | True | 2.06 | 299.7 | 145.6 | 0.000 | 5.39 |
| internvl3.5-2b-hf | default | spatial_cvbench | 16 | True | 4.35 | 627.7 | 144.3 | 0.000 | 6.18 |
| internvl3.5-2b-hf | default | spatial_cvbench | 32 | True | 1.62 | 266.7 | 164.3 | 0.020 | 10.15 |
| internvl3.5-2b-hf | default | spatial_embspatial | 8 | True | 2.48 | 313.4 | 126.2 | 0.000 | 5.44 |
| internvl3.5-2b-hf | default | spatial_embspatial | 16 | True | 4.80 | 590.5 | 122.9 | 0.000 | 6.12 |
| internvl3.5-2b-hf | default | spatial_embspatial | 32 | True | 6.06 | 756.1 | 124.7 | 0.000 | 7.65 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 8 | True | 7.72 | 370.5 | 48.0 | 0.000 | 2.98 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 16 | True | 11.50 | 551.3 | 47.9 | 0.000 | 4.22 |
| qwen3.5-0.8b | qwen:1024 | refexp_box | 32 | True | 14.93 | 715.6 | 47.9 | 0.000 | 6.70 |
| qwen3.5-0.8b | qwen:1024 | ovd | 8 | True | 0.30 | 302.3 | 1003.3 | 0.391 | 3.07 |
| qwen3.5-0.8b | qwen:1024 | ovd | 16 | True | 0.57 | 647.0 | 1131.6 | 0.484 | 4.40 |
| qwen3.5-0.8b | qwen:1024 | ovd | 32 | True | 1.11 | 1252.8 | 1131.6 | 0.484 | 7.05 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 8 | True | 7.53 | 211.8 | 28.1 | 0.000 | 9.68 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 16 | True | 8.78 | 246.7 | 28.1 | 0.000 | 14.51 |
| internvl3.5-2b-hf | internvl:native | refexp_box | 32 | True | 10.18 | 286.1 | 28.1 | 0.000 | 24.29 |
| internvl3.5-2b-hf | internvl:native | ovd | 8 | True | 0.37 | 86.5 | 234.8 | 0.016 | 10.21 |
| internvl3.5-2b-hf | internvl:native | ovd | 16 | True | 0.24 | 67.0 | 278.8 | 0.047 | 16.16 |
| internvl3.5-2b-hf | internvl:native | ovd | 32 | True | 0.25 | 68.7 | 278.8 | 0.047 | 27.59 |

#### 开思考的实测（预算 4096；开发样本 ≤ 8 条）

| 模型 | 任务 | 批大小 | 样本数 | ok | 样本/秒 | 平均生成 token | 截断率 | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 8 | 8 | True | 4.52 | 118.2 | 0.000 | 2.23 |
| qwen3.5-0.8b | refexp_poly | 8 | 8 | True | 0.15 | 2265.8 | 0.500 | 2.71 |
| qwen3.5-0.8b | grid16 | 8 | 8 | True | 0.15 | 3253.5 | 0.750 | 2.71 |
| qwen3.5-0.8b | count | 8 | 8 | True | 0.14 | 1482.4 | 0.250 | 2.70 |
| qwen3.5-0.8b | ovd | 8 | 8 | True | 0.15 | 1788.1 | 0.250 | 2.74 |
| qwen3.5-0.8b | rel_depth_nyu | 8 | 8 | True | 0.16 | 2731.2 | 0.500 | 2.44 |
| qwen3.5-0.8b | rel_depth_blink | 8 | 8 | True | 0.15 | 1421.8 | 0.250 | 2.71 |
| qwen3.5-0.8b | spatial_cvbench | 8 | 8 | True | 0.14 | 2446.5 | 0.500 | 2.71 |
| qwen3.5-0.8b | spatial_embspatial | 8 | 8 | True | 0.15 | 1288.0 | 0.250 | 2.72 |
| qwen3.5-2b | refexp_box | 8 | 8 | True | 3.79 | 126.5 | 0.000 | 4.99 |
| qwen3.5-2b | refexp_poly | 8 | 8 | True | 0.15 | 2271.0 | 0.500 | 5.43 |
| qwen3.5-2b | grid16 | 8 | 8 | True | 0.14 | 3758.9 | 0.875 | 5.44 |
| qwen3.5-2b | count | 8 | 8 | True | 0.22 | 749.4 | 0.000 | 5.19 |
| qwen3.5-2b | ovd | 8 | 8 | True | 0.21 | 1488.1 | 0.000 | 5.25 |
| qwen3.5-2b | rel_depth_nyu | 8 | 8 | True | 0.15 | 3168.5 | 0.750 | 5.16 |
| qwen3.5-2b | rel_depth_blink | 8 | 8 | True | 0.14 | 1905.8 | 0.250 | 5.43 |
| qwen3.5-2b | spatial_cvbench | 8 | 8 | True | 0.14 | 1330.6 | 0.125 | 5.43 |
| qwen3.5-2b | spatial_embspatial | 8 | 8 | True | 0.14 | 1337.2 | 0.250 | 5.44 |
| gemma-4-e2b-it | refexp_box | 8 | 8 | True | 0.56 | 560.4 | 0.000 | 10.92 |
| gemma-4-e2b-it | refexp_poly | 8 | 8 | True | 1.88 | 180.5 | 0.000 | 10.93 |
| gemma-4-e2b-it | grid16 | 8 | 8 | True | 0.46 | 676.1 | 0.000 | 10.93 |
| gemma-4-e2b-it | count | 8 | 8 | True | 0.66 | 323.8 | 0.000 | 10.92 |
| gemma-4-e2b-it | ovd | 8 | 8 | True | 0.35 | 412.1 | 0.000 | 10.95 |
| gemma-4-e2b-it | rel_depth_nyu | 8 | 8 | True | 0.46 | 583.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | rel_depth_blink | 8 | 8 | True | 0.33 | 707.4 | 0.000 | 10.93 |
| gemma-4-e2b-it | spatial_cvbench | 8 | 8 | True | 0.37 | 497.1 | 0.000 | 10.92 |
| gemma-4-e2b-it | spatial_embspatial | 8 | 8 | True | 0.20 | 755.8 | 0.000 | 10.93 |
| internvl3.5-2b-hf | refexp_box | 8 | 8 | True | 0.05 | 3104.4 | 0.625 | 9.19 |
| internvl3.5-2b-hf | refexp_poly | 8 | 8 | True | 0.13 | 3443.6 | 0.625 | 9.24 |
| internvl3.5-2b-hf | grid16 | 8 | 8 | True | 0.13 | 4096.0 | 1.000 | 9.25 |
| internvl3.5-2b-hf | count | 8 | 8 | True | 0.13 | 3153.8 | 0.750 | 9.18 |
| internvl3.5-2b-hf | ovd | 8 | 8 | True | 0.06 | 3331.1 | 0.750 | 9.18 |
| internvl3.5-2b-hf | rel_depth_nyu | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | rel_depth_blink | 8 | 8 | True | 0.19 | 4096.0 | 1.000 | 8.98 |
| internvl3.5-2b-hf | spatial_cvbench | 8 | 8 | True | 0.14 | 3180.6 | 0.750 | 9.21 |
| internvl3.5-2b-hf | spatial_embspatial | 8 | 8 | True | 0.14 | 4096.0 | 1.000 | 9.25 |
