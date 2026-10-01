#### 验收汇总（模型 × 任务；判据：答案一致率 ≥ 0.98 且配对差 95% CI 含 0；检测无 GT，只判一致率）

| 模型 | 任务 | n | 答案一致率 | 不一致条数 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 | 通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 100 | 0.3000 | 70 | acc@0.5 | 0.8000 | 0.8200 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.2800 | 否 |
| qwen3.5-0.8b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 | 是 |
| qwen3.5-0.8b | grid16 | 100 | 1.0000 | 0 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-0.8b | count | 49 | 1.0000 | 0 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9592 | 是 |
| qwen3.5-0.8b | ovd | 100 | 0.6300 | 37 | — | — | — | — | — | — | 0.0000 | 否 |
| qwen3.5-0.8b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6300 | 是 |
| qwen3.5-0.8b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5833 | 是 |
| qwen3.5-0.8b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.0200 | 0.0400 | -0.0200 | [-0.0600, 0.0000] | 是 | 0.5600 | 是 |
| qwen3.5-0.8b | spatial_embspatial | 50 | 0.9600 | 2 | accuracy | 0.0200 | 0.0200 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5200 | 否 |
| qwen3.5-2b | refexp_box | 100 | 0.2900 | 71 | acc@0.5 | 0.8600 | 0.8600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2800 | 否 |
| qwen3.5-2b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 | 是 |
| qwen3.5-2b | grid16 | 100 | 1.0000 | 0 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 | 是 |
| qwen3.5-2b | count | 49 | 1.0000 | 0 | accuracy | 0.7143 | 0.7143 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-2b | ovd | 100 | 0.4500 | 55 | — | — | — | — | — | — | 0.0200 | 否 |
| qwen3.5-2b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6200 | 是 |
| qwen3.5-2b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.4167 | 是 |
| qwen3.5-2b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.0400 | 0.0600 | -0.0200 | [-0.0600, 0.0000] | 是 | 0.7400 | 是 |
| qwen3.5-2b | spatial_embspatial | 50 | 0.9800 | 1 | accuracy | 0.1000 | 0.1200 | -0.0200 | [-0.0600, 0.0000] | 是 | 0.5400 | 是 |
| gemma-4-e2b-it | refexp_box | 100 | 0.9100 | 9 | acc@0.5 | 0.0800 | 0.0800 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7200 | 否 |
| gemma-4-e2b-it | refexp_poly | 100 | 0.6900 | 31 | cIoU | 0.2413 | 0.2340 | 0.0073 | [-0.0016, 0.0163] | 是 | 0.5400 | 否 |
| gemma-4-e2b-it | grid16 | 100 | 1.0000 | 0 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9500 | 是 |
| gemma-4-e2b-it | count | 49 | 0.9796 | 1 | accuracy | 0.3673 | 0.3673 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 | 否 |
| gemma-4-e2b-it | ovd | 100 | 0.4500 | 55 | — | — | — | — | — | — | 0.4100 | 否 |
| gemma-4-e2b-it | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7000 | 是 |
| gemma-4-e2b-it | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.4167 | 是 |
| gemma-4-e2b-it | spatial_cvbench | 50 | 1.0000 | 0 | accuracy | 0.2600 | 0.2600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6800 | 是 |
| gemma-4-e2b-it | spatial_embspatial | 50 | 1.0000 | 0 | accuracy | 0.0200 | 0.0200 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6000 | 是 |
| internvl3.5-2b-hf | refexp_box | 100 | 0.0900 | 91 | acc@0.5 | 0.7100 | 0.7300 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.0900 | 否 |
| internvl3.5-2b-hf | refexp_poly | 100 | 0.4300 | 57 | cIoU | 0.1031 | 0.0918 | 0.0113 | [-0.0111, 0.0344] | 是 | 0.1800 | 否 |
| internvl3.5-2b-hf | grid16 | 100 | 0.9900 | 1 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.8700 | 是 |
| internvl3.5-2b-hf | count | 49 | 0.9796 | 1 | accuracy | 0.4490 | 0.4490 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7755 | 否 |
| internvl3.5-2b-hf | ovd | 100 | 1.0000 | 0 | — | — | — | — | — | — | 0.3400 | 是 |
| internvl3.5-2b-hf | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5800 | 是 |
| internvl3.5-2b-hf | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6667 | 是 |
| internvl3.5-2b-hf | spatial_cvbench | 50 | 0.9600 | 2 | accuracy | 0.1000 | 0.1000 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.7200 | 否 |
| internvl3.5-2b-hf | spatial_embspatial | 50 | 0.9200 | 4 | accuracy | 0.1400 | 0.2200 | -0.0800 | [-0.1600, -0.0200] | 否 | 0.6800 | 否 |
| qwen3.5-4b | refexp_box | 100 | 0.3300 | 67 | acc@0.5 | 0.8700 | 0.8700 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 | 否 |
| qwen3.5-4b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2700 | 是 |
| qwen3.5-4b | grid16 | 100 | 0.9900 | 1 | gIoU | 0.0111 | 0.0111 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 | 是 |
| qwen3.5-4b | count | 49 | 1.0000 | 0 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-4b | ovd | 100 | 0.3000 | 70 | — | — | — | — | — | — | 0.0100 | 否 |
| qwen3.5-4b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6400 | 是 |
| qwen3.5-4b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6667 | 是 |
| qwen3.5-4b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.0400 | 0.0400 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6800 | 是 |
| qwen3.5-4b | spatial_embspatial | 50 | 0.8800 | 6 | accuracy | 0.3000 | 0.3800 | -0.0800 | [-0.1800, 0.0005] | 是 | 0.6600 | 否 |
| gemma-4-e4b-it | refexp_box | 100 | 0.9200 | 8 | acc@0.5 | 0.1300 | 0.1300 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6900 | 否 |
| gemma-4-e4b-it | refexp_poly | 100 | 0.9800 | 2 | cIoU | 0.0381 | 0.0336 | 0.0045 | [-0.0000, 0.0122] | 是 | 0.1600 | 是 |
| gemma-4-e4b-it | grid16 | 100 | 0.9800 | 2 | gIoU | 0.0112 | 0.0122 | -0.0010 | [-0.0038, 0.0010] | 是 | 0.9700 | 是 |
| gemma-4-e4b-it | count | 49 | 0.9592 | 2 | accuracy | 0.5102 | 0.4898 | 0.0204 | [0.0000, 0.0612] | 是 | 0.9592 | 否 |
| gemma-4-e4b-it | ovd | 100 | 0.1900 | 81 | — | — | — | — | — | — | 0.1700 | 否 |
| gemma-4-e4b-it | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6700 | 是 |
| gemma-4-e4b-it | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5833 | 是 |
| gemma-4-e4b-it | spatial_cvbench | 50 | 0.9600 | 2 | accuracy | 0.3400 | 0.3800 | -0.0400 | [-0.1000, 0.0000] | 是 | 0.7000 | 否 |
| gemma-4-e4b-it | spatial_embspatial | 50 | 1.0000 | 0 | accuracy | 0.0800 | 0.0800 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6600 | 是 |
| internvl3.5-4b-hf | refexp_box | 100 | 0.4800 | 52 | acc@0.5 | 0.8700 | 0.8600 | 0.0100 | [0.0000, 0.0400] | 是 | 0.4800 | 否 |
| internvl3.5-4b-hf | refexp_poly | 100 | 0.3700 | 63 | cIoU | 0.1630 | 0.1639 | -0.0009 | [-0.0311, 0.0276] | 是 | 0.2500 | 否 |
| internvl3.5-4b-hf | grid16 | 100 | 0.9700 | 3 | gIoU | 0.0044 | 0.0040 | 0.0004 | [0.0000, 0.0012] | 是 | 0.9700 | 否 |
| internvl3.5-4b-hf | count | 49 | 1.0000 | 0 | accuracy | 0.6735 | 0.6735 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 | 是 |
| internvl3.5-4b-hf | ovd | 100 | 0.4500 | 55 | — | — | — | — | — | — | 0.0200 | 否 |
| internvl3.5-4b-hf | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5300 | 是 |
| internvl3.5-4b-hf | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3333 | 是 |
| internvl3.5-4b-hf | spatial_cvbench | 50 | 1.0000 | 0 | accuracy | 0.2400 | 0.2400 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7200 | 是 |
| internvl3.5-4b-hf | spatial_embspatial | 50 | 0.9200 | 4 | accuracy | 0.4600 | 0.4600 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.8400 | 否 |
| qwen3.5-9b | refexp_box | 100 | 0.3500 | 65 | acc@0.5 | 0.8900 | 0.8900 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 | 否 |
| qwen3.5-9b | refexp_poly | 100 | 1.0000 | 0 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3800 | 是 |
| qwen3.5-9b | grid16 | 100 | 0.9500 | 5 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9400 | 否 |
| qwen3.5-9b | count | 49 | 1.0000 | 0 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 | 是 |
| qwen3.5-9b | ovd | 100 | 0.3200 | 68 | — | — | — | — | — | — | 0.0000 | 否 |
| qwen3.5-9b | rel_depth_nyu | 100 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.8200 | 是 |
| qwen3.5-9b | rel_depth_blink | 12 | 1.0000 | 0 | accuracy | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.4167 | 是 |
| qwen3.5-9b | spatial_cvbench | 50 | 0.9800 | 1 | accuracy | 0.1200 | 0.1000 | 0.0200 | [0.0000, 0.0600] | 是 | 0.7400 | 是 |
| qwen3.5-9b | spatial_embspatial | 50 | 0.9800 | 1 | accuracy | 0.3200 | 0.3200 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6800 | 是 |

| 模型 | all_pass | 不通过的任务 |
|---|---|---|
| qwen3.5-0.8b | 否 | refexp_box, ovd, spatial_embspatial |
| qwen3.5-2b | 否 | refexp_box, ovd |
| gemma-4-e2b-it | 否 | refexp_box, refexp_poly, count, ovd |
| internvl3.5-2b-hf | 否 | refexp_box, refexp_poly, count, spatial_cvbench, spatial_embspatial |
| qwen3.5-4b | 否 | refexp_box, ovd, spatial_embspatial |
| gemma-4-e4b-it | 否 | refexp_box, count, ovd, spatial_cvbench |
| internvl3.5-4b-hf | 否 | refexp_box, refexp_poly, grid16, ovd, spatial_embspatial |
| qwen3.5-9b | 否 | refexp_box, grid16, ovd |

#### 截断率与解析失败分布（逐条 bs=1 / 批量；截断 = stopped=False）

| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 | 思考未闭合 bs1 / 批量 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 128 | 0.02 / 0.02 | 2/100 {"truncated": 2} | 2/100 {"truncated": 2} | 47.0 / 47.0 | 0 / 0 |
| qwen3.5-0.8b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 66.3 / 67.5 | 0 / 0 |
| qwen3.5-0.8b | grid16 | 1024 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 1024.0 / 1024.0 | 0 / 0 |
| qwen3.5-0.8b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.3 / 2.6 | 0 / 0 |
| qwen3.5-0.8b | ovd | 512 | 0.66 / 0.66 | 66/100 {"truncated": 66} | 66/100 {"truncated": 66} | 416.1 / 413.6 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-0.8b | spatial_cvbench | 64 | 0.96 / 0.94 | 48/50 {"truncated": 48} | 47/50 {"truncated": 47} | 63.8 / 64.0 | 0 / 0 |
| qwen3.5-0.8b | spatial_embspatial | 64 | 0.92 / 0.96 | 46/50 {"truncated": 46} | 48/50 {"truncated": 48} | 59.9 / 61.6 | 0 / 0 |
| qwen3.5-2b | refexp_box | 128 | 0.01 / 0.01 | 1/100 {"truncated": 1} | 1/100 {"truncated": 1} | 46.7 / 46.7 | 0 / 0 |
| qwen3.5-2b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 43.4 / 42.3 | 0 / 0 |
| qwen3.5-2b | grid16 | 1024 | 0.99 / 0.99 | 100/100 {"truncated": 99, "wrong_row_count": 1} | 100/100 {"truncated": 99, "wrong_row_count": 1} | 1016.3 / 1016.3 | 0 / 0 |
| qwen3.5-2b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| qwen3.5-2b | ovd | 512 | 0.47 / 0.47 | 47/100 {"truncated": 47} | 47/100 {"truncated": 47} | 367.8 / 365.7 | 0 / 0 |
| qwen3.5-2b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-2b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-2b | spatial_cvbench | 64 | 0.94 / 0.92 | 47/50 {"truncated": 47} | 46/50 {"truncated": 46} | 62.9 / 62.4 | 0 / 0 |
| qwen3.5-2b | spatial_embspatial | 64 | 0.86 / 0.84 | 43/50 {"truncated": 43} | 42/50 {"truncated": 42} | 57.2 / 56.3 | 0 / 0 |
| gemma-4-e2b-it | refexp_box | 128 | 0.13 / 0.12 | 88/100 {"multiple_boxes": 3, "no_box": 72, "truncated": 13} | 87/100 {"multiple_boxes": 3, "no_box": 72, "truncated": 12} | 44.0 / 43.8 | 0 / 0 |
| gemma-4-e2b-it | refexp_poly | 1024 | 0.00 / 0.00 | 2/100 {"malformed": 1, "too_few_points": 1} | 2/100 {"malformed": 1, "too_few_points": 1} | 84.5 / 88.2 | 0 / 0 |
| gemma-4-e2b-it | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"empty": 3, "wrong_row_count": 97} | 100/100 {"empty": 3, "wrong_row_count": 97} | 245.3 / 244.7 | 0 / 0 |
| gemma-4-e2b-it | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| gemma-4-e2b-it | ovd | 512 | 0.01 / 0.01 | 35/100 {"no_box": 34, "truncated": 1} | 35/100 {"no_box": 34, "truncated": 1} | 156.9 / 159.0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e2b-it | spatial_cvbench | 64 | 0.70 / 0.70 | 35/50 {"truncated": 35} | 35/50 {"truncated": 35} | 56.5 / 56.5 | 0 / 0 |
| gemma-4-e2b-it | spatial_embspatial | 64 | 0.98 / 0.98 | 49/50 {"truncated": 49} | 49/50 {"truncated": 49} | 63.4 / 63.4 | 0 / 0 |
| internvl3.5-2b-hf | refexp_box | 128 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 | 0 / 0 |
| internvl3.5-2b-hf | refexp_poly | 1024 | 0.02 / 0.02 | 22/100 {"ambiguous": 3, "malformed": 2, "too_few_points": 15, "truncated": 2} | 21/100 {"ambiguous": 1, "malformed": 4, "too_few_points": 14, "truncated": 2} | 254.8 / 265.2 | 0 / 0 |
| internvl3.5-2b-hf | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"wrong_row_count": 100} | 99/100 {"wrong_row_count": 99} | 252.2 / 253.0 | 0 / 0 |
| internvl3.5-2b-hf | count | 64 | 0.29 / 0.31 | 14/49 {"thinking_unclosed": 3, "truncated": 11} | 15/49 {"thinking_unclosed": 4, "truncated": 11} | 28.0 / 29.1 | 3 / 4 |
| internvl3.5-2b-hf | ovd | 512 | 0.04 / 0.05 | 100/100 {"no_box": 96, "truncated": 4} | 100/100 {"no_box": 95, "truncated": 5} | 159.4 / 159.7 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-2b-hf | spatial_cvbench | 64 | 0.88 / 0.88 | 44/50 {"truncated": 44} | 44/50 {"truncated": 44} | 61.1 / 61.7 | 0 / 0 |
| internvl3.5-2b-hf | spatial_embspatial | 64 | 0.62 / 0.56 | 31/50 {"truncated": 31} | 28/50 {"truncated": 28} | 44.3 / 42.6 | 0 / 0 |
| qwen3.5-4b | refexp_box | 128 | 0.03 / 0.03 | 3/100 {"truncated": 3} | 3/100 {"truncated": 3} | 48.0 / 48.0 | 0 / 0 |
| qwen3.5-4b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 40.8 / 41.1 | 0 / 0 |
| qwen3.5-4b | grid16 | 1024 | 0.00 / 0.00 | 1/100 {"wrong_row_count": 1} | 0/100 {} | 272.3 / 272.0 | 0 / 0 |
| qwen3.5-4b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| qwen3.5-4b | ovd | 512 | 0.29 / 0.30 | 29/100 {"truncated": 29} | 30/100 {"truncated": 30} | 339.2 / 339.1 | 0 / 0 |
| qwen3.5-4b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-4b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-4b | spatial_cvbench | 64 | 0.94 / 0.96 | 47/50 {"truncated": 47} | 48/50 {"truncated": 48} | 63.2 / 63.6 | 0 / 0 |
| qwen3.5-4b | spatial_embspatial | 64 | 0.68 / 0.62 | 35/50 {"no_choice": 1, "truncated": 34} | 31/50 {"truncated": 31} | 48.4 / 44.2 | 0 / 0 |
| gemma-4-e4b-it | refexp_box | 128 | 0.08 / 0.09 | 82/100 {"multiple_boxes": 1, "no_box": 73, "truncated": 8} | 83/100 {"multiple_boxes": 1, "no_box": 73, "truncated": 9} | 44.6 / 46.3 | 0 / 0 |
| gemma-4-e4b-it | refexp_poly | 1024 | 0.00 / 0.00 | 93/100 {"malformed": 93} | 93/100 {"malformed": 93} | 71.5 / 71.5 | 0 / 0 |
| gemma-4-e4b-it | grid16 | 1024 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 272.0 / 272.0 | 0 / 0 |
| gemma-4-e4b-it | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 | 0 / 0 |
| gemma-4-e4b-it | ovd | 512 | 0.02 / 0.04 | 3/100 {"no_box": 1, "truncated": 2} | 5/100 {"no_box": 1, "truncated": 4} | 193.6 / 193.4 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| gemma-4-e4b-it | spatial_cvbench | 64 | 0.62 / 0.58 | 32/50 {"no_choice": 1, "truncated": 31} | 30/50 {"no_choice": 1, "truncated": 29} | 52.6 / 52.2 | 0 / 0 |
| gemma-4-e4b-it | spatial_embspatial | 64 | 0.86 / 0.86 | 43/50 {"truncated": 43} | 43/50 {"truncated": 43} | 61.2 / 61.2 | 0 / 0 |
| internvl3.5-4b-hf | refexp_box | 128 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 | 0 / 0 |
| internvl3.5-4b-hf | refexp_poly | 1024 | 0.04 / 0.06 | 5/100 {"out_of_range": 1, "truncated": 4} | 6/100 {"truncated": 6} | 117.7 / 135.1 | 0 / 0 |
| internvl3.5-4b-hf | grid16 | 1024 | 0.00 / 0.00 | 2/100 {"multiple_blocks": 2} | 1/100 {"multiple_blocks": 1} | 272.0 / 272.0 | 0 / 0 |
| internvl3.5-4b-hf | count | 64 | 0.00 / 0.00 | 1/49 {"multiple_numbers": 1} | 1/49 {"multiple_numbers": 1} | 7.6 / 7.6 | 0 / 0 |
| internvl3.5-4b-hf | ovd | 512 | 0.47 / 0.46 | 47/100 {"truncated": 47} | 46/100 {"truncated": 46} | 388.7 / 392.9 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| internvl3.5-4b-hf | spatial_cvbench | 64 | 0.70 / 0.70 | 35/50 {"truncated": 35} | 35/50 {"truncated": 35} | 53.8 / 53.9 | 0 / 0 |
| internvl3.5-4b-hf | spatial_embspatial | 64 | 0.28 / 0.28 | 17/50 {"multiple_choices": 3, "truncated": 14} | 17/50 {"multiple_choices": 3, "truncated": 14} | 34.6 / 34.4 | 0 / 0 |
| qwen3.5-9b | refexp_box | 128 | 0.02 / 0.02 | 2/100 {"truncated": 2} | 2/100 {"truncated": 2} | 46.9 / 47.0 | 0 / 0 |
| qwen3.5-9b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 50.0 / 49.7 | 0 / 0 |
| qwen3.5-9b | grid16 | 1024 | 0.00 / 0.00 | 14/100 {"wrong_row_count": 14} | 13/100 {"wrong_row_count": 13} | 274.4 / 274.2 | 0 / 0 |
| qwen3.5-9b | count | 64 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 | 0 / 0 |
| qwen3.5-9b | ovd | 512 | 0.36 / 0.34 | 36/100 {"truncated": 36} | 34/100 {"truncated": 34} | 346.4 / 346.2 | 0 / 0 |
| qwen3.5-9b | rel_depth_nyu | 64 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-9b | rel_depth_blink | 64 | 1.00 / 1.00 | 12/12 {"truncated": 12} | 12/12 {"truncated": 12} | 64.0 / 64.0 | 0 / 0 |
| qwen3.5-9b | spatial_cvbench | 64 | 0.86 / 0.88 | 43/50 {"truncated": 43} | 44/50 {"truncated": 44} | 63.3 / 63.3 | 0 / 0 |
| qwen3.5-9b | spatial_embspatial | 64 | 0.68 / 0.66 | 34/50 {"truncated": 34} | 33/50 {"truncated": 33} | 46.9 / 45.7 | 0 / 0 |

#### 两遍不一致的细目（只描述两遍 parsed 的差别）

| 模型 | 任务 | 一遍失败一遍成功 | 两遍都失败 | 两遍都成功但 parsed 不同 | 失败原因不同 | 生成 token 数相同 | 框/检测的细目 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 0 | 2 | 70/98 | 0 | 97/100 | 两框 IoU {"n": 70, "min": 0.0, "median": 0.9937, "max": 0.9988}；坐标最大绝对差 px {"n": 70, "min": 0.426, "median": 1.28, "max": 391.68}；IoU<0.9 的 3 条 |
| qwen3.5-0.8b | refexp_poly | 0 | 100 | 0/0 | 0 | 98/100 | — |
| qwen3.5-0.8b | grid16 | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-0.8b | count | 0 | 0 | 0/49 | 0 | 47/49 | — |
| qwen3.5-0.8b | ovd | 6 | 63 | 31/31 | 6 | 85/100 | 标签序列相同 22；框数相同 24；标签与框数都相同时逐框 IoU 的最小值 {"n": 22, "min": 0.7075, "median": 0.9851, "max": 0.9982} |
| qwen3.5-0.8b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-0.8b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-0.8b | spatial_cvbench | 1 | 47 | 0/2 | 1 | 49/50 | — |
| qwen3.5-0.8b | spatial_embspatial | 2 | 46 | 0/2 | 2 | 48/50 | — |
| qwen3.5-2b | refexp_box | 0 | 1 | 71/99 | 0 | 100/100 | 两框 IoU {"n": 71, "min": 0.7975, "median": 0.9935, "max": 0.9989}；坐标最大绝对差 px {"n": 71, "min": 0.404, "median": 1.125, "max": 23.077}；IoU<0.9 的 2 条 |
| qwen3.5-2b | refexp_poly | 0 | 100 | 0/0 | 0 | 96/100 | — |
| qwen3.5-2b | grid16 | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-2b | count | 0 | 0 | 0/49 | 0 | 49/49 | — |
| qwen3.5-2b | ovd | 8 | 43 | 47/49 | 8 | 85/100 | 标签序列相同 42；框数相同 42；标签与框数都相同时逐框 IoU 的最小值 {"n": 42, "min": 0.0, "median": 0.9805, "max": 0.9983} |
| qwen3.5-2b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-2b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-2b | spatial_cvbench | 1 | 46 | 0/3 | 1 | 49/50 | — |
| qwen3.5-2b | spatial_embspatial | 1 | 42 | 0/7 | 1 | 49/50 | — |
| gemma-4-e2b-it | refexp_box | 1 | 87 | 8/12 | 2 | 95/100 | 两框 IoU {"n": 8, "min": 0.384, "median": 0.9963, "max": 0.9987}；坐标最大绝对差 px {"n": 8, "min": 0.544, "median": 0.853, "max": 64.0}；IoU<0.9 的 1 条 |
| gemma-4-e2b-it | refexp_poly | 0 | 2 | 46/98 | 0 | 85/100 | — |
| gemma-4-e2b-it | grid16 | 0 | 100 | 0/0 | 0 | 96/100 | — |
| gemma-4-e2b-it | count | 0 | 0 | 1/49 | 0 | 49/49 | — |
| gemma-4-e2b-it | ovd | 2 | 34 | 53/64 | 2 | 81/100 | 标签序列相同 40；框数相同 42；标签与框数都相同时逐框 IoU 的最小值 {"n": 40, "min": 0.7563, "median": 0.9888, "max": 0.9989} |
| gemma-4-e2b-it | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| gemma-4-e2b-it | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| gemma-4-e2b-it | spatial_cvbench | 0 | 35 | 0/15 | 0 | 50/50 | — |
| gemma-4-e2b-it | spatial_embspatial | 0 | 49 | 0/1 | 0 | 50/50 | — |
| internvl3.5-2b-hf | refexp_box | 0 | 0 | 91/100 | 0 | 96/100 | 两框 IoU {"n": 91, "min": 0.3645, "median": 0.9776, "max": 0.9986}；坐标最大绝对差 px {"n": 91, "min": 0.375, "median": 3.416, "max": 114.56}；IoU<0.9 的 9 条 |
| internvl3.5-2b-hf | refexp_poly | 13 | 15 | 45/72 | 15 | 34/100 | — |
| internvl3.5-2b-hf | grid16 | 1 | 99 | 0/0 | 1 | 87/100 | — |
| internvl3.5-2b-hf | count | 1 | 14 | 0/34 | 2 | 47/49 | — |
| internvl3.5-2b-hf | ovd | 0 | 100 | 0/0 | 5 | 38/100 | 标签序列相同 0；框数相同 0；标签与框数都相同时逐框 IoU 的最小值 null |
| internvl3.5-2b-hf | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| internvl3.5-2b-hf | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| internvl3.5-2b-hf | spatial_cvbench | 2 | 43 | 0/5 | 2 | 48/50 | — |
| internvl3.5-2b-hf | spatial_embspatial | 3 | 28 | 2/19 | 3 | 46/50 | — |
| qwen3.5-4b | refexp_box | 0 | 3 | 67/97 | 0 | 99/100 | 两框 IoU {"n": 67, "min": 0.7705, "median": 0.9945, "max": 0.9988}；坐标最大绝对差 px {"n": 67, "min": 0.375, "median": 0.72, "max": 49.0}；IoU<0.9 的 2 条 |
| qwen3.5-4b | refexp_poly | 0 | 100 | 0/0 | 0 | 93/100 | — |
| qwen3.5-4b | grid16 | 1 | 0 | 0/99 | 1 | 99/100 | — |
| qwen3.5-4b | count | 0 | 0 | 0/49 | 0 | 49/49 | — |
| qwen3.5-4b | ovd | 1 | 29 | 69/70 | 1 | 75/100 | 标签序列相同 55；框数相同 57；标签与框数都相同时逐框 IoU 的最小值 {"n": 55, "min": 0.0, "median": 0.9667, "max": 0.9985} |
| qwen3.5-4b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-4b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-4b | spatial_cvbench | 1 | 47 | 0/2 | 1 | 48/50 | — |
| qwen3.5-4b | spatial_embspatial | 6 | 30 | 0/14 | 6 | 44/50 | — |
| gemma-4-e4b-it | refexp_box | 1 | 82 | 7/17 | 1 | 86/100 | 两框 IoU {"n": 7, "min": 0.9858, "median": 0.9955, "max": 0.9985}；坐标最大绝对差 px {"n": 7, "min": 0.478, "median": 0.64, "max": 1.28}；IoU<0.9 的 0 条 |
| gemma-4-e4b-it | refexp_poly | 0 | 93 | 6/7 | 0 | 97/100 | — |
| gemma-4-e4b-it | grid16 | 0 | 0 | 3/100 | 0 | 100/100 | — |
| gemma-4-e4b-it | count | 0 | 0 | 2/49 | 0 | 49/49 | — |
| gemma-4-e4b-it | ovd | 2 | 3 | 79/95 | 2 | 82/100 | 标签序列相同 64；框数相同 65；标签与框数都相同时逐框 IoU 的最小值 {"n": 64, "min": 0.0747, "median": 0.9873, "max": 0.9984} |
| gemma-4-e4b-it | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| gemma-4-e4b-it | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| gemma-4-e4b-it | spatial_cvbench | 2 | 30 | 0/18 | 2 | 46/50 | — |
| gemma-4-e4b-it | spatial_embspatial | 0 | 43 | 0/7 | 0 | 49/50 | — |
| internvl3.5-4b-hf | refexp_box | 0 | 0 | 52/100 | 0 | 97/100 | 两框 IoU {"n": 52, "min": 0.5045, "median": 0.9905, "max": 0.9979}；坐标最大绝对差 px {"n": 52, "min": 0.426, "median": 1.92, "max": 99.2}；IoU<0.9 的 5 条 |
| internvl3.5-4b-hf | refexp_poly | 5 | 3 | 67/92 | 6 | 63/100 | — |
| internvl3.5-4b-hf | grid16 | 1 | 1 | 2/98 | 1 | 100/100 | — |
| internvl3.5-4b-hf | count | 0 | 1 | 0/48 | 0 | 48/49 | — |
| internvl3.5-4b-hf | ovd | 7 | 43 | 48/50 | 7 | 75/100 | 标签序列相同 33；框数相同 35；标签与框数都相同时逐框 IoU 的最小值 {"n": 33, "min": 0.4091, "median": 0.9279, "max": 0.9963} |
| internvl3.5-4b-hf | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| internvl3.5-4b-hf | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| internvl3.5-4b-hf | spatial_cvbench | 0 | 35 | 0/15 | 0 | 48/50 | — |
| internvl3.5-4b-hf | spatial_embspatial | 4 | 15 | 0/31 | 4 | 45/50 | — |
| qwen3.5-9b | refexp_box | 0 | 2 | 65/98 | 0 | 99/100 | 两框 IoU {"n": 65, "min": 0.7456, "median": 0.9955, "max": 0.9988}；坐标最大绝对差 px {"n": 65, "min": 0.335, "median": 0.956, "max": 12.296}；IoU<0.9 的 1 条 |
| qwen3.5-9b | refexp_poly | 0 | 100 | 0/0 | 0 | 98/100 | — |
| qwen3.5-9b | grid16 | 5 | 11 | 0/84 | 5 | 94/100 | — |
| qwen3.5-9b | count | 0 | 0 | 0/49 | 0 | 49/49 | — |
| qwen3.5-9b | ovd | 6 | 32 | 62/62 | 6 | 84/100 | 标签序列相同 52；框数相同 55；标签与框数都相同时逐框 IoU 的最小值 {"n": 52, "min": 0.0, "median": 0.975, "max": 0.9985} |
| qwen3.5-9b | rel_depth_nyu | 0 | 100 | 0/0 | 0 | 100/100 | — |
| qwen3.5-9b | rel_depth_blink | 0 | 12 | 0/0 | 0 | 12/12 | — |
| qwen3.5-9b | spatial_cvbench | 1 | 43 | 0/6 | 1 | 48/50 | — |
| qwen3.5-9b | spatial_embspatial | 1 | 33 | 0/16 | 1 | 49/50 | — |

#### 每遍的墙钟与生成 token（compute.json；墙钟只含 generate）

| 模型 | 任务 | 墙钟 s bs1 / 批量 | 生成 token bs1 / 批量 |
|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 58.0 / 7.6 | 4700 / 4699 |
| qwen3.5-0.8b | refexp_poly | 80.3 / 23.7 | 6631 / 6747 |
| qwen3.5-0.8b | grid16 | 1211.4 / 92.7 | 102400 / 102400 |
| qwen3.5-0.8b | count | 2.4 / 0.5 | 111 / 128 |
| qwen3.5-0.8b | ovd | 494.8 / 46.8 | 41608 / 41361 |
| qwen3.5-0.8b | rel_depth_nyu | 77.4 / 5.9 | 6400 / 6400 |
| qwen3.5-0.8b | rel_depth_blink | 9.3 / 0.9 | 768 / 768 |
| qwen3.5-0.8b | spatial_cvbench | 38.5 / 3.5 | 3189 / 3198 |
| qwen3.5-0.8b | spatial_embspatial | 36.2 / 3.5 | 2996 / 3081 |
| qwen3.5-0.8b | 合计 | 2008.4 / 185.2 | |
| qwen3.5-2b | refexp_box | 59.3 / 6.6 | 4672 / 4672 |
| qwen3.5-2b | refexp_poly | 55.1 / 9.9 | 4342 / 4234 |
| qwen3.5-2b | grid16 | 1253.5 / 94.8 | 101631 / 101631 |
| qwen3.5-2b | count | 2.5 / 0.4 | 104 / 104 |
| qwen3.5-2b | ovd | 444.0 / 47.7 | 36781 / 36573 |
| qwen3.5-2b | rel_depth_nyu | 79.1 / 6.2 | 6400 / 6400 |
| qwen3.5-2b | rel_depth_blink | 9.5 / 0.9 | 768 / 768 |
| qwen3.5-2b | spatial_cvbench | 39.0 / 3.6 | 3145 / 3118 |
| qwen3.5-2b | spatial_embspatial | 35.6 / 3.7 | 2858 / 2813 |
| qwen3.5-2b | 合计 | 1977.7 / 173.9 | |
| gemma-4-e2b-it | refexp_box | 80.8 / 17.4 | 4395 / 4379 |
| gemma-4-e2b-it | refexp_poly | 159.9 / 58.7 | 8448 / 8815 |
| gemma-4-e2b-it | grid16 | 452.1 / 35.7 | 24530 / 24470 |
| gemma-4-e2b-it | count | 2.9 / 0.7 | 103 / 103 |
| gemma-4-e2b-it | ovd | 291.0 / 47.5 | 15688 / 15900 |
| gemma-4-e2b-it | rel_depth_nyu | 117.1 / 9.1 | 6400 / 6400 |
| gemma-4-e2b-it | rel_depth_blink | 14.1 / 1.4 | 768 / 768 |
| gemma-4-e2b-it | spatial_cvbench | 51.7 / 5.4 | 2823 / 2823 |
| gemma-4-e2b-it | spatial_embspatial | 57.5 / 5.4 | 3172 / 3172 |
| gemma-4-e2b-it | 合计 | 1227.0 / 181.5 | |
| internvl3.5-2b-hf | refexp_box | 31.0 / 7.5 | 2836 / 2838 |
| internvl3.5-2b-hf | refexp_poly | 289.6 / 94.8 | 25482 / 26523 |
| internvl3.5-2b-hf | grid16 | 262.5 / 22.0 | 25218 / 25299 |
| internvl3.5-2b-hf | count | 15.1 / 2.8 | 1371 / 1425 |
| internvl3.5-2b-hf | ovd | 161.5 / 46.6 | 15941 / 15973 |
| internvl3.5-2b-hf | rel_depth_nyu | 66.3 / 8.2 | 6400 / 6400 |
| internvl3.5-2b-hf | rel_depth_blink | 7.9 / 2.1 | 768 / 768 |
| internvl3.5-2b-hf | spatial_cvbench | 32.0 / 4.9 | 3057 / 3084 |
| internvl3.5-2b-hf | spatial_embspatial | 24.1 / 2.9 | 2217 / 2130 |
| internvl3.5-2b-hf | 合计 | 890.0 / 191.7 | |
| qwen3.5-4b | refexp_box | 79.5 / 11.1 | 4800 / 4799 |
| qwen3.5-4b | refexp_poly | 68.6 / 10.4 | 4080 / 4113 |
| qwen3.5-4b | grid16 | 442.2 / 34.2 | 27234 / 27200 |
| qwen3.5-4b | count | 3.0 / 0.6 | 105 / 105 |
| qwen3.5-4b | ovd | 562.1 / 67.1 | 33919 / 33913 |
| qwen3.5-4b | rel_depth_nyu | 108.2 / 8.3 | 6400 / 6400 |
| qwen3.5-4b | rel_depth_blink | 12.8 / 1.2 | 768 / 768 |
| qwen3.5-4b | spatial_cvbench | 52.7 / 5.0 | 3161 / 3180 |
| qwen3.5-4b | spatial_embspatial | 40.8 / 5.1 | 2418 / 2208 |
| qwen3.5-4b | 合计 | 1369.8 / 143.0 | |
| gemma-4-e4b-it | refexp_box | 102.0 / 19.4 | 4462 / 4630 |
| gemma-4-e4b-it | refexp_poly | 165.3 / 24.8 | 7151 / 7150 |
| gemma-4-e4b-it | grid16 | 663.3 / 50.8 | 27200 / 27200 |
| gemma-4-e4b-it | count | 3.5 / 0.8 | 104 / 104 |
| gemma-4-e4b-it | ovd | 451.4 / 79.9 | 19363 / 19336 |
| gemma-4-e4b-it | rel_depth_nyu | 143.7 / 11.1 | 6400 / 6400 |
| gemma-4-e4b-it | rel_depth_blink | 17.2 / 1.7 | 768 / 768 |
| gemma-4-e4b-it | spatial_cvbench | 59.2 / 6.7 | 2630 / 2608 |
| gemma-4-e4b-it | spatial_embspatial | 68.8 / 6.9 | 3062 / 3061 |
| gemma-4-e4b-it | 合计 | 1674.4 / 202.1 | |
| internvl3.5-4b-hf | refexp_box | 38.6 / 5.9 | 2839 / 2842 |
| internvl3.5-4b-hf | refexp_poly | 175.7 / 117.5 | 11766 / 13509 |
| internvl3.5-4b-hf | grid16 | 363.7 / 38.3 | 27200 / 27200 |
| internvl3.5-4b-hf | count | 5.5 / 1.2 | 372 / 373 |
| internvl3.5-4b-hf | ovd | 507.5 / 72.0 | 38874 / 39290 |
| internvl3.5-4b-hf | rel_depth_nyu | 84.0 / 9.2 | 6400 / 6400 |
| internvl3.5-4b-hf | rel_depth_blink | 10.3 / 2.3 | 768 / 768 |
| internvl3.5-4b-hf | spatial_cvbench | 35.5 / 5.9 | 2692 / 2693 |
| internvl3.5-4b-hf | spatial_embspatial | 23.0 / 3.7 | 1728 / 1721 |
| internvl3.5-4b-hf | 合计 | 1243.8 / 256.0 | |
| qwen3.5-9b | refexp_box | 80.4 / 10.6 | 4694 / 4695 |
| qwen3.5-9b | refexp_poly | 84.7 / 34.2 | 4999 / 4972 |
| qwen3.5-9b | grid16 | 455.3 / 36.4 | 27442 / 27425 |
| qwen3.5-9b | count | 3.2 / 0.8 | 106 / 106 |
| qwen3.5-9b | ovd | 579.5 / 62.1 | 34644 / 34616 |
| qwen3.5-9b | rel_depth_nyu | 105.8 / 8.5 | 6400 / 6400 |
| qwen3.5-9b | rel_depth_blink | 12.7 / 1.3 | 768 / 768 |
| qwen3.5-9b | spatial_cvbench | 52.4 / 5.6 | 3167 / 3167 |
| qwen3.5-9b | spatial_embspatial | 39.2 / 5.0 | 2345 / 2285 |
| qwen3.5-9b | 合计 | 1413.3 / 164.4 | |
