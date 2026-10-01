#### 验收汇总（三种一致口径；判据：一致率 ≥ 0.98 且配对差 95% CI 含 0；计数用默认的启发式问句）

| 模型 | 任务 | n | (a) 逐位 一致率 / 通过 | (b) IoU≥0.9 一致率 / 通过 | (c) 判定结果 一致率 / 通过 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 100 | 0.3000 / 否 | 0.9700 / 否 | 0.9800 / 是 | acc@0.5 | 0.8000 | 0.8200 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.2800 |
| qwen3.5-0.8b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 |
| qwen3.5-0.8b | grid16 | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-0.8b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9592 |
| qwen3.5-0.8b | ovd | 100 | 0.5900 / 否 | 0.8100 / 否 | — / — | — | — | — | — | — | — | 0.0000 |
| qwen3.5-0.8b | rel_depth_nyu | 100 | 0.7800 / 否 | 0.7800 / 否 | 0.7800 / 否 | accuracy | 0.6300 | 0.5700 | 0.0600 | [-0.0300, 0.1500] | 是 | 0.1400 |
| qwen3.5-0.8b | rel_depth_blink | 12 | 0.7500 / 否 | 0.7500 / 否 | 0.7500 / 否 | accuracy | 0.7500 | 0.7500 | 0.0000 | [-0.2500, 0.2500] | 是 | 0.1667 |
| qwen3.5-0.8b | spatial_cvbench | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.7200 | 0.7000 | 0.0200 | [-0.0400, 0.1000] | 是 | 0.3200 |
| qwen3.5-0.8b | spatial_embspatial | 50 | 0.8400 / 否 | 0.8400 / 否 | 0.8400 / 否 | accuracy | 0.3600 | 0.4000 | -0.0400 | [-0.1200, 0.0400] | 是 | 0.4000 |
| qwen3.5-2b | refexp_box | 100 | 0.2900 / 否 | 0.9800 / 是 | 1.0000 / 是 | acc@0.5 | 0.8600 | 0.8600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2800 |
| qwen3.5-2b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 |
| qwen3.5-2b | grid16 | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 |
| qwen3.5-2b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7143 | 0.7143 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-2b | ovd | 100 | 0.3500 / 否 | 0.7200 / 否 | — / — | — | — | — | — | — | — | 0.0200 |
| qwen3.5-2b | rel_depth_nyu | 100 | 0.9100 / 否 | 0.9100 / 否 | 0.9100 / 否 | accuracy | 0.7500 | 0.6800 | 0.0700 | [0.0100, 0.1300] | 否 | 0.2000 |
| qwen3.5-2b | rel_depth_blink | 12 | 0.8333 / 否 | 0.8333 / 否 | 0.8333 / 否 | accuracy | 0.6667 | 0.7500 | -0.0833 | [-0.2500, 0.0000] | 是 | 0.0000 |
| qwen3.5-2b | spatial_cvbench | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.8200 | 0.8200 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.5400 |
| qwen3.5-2b | spatial_embspatial | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.5400 | 0.5000 | 0.0400 | [0.0000, 0.1000] | 是 | 0.3800 |
| gemma-4-e2b-it | refexp_box | 100 | 0.9100 / 否 | 0.9800 / 是 | 1.0000 / 是 | acc@0.5 | 0.0800 | 0.0800 | 0.0000 | [0.0000, 0.0000] | 是 | 0.7000 |
| gemma-4-e2b-it | refexp_poly | 100 | 0.6900 / 否 | 0.6900 / 否 | 0.6900 / 否 | cIoU | 0.2413 | 0.2340 | 0.0073 | [-0.0016, 0.0163] | 是 | 0.5400 |
| gemma-4-e2b-it | grid16 | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9500 |
| gemma-4-e2b-it | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.3673 | 0.3673 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 |
| gemma-4-e2b-it | ovd | 100 | 0.4500 / 否 | 0.8000 / 否 | — / — | — | — | — | — | — | — | 0.4100 |
| gemma-4-e2b-it | rel_depth_nyu | 100 | 0.8400 / 否 | 0.8400 / 否 | 0.8400 / 否 | accuracy | 0.6400 | 0.6200 | 0.0200 | [-0.0500, 0.0900] | 是 | 0.0600 |
| gemma-4-e2b-it | rel_depth_blink | 12 | 0.9167 / 否 | 0.9167 / 否 | 0.9167 / 否 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.0833 |
| gemma-4-e2b-it | spatial_cvbench | 50 | 0.8600 / 否 | 0.8600 / 否 | 0.8600 / 否 | accuracy | 0.7200 | 0.7400 | -0.0200 | [-0.1200, 0.0600] | 是 | 0.4200 |
| gemma-4-e2b-it | spatial_embspatial | 50 | 0.7600 / 否 | 0.7600 / 否 | 0.7600 / 否 | accuracy | 0.1800 | 0.2000 | -0.0200 | [-0.1200, 0.0800] | 是 | 0.1400 |
| internvl3.5-2b-hf | refexp_box | 100 | 0.0900 / 否 | 0.9100 / 否 | 0.9800 / 是 | acc@0.5 | 0.7100 | 0.7300 | -0.0200 | [-0.0500, 0.0000] | 是 | 0.0900 |
| internvl3.5-2b-hf | refexp_poly | 100 | 0.4300 / 否 | 0.4300 / 否 | 0.4300 / 否 | cIoU | 0.1031 | 0.0918 | 0.0113 | [-0.0111, 0.0344] | 是 | 0.1800 |
| internvl3.5-2b-hf | grid16 | 100 | 0.9900 / 是 | 0.9900 / 是 | 0.9900 / 是 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.8700 |
| internvl3.5-2b-hf | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.4694 | 0.4694 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6939 |
| internvl3.5-2b-hf | ovd | 100 | 1.0000 / 是 | 1.0000 / 是 | — / — | — | — | — | — | — | — | 0.3400 |
| internvl3.5-2b-hf | rel_depth_nyu | 100 | 0.8900 / 否 | 0.8900 / 否 | 0.8900 / 否 | accuracy | 0.5500 | 0.6000 | -0.0500 | [-0.1100, 0.0200] | 是 | 0.3100 |
| internvl3.5-2b-hf | rel_depth_blink | 12 | 0.6667 / 否 | 0.6667 / 否 | 0.6667 / 否 | accuracy | 0.5833 | 0.5833 | 0.0000 | [-0.3333, 0.3333] | 是 | 0.1667 |
| internvl3.5-2b-hf | spatial_cvbench | 50 | 0.9200 / 否 | 0.9200 / 否 | 0.9200 / 否 | accuracy | 0.7600 | 0.8000 | -0.0400 | [-0.1200, 0.0400] | 是 | 0.3800 |
| internvl3.5-2b-hf | spatial_embspatial | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.3400 | 0.3400 | 0.0000 | [-0.0800, 0.0800] | 是 | 0.4400 |
| qwen3.5-4b | refexp_box | 100 | 0.3300 / 否 | 0.9800 / 是 | 1.0000 / 是 | acc@0.5 | 0.8700 | 0.8700 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3000 |
| qwen3.5-4b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2700 |
| qwen3.5-4b | grid16 | 100 | 0.9900 / 是 | 0.9900 / 是 | 0.9900 / 是 | gIoU | 0.0111 | 0.0111 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9900 |
| qwen3.5-4b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-4b | ovd | 100 | 0.1300 / 否 | 0.5500 / 否 | — / — | — | — | — | — | — | — | 0.0100 |
| qwen3.5-4b | rel_depth_nyu | 100 | 0.8700 / 否 | 0.8700 / 否 | 0.8700 / 否 | accuracy | 0.7100 | 0.7700 | -0.0600 | [-0.1200, 0.0000] | 是 | 0.2900 |
| qwen3.5-4b | rel_depth_blink | 12 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] | 是 | 0.1667 |
| qwen3.5-4b | spatial_cvbench | 50 | 0.9600 / 否 | 0.9600 / 否 | 0.9600 / 否 | accuracy | 0.9000 | 0.9000 | 0.0000 | [-0.0600, 0.0600] | 是 | 0.5200 |
| qwen3.5-4b | spatial_embspatial | 50 | 0.8600 / 否 | 0.8600 / 否 | 0.8600 / 否 | accuracy | 0.3400 | 0.4200 | -0.0800 | [-0.1800, 0.0005] | 是 | 0.4000 |
| gemma-4-e4b-it | refexp_box | 100 | 0.9200 / 否 | 0.9900 / 是 | 1.0000 / 是 | acc@0.5 | 0.1300 | 0.1300 | 0.0000 | [0.0000, 0.0000] | 是 | 0.6900 |
| gemma-4-e4b-it | refexp_poly | 100 | 0.9800 / 是 | 0.9800 / 是 | 0.9800 / 是 | cIoU | 0.0381 | 0.0336 | 0.0045 | [-0.0000, 0.0122] | 是 | 0.1600 |
| gemma-4-e4b-it | grid16 | 100 | 0.9800 / 是 | 0.9800 / 是 | 0.9800 / 是 | gIoU | 0.0112 | 0.0122 | -0.0010 | [-0.0038, 0.0010] | 是 | 0.9700 |
| gemma-4-e4b-it | count | 49 | 0.9592 / 否 | 0.9592 / 否 | 0.9592 / 否 | accuracy | 0.5102 | 0.4898 | 0.0204 | [0.0000, 0.0612] | 是 | 0.9592 |
| gemma-4-e4b-it | ovd | 100 | 0.1700 / 否 | 0.7500 / 否 | — / — | — | — | — | — | — | — | 0.1700 |
| gemma-4-e4b-it | rel_depth_nyu | 100 | 0.9400 / 否 | 0.9400 / 否 | 0.9400 / 否 | accuracy | 0.7100 | 0.7000 | 0.0100 | [-0.0300, 0.0500] | 是 | 0.2200 |
| gemma-4-e4b-it | rel_depth_blink | 12 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2500 |
| gemma-4-e4b-it | spatial_cvbench | 50 | 0.9400 / 否 | 0.9400 / 否 | 0.9400 / 否 | accuracy | 0.7000 | 0.6600 | 0.0400 | [0.0000, 0.1000] | 是 | 0.5600 |
| gemma-4-e4b-it | spatial_embspatial | 50 | 0.8600 / 否 | 0.8600 / 否 | 0.8600 / 否 | accuracy | 0.2200 | 0.2400 | -0.0200 | [-0.0800, 0.0400] | 是 | 0.2800 |
| internvl3.5-4b-hf | refexp_box | 100 | 0.4800 / 否 | 0.9500 / 否 | 0.9900 / 是 | acc@0.5 | 0.8700 | 0.8600 | 0.0100 | [0.0000, 0.0400] | 是 | 0.4800 |
| internvl3.5-4b-hf | refexp_poly | 100 | 0.3700 / 否 | 0.3700 / 否 | 0.3700 / 否 | cIoU | 0.1630 | 0.1639 | -0.0009 | [-0.0311, 0.0276] | 是 | 0.2500 |
| internvl3.5-4b-hf | grid16 | 100 | 0.9700 / 否 | 0.9700 / 否 | 0.9700 / 否 | gIoU | 0.0044 | 0.0040 | 0.0004 | [0.0000, 0.0012] | 是 | 0.9700 |
| internvl3.5-4b-hf | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.6735 | 0.6735 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 |
| internvl3.5-4b-hf | ovd | 100 | 0.1400 / 否 | 0.3100 / 否 | — / — | — | — | — | — | — | — | 0.0200 |
| internvl3.5-4b-hf | rel_depth_nyu | 100 | 0.8700 / 否 | 0.8700 / 否 | 0.8700 / 否 | accuracy | 0.7100 | 0.7200 | -0.0100 | [-0.0800, 0.0600] | 是 | 0.3400 |
| internvl3.5-4b-hf | rel_depth_blink | 12 | 0.8333 / 否 | 0.8333 / 否 | 0.8333 / 否 | accuracy | 0.6667 | 0.5000 | 0.1667 | [0.0000, 0.4167] | 是 | 0.1667 |
| internvl3.5-4b-hf | spatial_cvbench | 50 | 0.9200 / 否 | 0.9200 / 否 | 0.9200 / 否 | accuracy | 0.8000 | 0.7800 | 0.0200 | [-0.0400, 0.0800] | 是 | 0.5000 |
| internvl3.5-4b-hf | spatial_embspatial | 50 | 0.9600 / 否 | 0.9600 / 否 | 0.9600 / 否 | accuracy | 0.5800 | 0.6200 | -0.0400 | [-0.1000, 0.0000] | 是 | 0.8200 |
| qwen3.5-9b | refexp_box | 100 | 0.3500 / 否 | 0.9900 / 是 | 1.0000 / 是 | acc@0.5 | 0.8900 | 0.8900 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3300 |
| qwen3.5-9b | refexp_poly | 100 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.3800 |
| qwen3.5-9b | grid16 | 100 | 0.9500 / 否 | 0.9500 / 否 | 0.9500 / 否 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9400 |
| qwen3.5-9b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-9b | ovd | 100 | 0.1000 / 否 | 0.4900 / 否 | — / — | — | — | — | — | — | — | 0.0000 |
| qwen3.5-9b | rel_depth_nyu | 100 | 0.9700 / 否 | 0.9700 / 否 | 0.9700 / 否 | accuracy | 0.7100 | 0.7100 | 0.0000 | [-0.0300, 0.0300] | 是 | 0.5300 |
| qwen3.5-9b | rel_depth_blink | 12 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] | 是 | 0.2500 |
| qwen3.5-9b | spatial_cvbench | 50 | 0.9800 / 是 | 0.9800 / 是 | 0.9800 / 是 | accuracy | 0.8600 | 0.8600 | 0.0000 | [0.0000, 0.0000] | 是 | 0.5800 |
| qwen3.5-9b | spatial_embspatial | 50 | 0.9000 / 否 | 0.9000 / 否 | 0.9000 / 否 | accuracy | 0.6000 | 0.5800 | 0.0200 | [-0.0400, 0.1000] | 是 | 0.5400 |

#### 每种口径的总判定

| 模型 | (a) 逐位 all_pass（不判的任务） | (b) IoU≥0.9 all_pass（不判的任务） | (c) 判定结果 all_pass（不判的任务） | (a) 逐位 不通过的任务 | (b) IoU≥0.9 不通过的任务 | (c) 判定结果 不通过的任务 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 否（—） | 否（—） | 否（ovd） | refexp_box, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_box, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| qwen3.5-2b | 否（—） | 否（—） | 否（ovd） | refexp_box, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| gemma-4-e2b-it | 否（—） | 否（—） | 否（ovd） | refexp_box, refexp_poly, count, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, count, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| internvl3.5-2b-hf | 否（—） | 否（—） | 否（ovd） | refexp_box, refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_box, refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, count, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| qwen3.5-4b | 否（—） | 否（—） | 否（ovd） | refexp_box, ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | rel_depth_nyu, spatial_cvbench, spatial_embspatial |
| gemma-4-e4b-it | 否（—） | 否（—） | 否（ovd） | refexp_box, count, ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | count, ovd, rel_depth_nyu, spatial_cvbench, spatial_embspatial | count, rel_depth_nyu, spatial_cvbench, spatial_embspatial |
| internvl3.5-4b-hf | 否（—） | 否（—） | 否（ovd） | refexp_box, refexp_poly, grid16, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_box, refexp_poly, grid16, ovd, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | refexp_poly, grid16, rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial |
| qwen3.5-9b | 否（—） | 否（—） | 否（ovd） | refexp_box, grid16, ovd, rel_depth_nyu, spatial_embspatial | grid16, ovd, rel_depth_nyu, spatial_embspatial | grid16, rel_depth_nyu, spatial_embspatial |

#### 计数：CountBenchQA 人工问句（--task count --count-questions countbenchqa）

| 模型 | 任务 | n | (a) 逐位 一致率 / 通过 | (b) IoU≥0.9 一致率 / 通过 | (c) 判定结果 一致率 / 通过 | 指标 | 逐条 | 批量 | 差（逐条−批量） | 配对差 95% CI | CI 含 0 | raw_text 逐字相同 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| qwen3.5-2b | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.8367 | 0.8163 | 0.0204 | [0.0000, 0.0612] | 是 | 0.9796 |
| gemma-4-e2b-it | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.5306 | 0.5306 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| internvl3.5-2b-hf | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9592 |
| qwen3.5-4b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.8571 | 0.8571 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| gemma-4-e4b-it | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |
| internvl3.5-4b-hf | count | 49 | 0.9796 / 否 | 0.9796 / 否 | 0.9796 / 否 | accuracy | 0.7347 | 0.7347 | 0.0000 | [0.0000, 0.0000] | 是 | 0.9796 |
| qwen3.5-9b | count | 49 | 1.0000 / 是 | 1.0000 / 是 | 1.0000 / 是 | accuracy | 0.9184 | 0.9184 | 0.0000 | [0.0000, 0.0000] | 是 | 1.0000 |

| 模型 | (a) 逐位 all_pass | (b) IoU≥0.9 all_pass | (c) 判定结果 all_pass |
|---|---|---|---|
| qwen3.5-0.8b | 是 | 是 | 是 |
| qwen3.5-2b | 否 | 否 | 否 |
| gemma-4-e2b-it | 是 | 是 | 是 |
| internvl3.5-2b-hf | 否 | 否 | 否 |
| qwen3.5-4b | 是 | 是 | 是 |
| gemma-4-e4b-it | 是 | 是 | 是 |
| internvl3.5-4b-hf | 否 | 否 | 否 |
| qwen3.5-9b | 是 | 是 | 是 |

#### 截断率与解析失败（逐条 bs=1 / 批量；截断 = stopped=False；启发式问句）

| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 256 | 0.01 / 0.01 | 2/100 {"multiple_boxes": 1, "truncated": 1} | 2/100 {"multiple_boxes": 1, "truncated": 1} | 49.2 / 49.2 |
| qwen3.5-0.8b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 66.3 / 67.5 |
| qwen3.5-0.8b | grid16 | 1024 | 1.00 / 1.00 | 100/100 {"truncated": 100} | 100/100 {"truncated": 100} | 1024.0 / 1024.0 |
| qwen3.5-0.8b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.3 / 2.6 |
| qwen3.5-0.8b | ovd | 2048 | 0.61 / 0.65 | 61/100 {"truncated": 61} | 65/100 {"truncated": 65} | 1361.8 / 1413.2 |
| qwen3.5-0.8b | rel_depth_nyu | 1024 | 0.02 / 0.02 | 2/100 {"truncated": 2} | 4/100 {"multiple_choices": 2, "truncated": 2} | 262.3 / 251.1 |
| qwen3.5-0.8b | rel_depth_blink | 1024 | 0.08 / 0.00 | 2/12 {"multiple_choices": 1, "truncated": 1} | 2/12 {"multiple_choices": 2} | 346.1 / 329.0 |
| qwen3.5-0.8b | spatial_cvbench | 1024 | 0.00 / 0.04 | 0/50 {} | 2/50 {"truncated": 2} | 136.2 / 163.2 |
| qwen3.5-0.8b | spatial_embspatial | 1024 | 0.00 / 0.00 | 17/50 {"multiple_choices": 17} | 18/50 {"multiple_choices": 18} | 133.8 / 131.2 |
| qwen3.5-2b | refexp_box | 256 | 0.00 / 0.00 | 1/100 {"multiple_boxes": 1} | 1/100 {"multiple_boxes": 1} | 47.4 / 47.4 |
| qwen3.5-2b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 43.4 / 42.3 |
| qwen3.5-2b | grid16 | 1024 | 0.99 / 0.99 | 100/100 {"truncated": 99, "wrong_row_count": 1} | 100/100 {"truncated": 99, "wrong_row_count": 1} | 1016.3 / 1016.3 |
| qwen3.5-2b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| qwen3.5-2b | ovd | 2048 | 0.38 / 0.37 | 38/100 {"truncated": 38} | 37/100 {"truncated": 37} | 965.2 / 957.6 |
| qwen3.5-2b | rel_depth_nyu | 1024 | 0.00 / 0.02 | 0/100 {} | 2/100 {"truncated": 2} | 183.9 / 200.1 |
| qwen3.5-2b | rel_depth_blink | 1024 | 0.00 / 0.00 | 2/12 {"multiple_choices": 2} | 1/12 {"multiple_choices": 1} | 255.3 / 253.7 |
| qwen3.5-2b | spatial_cvbench | 1024 | 0.02 / 0.04 | 4/50 {"multiple_choices": 3, "truncated": 1} | 2/50 {"truncated": 2} | 182.0 / 182.6 |
| qwen3.5-2b | spatial_embspatial | 1024 | 0.00 / 0.00 | 20/50 {"multiple_choices": 20} | 19/50 {"multiple_choices": 19} | 123.0 / 124.7 |
| gemma-4-e2b-it | refexp_box | 256 | 0.03 / 0.04 | 88/100 {"multiple_boxes": 3, "no_box": 82, "truncated": 3} | 87/100 {"multiple_boxes": 3, "no_box": 80, "truncated": 4} | 55.6 / 55.1 |
| gemma-4-e2b-it | refexp_poly | 1024 | 0.00 / 0.00 | 2/100 {"malformed": 1, "too_few_points": 1} | 2/100 {"malformed": 1, "too_few_points": 1} | 84.5 / 88.2 |
| gemma-4-e2b-it | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"empty": 3, "wrong_row_count": 97} | 100/100 {"empty": 3, "wrong_row_count": 97} | 245.3 / 244.7 |
| gemma-4-e2b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| gemma-4-e2b-it | ovd | 2048 | 0.00 / 0.00 | 35/100 {"no_box": 35} | 35/100 {"no_box": 35} | 158.2 / 159.6 |
| gemma-4-e2b-it | rel_depth_nyu | 1024 | 0.00 / 0.00 | 1/100 {"multiple_choices": 1} | 6/100 {"multiple_choices": 6} | 281.4 / 280.0 |
| gemma-4-e2b-it | rel_depth_blink | 1024 | 0.00 / 0.00 | 2/12 {"multiple_choices": 2} | 1/12 {"multiple_choices": 1} | 288.8 / 270.5 |
| gemma-4-e2b-it | spatial_cvbench | 1024 | 0.00 / 0.00 | 8/50 {"multiple_choices": 7, "no_choice": 1} | 4/50 {"multiple_choices": 3, "no_choice": 1} | 190.8 / 196.5 |
| gemma-4-e2b-it | spatial_embspatial | 1024 | 0.00 / 0.02 | 36/50 {"multiple_choices": 35, "no_choice": 1} | 35/50 {"multiple_choices": 33, "no_choice": 1, "truncated": 1} | 337.6 / 346.4 |
| internvl3.5-2b-hf | refexp_box | 256 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 |
| internvl3.5-2b-hf | refexp_poly | 1024 | 0.02 / 0.02 | 22/100 {"ambiguous": 3, "malformed": 2, "too_few_points": 15, "truncated": 2} | 21/100 {"ambiguous": 1, "malformed": 4, "too_few_points": 14, "truncated": 2} | 254.8 / 265.2 |
| internvl3.5-2b-hf | grid16 | 1024 | 0.00 / 0.00 | 100/100 {"wrong_row_count": 100} | 99/100 {"wrong_row_count": 99} | 252.2 / 253.0 |
| internvl3.5-2b-hf | count | 256 | 0.08 / 0.14 | 13/49 {"multiple_numbers": 9, "thinking_unclosed": 2, "truncated": 2} | 14/49 {"multiple_numbers": 7, "thinking_unclosed": 3, "truncated": 4} | 61.6 / 68.6 |
| internvl3.5-2b-hf | ovd | 2048 | 0.02 / 0.03 | 100/100 {"no_box": 98, "truncated": 2} | 100/100 {"no_box": 97, "truncated": 3} | 198.2 / 214.0 |
| internvl3.5-2b-hf | rel_depth_nyu | 1024 | 0.00 / 0.00 | 2/100 {"multiple_choices": 2} | 0/100 {} | 164.8 / 167.6 |
| internvl3.5-2b-hf | rel_depth_blink | 1024 | 0.00 / 0.00 | 1/12 {"multiple_choices": 1} | 1/12 {"multiple_choices": 1} | 198.1 / 204.1 |
| internvl3.5-2b-hf | spatial_cvbench | 1024 | 0.00 / 0.00 | 2/50 {"multiple_choices": 2} | 1/50 {"multiple_choices": 1} | 144.8 / 144.3 |
| internvl3.5-2b-hf | spatial_embspatial | 1024 | 0.00 / 0.00 | 19/50 {"multiple_choices": 19} | 19/50 {"multiple_choices": 19} | 132.0 / 122.9 |
| qwen3.5-4b | refexp_box | 256 | 0.00 / 0.00 | 3/100 {"multiple_boxes": 3} | 3/100 {"multiple_boxes": 3} | 49.1 / 49.1 |
| qwen3.5-4b | refexp_poly | 1024 | 0.00 / 0.00 | 100/100 {"malformed": 100} | 100/100 {"malformed": 100} | 40.8 / 41.1 |
| qwen3.5-4b | grid16 | 1024 | 0.00 / 0.00 | 1/100 {"wrong_row_count": 1} | 0/100 {} | 272.3 / 272.0 |
| qwen3.5-4b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| qwen3.5-4b | ovd | 2048 | 0.14 / 0.13 | 14/100 {"truncated": 14} | 13/100 {"truncated": 13} | 623.4 / 613.8 |
| qwen3.5-4b | rel_depth_nyu | 1024 | 0.02 / 0.01 | 5/100 {"multiple_choices": 3, "truncated": 2} | 3/100 {"multiple_choices": 2, "truncated": 1} | 225.3 / 210.2 |
| qwen3.5-4b | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 191.5 / 184.3 |
| qwen3.5-4b | spatial_cvbench | 1024 | 0.00 / 0.00 | 0/50 {} | 1/50 {"multiple_choices": 1} | 145.1 / 163.6 |
| qwen3.5-4b | spatial_embspatial | 1024 | 0.02 / 0.04 | 30/50 {"multiple_choices": 28, "no_choice": 1, "truncated": 1} | 27/50 {"multiple_choices": 25, "truncated": 2} | 194.1 / 194.2 |
| gemma-4-e4b-it | refexp_box | 256 | 0.02 / 0.03 | 82/100 {"multiple_boxes": 3, "no_box": 77, "truncated": 2} | 83/100 {"multiple_boxes": 3, "no_box": 77, "truncated": 3} | 49.5 / 52.6 |
| gemma-4-e4b-it | refexp_poly | 1024 | 0.00 / 0.00 | 93/100 {"malformed": 93} | 93/100 {"malformed": 93} | 71.5 / 71.5 |
| gemma-4-e4b-it | grid16 | 1024 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 272.0 / 272.0 |
| gemma-4-e4b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| gemma-4-e4b-it | ovd | 2048 | 0.00 / 0.00 | 1/100 {"no_box": 1} | 1/100 {"no_box": 1} | 197.5 / 199.5 |
| gemma-4-e4b-it | rel_depth_nyu | 1024 | 0.00 / 0.00 | 3/100 {"multiple_choices": 3} | 0/100 {} | 265.2 / 258.5 |
| gemma-4-e4b-it | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 227.8 / 271.0 |
| gemma-4-e4b-it | spatial_cvbench | 1024 | 0.00 / 0.00 | 8/50 {"multiple_choices": 7, "no_choice": 1} | 10/50 {"multiple_choices": 9, "no_choice": 1} | 132.1 / 134.9 |
| gemma-4-e4b-it | spatial_embspatial | 1024 | 0.02 / 0.02 | 33/50 {"multiple_choices": 27, "no_choice": 5, "truncated": 1} | 32/50 {"multiple_choices": 24, "no_choice": 7, "truncated": 1} | 263.1 / 254.1 |
| internvl3.5-4b-hf | refexp_box | 256 | 0.00 / 0.00 | 0/100 {} | 0/100 {} | 28.4 / 28.4 |
| internvl3.5-4b-hf | refexp_poly | 1024 | 0.04 / 0.06 | 5/100 {"out_of_range": 1, "truncated": 4} | 6/100 {"truncated": 6} | 117.7 / 135.1 |
| internvl3.5-4b-hf | grid16 | 1024 | 0.00 / 0.00 | 2/100 {"multiple_blocks": 2} | 1/100 {"multiple_blocks": 1} | 272.0 / 272.0 |
| internvl3.5-4b-hf | count | 256 | 0.00 / 0.00 | 1/49 {"multiple_numbers": 1} | 1/49 {"multiple_numbers": 1} | 7.6 / 7.6 |
| internvl3.5-4b-hf | ovd | 2048 | 0.19 / 0.15 | 19/100 {"truncated": 19} | 15/100 {"truncated": 15} | 761.5 / 736.6 |
| internvl3.5-4b-hf | rel_depth_nyu | 1024 | 0.00 / 0.00 | 4/100 {"multiple_choices": 4} | 2/100 {"multiple_choices": 2} | 150.3 / 151.9 |
| internvl3.5-4b-hf | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 232.2 / 249.9 |
| internvl3.5-4b-hf | spatial_cvbench | 1024 | 0.00 / 0.00 | 2/50 {"multiple_choices": 2} | 1/50 {"multiple_choices": 1} | 126.3 / 123.5 |
| internvl3.5-4b-hf | spatial_embspatial | 1024 | 0.00 / 0.00 | 7/50 {"multiple_choices": 7} | 6/50 {"multiple_choices": 6} | 43.7 / 40.8 |
| qwen3.5-9b | refexp_box | 256 | 0.00 / 0.00 | 2/100 {"multiple_boxes": 2} | 2/100 {"multiple_boxes": 2} | 47.7 / 47.7 |
| qwen3.5-9b | refexp_poly | 1024 | 0.01 / 0.01 | 100/100 {"malformed": 99, "truncated": 1} | 100/100 {"malformed": 99, "truncated": 1} | 50.0 / 49.7 |
| qwen3.5-9b | grid16 | 1024 | 0.00 / 0.00 | 14/100 {"wrong_row_count": 14} | 13/100 {"wrong_row_count": 13} | 274.4 / 274.2 |
| qwen3.5-9b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| qwen3.5-9b | ovd | 2048 | 0.16 / 0.13 | 16/100 {"truncated": 16} | 13/100 {"truncated": 13} | 688.1 / 642.7 |
| qwen3.5-9b | rel_depth_nyu | 1024 | 0.00 / 0.01 | 1/100 {"multiple_choices": 1} | 2/100 {"multiple_choices": 1, "truncated": 1} | 176.1 / 185.6 |
| qwen3.5-9b | rel_depth_blink | 1024 | 0.00 / 0.00 | 0/12 {} | 0/12 {} | 205.1 / 191.0 |
| qwen3.5-9b | spatial_cvbench | 1024 | 0.00 / 0.00 | 1/50 {"multiple_choices": 1} | 2/50 {"multiple_choices": 2} | 148.1 / 156.2 |
| qwen3.5-9b | spatial_embspatial | 1024 | 0.02 / 0.02 | 16/50 {"multiple_choices": 15, "truncated": 1} | 16/50 {"multiple_choices": 15, "truncated": 1} | 155.1 / 152.8 |

#### 诊断（failure_detail：wrong_shape、refusal；关思考时自发的思考标记；思考未闭合；启发式问句）

| 模型 | 任务 | failure_detail bs1 | failure_detail 批量 | 自发思考标记 bs1 / 批量 | 思考未闭合 bs1 / 批量 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | refexp_poly | {"wrong_shape:bbox_2d": 98, "wrong_shape:point_2d": 1} | {"wrong_shape:bbox_2d": 98, "wrong_shape:point_2d": 1} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_nyu | {"refusal": 1} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_blink | {"refusal": 1} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-0.8b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | refexp_poly | {"wrong_shape:bbox_2d": 15, "wrong_shape:point_2d": 85} | {"wrong_shape:bbox_2d": 14, "wrong_shape:point_2d": 86} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | rel_depth_nyu | {} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | refexp_box | {"refusal": 60} | {"refusal": 61} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | refexp_poly | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | ovd | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | spatial_cvbench | {"refusal": 1} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | spatial_embspatial | {"refusal": 1} | {"refusal": 2} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | refexp_poly | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | count | {} | {} | 3 / 4 | 2 / 3 |
| internvl3.5-2b-hf | ovd | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | refexp_poly | {"wrong_shape:point_2d": 100} | {"wrong_shape:point_2d": 100} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | refexp_box | {"refusal": 63} | {"refusal": 63} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | refexp_poly | {"wrong_shape:box_2d": 93} | {"wrong_shape:box_2d": 93} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | ovd | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | spatial_cvbench | {"refusal": 1} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | refexp_poly | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | count | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | ovd | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_nyu | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | refexp_box | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | refexp_poly | {"wrong_shape:bbox_2d": 1, "wrong_shape:point_2d": 98} | {"wrong_shape:bbox_2d": 1, "wrong_shape:point_2d": 98} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | grid16 | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | ovd | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | rel_depth_nyu | {} | {"refusal": 1} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | rel_depth_blink | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | spatial_cvbench | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | spatial_embspatial | {} | {} | 0 / 0 | 0 / 0 |

#### 截断率与解析失败（逐条 bs=1 / 批量；截断 = stopped=False；CountBenchQA 问句（仅计数））

| 模型 | 任务 | max_new_tokens | 截断率 bs1 / 批量 | 解析失败 bs1 | 解析失败 批量 | 平均生成 token bs1 / 批量 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| qwen3.5-2b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| gemma-4-e2b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.1 / 2.1 |
| internvl3.5-2b-hf | count | 256 | 0.00 / 0.00 | 5/49 {"multiple_numbers": 5} | 4/49 {"multiple_numbers": 4} | 25.6 / 23.4 |
| qwen3.5-4b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| gemma-4-e4b-it | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |
| internvl3.5-4b-hf | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 9.1 / 9.1 |
| qwen3.5-9b | count | 256 | 0.00 / 0.00 | 0/49 {} | 0/49 {} | 2.2 / 2.2 |

#### 诊断（failure_detail：wrong_shape、refusal；关思考时自发的思考标记；思考未闭合；CountBenchQA 问句（仅计数））

| 模型 | 任务 | failure_detail bs1 | failure_detail 批量 | 自发思考标记 bs1 / 批量 | 思考未闭合 bs1 / 批量 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-2b | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e2b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-2b-hf | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-4b | count | {} | {} | 0 / 0 | 0 / 0 |
| gemma-4-e4b-it | count | {} | {} | 0 / 0 | 0 / 0 |
| internvl3.5-4b-hf | count | {} | {} | 0 / 0 | 0 / 0 |
| qwen3.5-9b | count | {} | {} | 0 / 0 | 0 / 0 |

#### 每遍的墙钟与生成 token（compute.json；墙钟只含 generate；启发式问句）

| 模型 | 任务 | 墙钟 s bs1 / 批量 | 生成 token bs1 / 批量 |
|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 59.9 / 9.8 | 4921 / 4920 |
| qwen3.5-0.8b | refexp_poly | 79.9 / 23.7 | 6631 / 6747 |
| qwen3.5-0.8b | grid16 | 1217.2 / 94.1 | 102400 / 102400 |
| qwen3.5-0.8b | count | 2.5 / 0.6 | 111 / 128 |
| qwen3.5-0.8b | ovd | 1626.6 / 186.7 | 136180 / 141319 |
| qwen3.5-0.8b | rel_depth_nyu | 311.8 / 44.2 | 26230 / 25109 |
| qwen3.5-0.8b | rel_depth_blink | 49.3 / 12.8 | 4153 / 3948 |
| qwen3.5-0.8b | spatial_cvbench | 81.2 / 22.1 | 6808 / 8160 |
| qwen3.5-0.8b | spatial_embspatial | 79.7 / 13.7 | 6691 / 6562 |
| qwen3.5-0.8b | 合计 | 3508.1 / 407.7 | |
| qwen3.5-2b | refexp_box | 62.4 / 7.4 | 4741 / 4741 |
| qwen3.5-2b | refexp_poly | 55.2 / 9.9 | 4342 / 4234 |
| qwen3.5-2b | grid16 | 1240.6 / 95.8 | 101631 / 101631 |
| qwen3.5-2b | count | 2.5 / 0.4 | 104 / 104 |
| qwen3.5-2b | ovd | 1181.7 / 177.5 | 96522 / 95764 |
| qwen3.5-2b | rel_depth_nyu | 229.6 / 45.0 | 18394 / 20013 |
| qwen3.5-2b | rel_depth_blink | 37.4 / 8.7 | 3064 / 3044 |
| qwen3.5-2b | spatial_cvbench | 110.9 / 23.1 | 9100 / 9128 |
| qwen3.5-2b | spatial_embspatial | 75.3 / 13.3 | 6152 / 6234 |
| qwen3.5-2b | 合计 | 2995.6 / 381.2 | |
| gemma-4-e2b-it | refexp_box | 101.7 / 27.4 | 5557 / 5506 |
| gemma-4-e2b-it | refexp_poly | 154.2 / 56.8 | 8448 / 8815 |
| gemma-4-e2b-it | grid16 | 447.9 / 35.4 | 24530 / 24470 |
| gemma-4-e2b-it | count | 2.9 / 0.7 | 103 / 103 |
| gemma-4-e2b-it | ovd | 295.3 / 49.2 | 15823 / 15957 |
| gemma-4-e2b-it | rel_depth_nyu | 515.1 / 59.4 | 28141 / 28000 |
| gemma-4-e2b-it | rel_depth_blink | 63.4 / 9.6 | 3465 / 3246 |
| gemma-4-e2b-it | spatial_cvbench | 180.3 / 47.1 | 9538 / 9823 |
| gemma-4-e2b-it | spatial_embspatial | 307.7 / 62.5 | 16882 / 17318 |
| gemma-4-e2b-it | 合计 | 2068.5 / 348.1 | |
| internvl3.5-2b-hf | refexp_box | 32.5 / 5.3 | 2836 / 2838 |
| internvl3.5-2b-hf | refexp_poly | 285.8 / 88.8 | 25482 / 26523 |
| internvl3.5-2b-hf | grid16 | 257.0 / 22.0 | 25218 / 25299 |
| internvl3.5-2b-hf | count | 33.0 / 9.7 | 3017 / 3361 |
| internvl3.5-2b-hf | ovd | 229.7 / 121.5 | 19824 / 21399 |
| internvl3.5-2b-hf | rel_depth_nyu | 171.5 / 26.8 | 16482 / 16760 |
| internvl3.5-2b-hf | rel_depth_blink | 24.8 / 11.9 | 2377 / 2449 |
| internvl3.5-2b-hf | spatial_cvbench | 77.6 / 19.8 | 7241 / 7217 |
| internvl3.5-2b-hf | spatial_embspatial | 68.9 / 11.0 | 6599 / 6146 |
| internvl3.5-2b-hf | 合计 | 1180.8 / 316.7 | |
| qwen3.5-4b | refexp_box | 82.2 / 13.1 | 4908 / 4907 |
| qwen3.5-4b | refexp_poly | 68.6 / 10.4 | 4080 / 4113 |
| qwen3.5-4b | grid16 | 445.8 / 34.9 | 27234 / 27200 |
| qwen3.5-4b | count | 3.2 / 0.6 | 105 / 105 |
| qwen3.5-4b | ovd | 1036.3 / 232.2 | 62337 / 61375 |
| qwen3.5-4b | rel_depth_nyu | 376.2 / 59.9 | 22534 / 21015 |
| qwen3.5-4b | rel_depth_blink | 37.7 / 5.1 | 2298 / 2212 |
| qwen3.5-4b | spatial_cvbench | 118.8 / 30.7 | 7254 / 8179 |
| qwen3.5-4b | spatial_embspatial | 158.7 / 52.1 | 9705 / 9709 |
| qwen3.5-4b | 合计 | 2327.5 / 439.0 | |
| gemma-4-e4b-it | refexp_box | 113.5 / 30.0 | 4950 / 5261 |
| gemma-4-e4b-it | refexp_poly | 163.4 / 24.3 | 7151 / 7150 |
| gemma-4-e4b-it | grid16 | 627.4 / 48.6 | 27200 / 27200 |
| gemma-4-e4b-it | count | 3.4 / 0.8 | 104 / 104 |
| gemma-4-e4b-it | ovd | 454.9 / 96.1 | 19748 / 19950 |
| gemma-4-e4b-it | rel_depth_nyu | 604.0 / 84.7 | 26521 / 25848 |
| gemma-4-e4b-it | rel_depth_blink | 61.8 / 15.5 | 2734 / 3252 |
| gemma-4-e4b-it | spatial_cvbench | 153.1 / 28.4 | 6607 / 6746 |
| gemma-4-e4b-it | spatial_embspatial | 307.8 / 75.6 | 13156 / 12705 |
| gemma-4-e4b-it | 合计 | 2489.3 / 404.0 | |
| internvl3.5-4b-hf | refexp_box | 38.6 / 5.9 | 2839 / 2842 |
| internvl3.5-4b-hf | refexp_poly | 173.6 / 117.0 | 11766 / 13509 |
| internvl3.5-4b-hf | grid16 | 344.0 / 37.6 | 27200 / 27200 |
| internvl3.5-4b-hf | count | 5.2 / 1.2 | 372 / 373 |
| internvl3.5-4b-hf | ovd | 1006.2 / 285.4 | 76150 / 73661 |
| internvl3.5-4b-hf | rel_depth_nyu | 193.5 / 32.8 | 15032 / 15188 |
| internvl3.5-4b-hf | rel_depth_blink | 37.8 / 11.4 | 2787 / 2999 |
| internvl3.5-4b-hf | spatial_cvbench | 84.9 / 23.3 | 6313 / 6176 |
| internvl3.5-4b-hf | spatial_embspatial | 29.9 / 6.2 | 2186 / 2038 |
| internvl3.5-4b-hf | 合计 | 1913.6 / 520.7 | |
| qwen3.5-9b | refexp_box | 80.8 / 11.3 | 4770 / 4771 |
| qwen3.5-9b | refexp_poly | 84.4 / 28.5 | 4999 / 4972 |
| qwen3.5-9b | grid16 | 451.9 / 36.9 | 27442 / 27425 |
| qwen3.5-9b | count | 3.1 / 0.8 | 106 / 106 |
| qwen3.5-9b | ovd | 1144.2 / 236.3 | 68815 / 64274 |
| qwen3.5-9b | rel_depth_nyu | 294.3 / 56.4 | 17607 / 18560 |
| qwen3.5-9b | rel_depth_blink | 42.1 / 6.8 | 2461 / 2292 |
| qwen3.5-9b | spatial_cvbench | 125.5 / 33.5 | 7406 / 7810 |
| qwen3.5-9b | spatial_embspatial | 132.0 / 33.8 | 7754 / 7641 |
| qwen3.5-9b | 合计 | 2358.3 / 444.2 | |
