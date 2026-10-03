#### 主表：三列（严格读数 / 有效率 / 条件读数）与 95% CI（metrics.json 的 three_columns）

| 模型 | 任务 | 指标（范围） | n | 严格 | 严格 95% CI | 有效 n_valid/n（率） | 条件 | 条件 95% CI |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | acc@0.5（all） | 56857 | 0.7817 | [0.7784, 0.7851] | 55852/56857（0.982） | 0.7958 | [0.7923, 0.7994] |
| qwen3.5-0.8b | refexp_poly | cIoU（all） | 8000 | 0.0000 | [0.0000, 0.0000] | 0/8000（0.000） | — | — |
| qwen3.5-0.8b | refexp_poly | gIoU（all） | 8000 | 0.0000 | [0.0000, 0.0000] | 0/8000（0.000） | — | — |
| qwen3.5-0.8b | grid16 | gIoU（all） | 500 | 0.0000 | [0.0000, 0.0000] | 0/500（0.000） | — | — |
| qwen3.5-0.8b | grid16 | cIoU（all） | 500 | 0.0000 | [0.0000, 0.0000] | 0/500（0.000） | — | — |
| qwen3.5-0.8b | count | accuracy（all） | 442 | 0.7624 | [0.7195, 0.8009] | 442/442（1.000） | 0.7624 | [0.7195, 0.8009] |
| qwen3.5-0.8b | ovd | F1@0.5（coco_val2017） | 500 | 0.2396 | [0.2079, 0.2711] | 266/500（0.532） | 0.5591 | [0.5147, 0.6043] |
| qwen3.5-0.8b | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.5986 | — | 406/463（0.877） | 0.6765 | — |
| qwen3.5-0.8b | rel_depth_nyu | accuracy（all） | 3270 | 0.7330 | [0.7162, 0.7489] | 3270/3270（1.000） | 0.7330 | [0.7162, 0.7489] |
| qwen3.5-0.8b | rel_depth_blink | accuracy（all） | 112 | 0.7589 | [0.6786, 0.8393] | 112/112（1.000） | 0.7589 | [0.6786, 0.8393] |
| qwen3.5-0.8b | spatial_cvbench | accuracy（all） | 2588 | 0.4726 | [0.4521, 0.4923] | 2588/2588（1.000） | 0.4726 | [0.4521, 0.4923] |
| qwen3.5-0.8b | spatial_embspatial | accuracy（all） | 3590 | 0.4632 | [0.4465, 0.4794] | 3590/3590（1.000） | 0.4632 | [0.4465, 0.4794] |
| qwen3.5-2b | refexp_box | acc@0.5（all） | 56857 | 0.8309 | [0.8278, 0.8341] | 56245/56857（0.989） | 0.8400 | [0.8372, 0.8428] |
| qwen3.5-2b | refexp_poly | cIoU（all） | 8000 | 0.0005 | [0.0001, 0.0009] | 12/8000（0.002） | 0.1588 | [0.0858, 0.3402] |
| qwen3.5-2b | refexp_poly | gIoU（all） | 8000 | 0.0003 | [0.0001, 0.0006] | 12/8000（0.002） | 0.2301 | [0.1095, 0.3723] |
| qwen3.5-2b | grid16 | gIoU（all） | 500 | 0.0005 | [0.0000, 0.0013] | 5/500（0.010） | 0.0501 | [0.0000, 0.1143] |
| qwen3.5-2b | grid16 | cIoU（all） | 500 | 0.0045 | [0.0000, 0.0120] | 5/500（0.010） | 0.1274 | [0.0000, 0.1540] |
| qwen3.5-2b | count | accuracy（all） | 442 | 0.8643 | [0.8326, 0.8937] | 442/442（1.000） | 0.8643 | [0.8326, 0.8937] |
| qwen3.5-2b | ovd | F1@0.5（coco_val2017） | 500 | 0.4168 | [0.3764, 0.4564] | 381/500（0.762） | 0.6292 | [0.5929, 0.6667] |
| qwen3.5-2b | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.6233 | — | 439/463（0.948） | 0.6508 | — |
| qwen3.5-2b | rel_depth_nyu | accuracy（all） | 3270 | 0.7425 | [0.7275, 0.7572] | 3270/3270（1.000） | 0.7425 | [0.7275, 0.7572] |
| qwen3.5-2b | rel_depth_blink | accuracy（all） | 112 | 0.5357 | [0.4375, 0.6252] | 112/112（1.000） | 0.5357 | [0.4375, 0.6252] |
| qwen3.5-2b | spatial_cvbench | accuracy（all） | 2588 | 0.5313 | [0.5112, 0.5510] | 2588/2588（1.000） | 0.5313 | [0.5112, 0.5510] |
| qwen3.5-2b | spatial_embspatial | accuracy（all） | 3590 | 0.5772 | [0.5610, 0.5936] | 3590/3590（1.000） | 0.5772 | [0.5610, 0.5936] |
| qwen3.5-4b | refexp_box | acc@0.5（all） | 56857 | 0.8474 | [0.8445, 0.8505] | 54927/56857（0.966） | 0.8771 | [0.8744, 0.8798] |
| qwen3.5-4b | refexp_poly | cIoU（all） | 8000 | 0.0000 | [0.0000, 0.0000] | 0/8000（0.000） | — | — |
| qwen3.5-4b | refexp_poly | gIoU（all） | 8000 | 0.0000 | [0.0000, 0.0000] | 0/8000（0.000） | — | — |
| qwen3.5-4b | grid16 | gIoU（all） | 500 | 0.0143 | [0.0068, 0.0224] | 499/500（0.998） | 0.0143 | [0.0073, 0.0240] |
| qwen3.5-4b | grid16 | cIoU（all） | 500 | 0.1022 | [0.0549, 0.1483] | 499/500（0.998） | 0.1023 | [0.0572, 0.1540] |
| qwen3.5-4b | count | accuracy（all） | 442 | 0.8643 | [0.8326, 0.8959] | 442/442（1.000） | 0.8643 | [0.8326, 0.8959] |
| qwen3.5-4b | ovd | F1@0.5（coco_val2017） | 500 | 0.4864 | [0.4531, 0.5239] | 443/500（0.886） | 0.5697 | [0.5333, 0.6095] |
| qwen3.5-4b | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.7032 | — | 456/463（0.985） | 0.7121 | — |
| qwen3.5-4b | rel_depth_nyu | accuracy（all） | 3270 | 0.5009 | [0.4847, 0.5193] | 3270/3270（1.000） | 0.5009 | [0.4847, 0.5193] |
| qwen3.5-4b | rel_depth_blink | accuracy（all） | 112 | 0.5179 | [0.4283, 0.6161] | 112/112（1.000） | 0.5179 | [0.4283, 0.6161] |
| qwen3.5-4b | spatial_cvbench | accuracy（all） | 2588 | 0.4915 | [0.4718, 0.5097] | 2588/2588（1.000） | 0.4915 | [0.4718, 0.5097] |
| qwen3.5-4b | spatial_embspatial | accuracy（all） | 3590 | 0.7376 | [0.7226, 0.7518] | 3590/3590（1.000） | 0.7376 | [0.7226, 0.7518] |
| qwen3.5-9b | refexp_box | acc@0.5（all） | 56857 | 0.8717 | [0.8691, 0.8745] | 55801/56857（0.981） | 0.8882 | [0.8853, 0.8907] |
| qwen3.5-9b | refexp_poly | cIoU（all） | 8000 | 0.0000 | [0.0000, 0.0000] | 0/8000（0.000） | — | — |
| qwen3.5-9b | refexp_poly | gIoU（all） | 8000 | 0.0000 | [0.0000, 0.0000] | 0/8000（0.000） | — | — |
| qwen3.5-9b | grid16 | gIoU（all） | 500 | 0.0015 | [0.0000, 0.0046] | 423/500（0.846） | 0.0018 | [0.0000, 0.0054] |
| qwen3.5-9b | grid16 | cIoU（all） | 500 | 0.0074 | [0.0000, 0.0225] | 423/500（0.846） | 0.0085 | [0.0000, 0.0272] |
| qwen3.5-9b | count | accuracy（all） | 442 | 0.8959 | [0.8643, 0.9231] | 442/442（1.000） | 0.8959 | [0.8643, 0.9231] |
| qwen3.5-9b | ovd | F1@0.5（coco_val2017） | 500 | 0.5522 | [0.5196, 0.5847] | 450/500（0.900） | 0.6279 | [0.6012, 0.6564] |
| qwen3.5-9b | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.7577 | — | 458/463（0.989） | 0.7640 | — |
| qwen3.5-9b | rel_depth_nyu | accuracy（all） | 3270 | 0.8951 | [0.8844, 0.9049] | 3270/3270（1.000） | 0.8951 | [0.8844, 0.9049] |
| qwen3.5-9b | rel_depth_blink | accuracy（all） | 112 | 0.8750 | [0.8125, 0.9286] | 112/112（1.000） | 0.8750 | [0.8125, 0.9286] |
| qwen3.5-9b | spatial_cvbench | accuracy（all） | 2588 | 0.7987 | [0.7832, 0.8141] | 2588/2588（1.000） | 0.7987 | [0.7832, 0.8141] |
| qwen3.5-9b | spatial_embspatial | accuracy（all） | 3590 | 0.7724 | [0.7596, 0.7855] | 3590/3590（1.000） | 0.7724 | [0.7596, 0.7855] |
| internvl3.5-2b-hf | refexp_box | acc@0.5（all） | 56857 | 0.7111 | [0.7073, 0.7144] | 56856/56857（1.000） | 0.7111 | [0.7073, 0.7146] |
| internvl3.5-2b-hf | refexp_poly | cIoU（all） | 8000 | 0.0763 | [0.0722, 0.0804] | 6073/8000（0.759） | 0.0962 | [0.0916, 0.1012] |
| internvl3.5-2b-hf | refexp_poly | gIoU（all） | 8000 | 0.0594 | [0.0570, 0.0615] | 6073/8000（0.759） | 0.0782 | [0.0753, 0.0811] |
| internvl3.5-2b-hf | grid16 | gIoU（all） | 500 | 0.0000 | [0.0000, 0.0000] | 1/500（0.002） | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | grid16 | cIoU（all） | 500 | 0.0000 | [0.0000, 0.0000] | 1/500（0.002） | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | count | accuracy（all） | 442 | 0.5769 | [0.5317, 0.6244] | 392/442（0.887） | 0.6505 | [0.6020, 0.6964] |
| internvl3.5-2b-hf | ovd | F1@0.5（coco_val2017） | 500 | 0.0000 | [0.0000, 0.0000] | 0/500（0.000） | — | — |
| internvl3.5-2b-hf | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.0472 | — | 99/463（0.214） | 0.1334 | — |
| internvl3.5-2b-hf | rel_depth_nyu | accuracy（all） | 3270 | 0.5251 | [0.5076, 0.5416] | 3270/3270（1.000） | 0.5251 | [0.5076, 0.5416] |
| internvl3.5-2b-hf | rel_depth_blink | accuracy（all） | 112 | 0.6161 | [0.5266, 0.7054] | 112/112（1.000） | 0.6161 | [0.5266, 0.7054] |
| internvl3.5-2b-hf | spatial_cvbench | accuracy（all） | 2588 | 0.6712 | [0.6530, 0.6886] | 2588/2588（1.000） | 0.6712 | [0.6530, 0.6886] |
| internvl3.5-2b-hf | spatial_embspatial | accuracy（all） | 3590 | 0.5362 | [0.5198, 0.5549] | 3590/3590（1.000） | 0.5362 | [0.5198, 0.5549] |
| internvl3.5-4b-hf | refexp_box | acc@0.5（all） | 56857 | 0.8448 | [0.8419, 0.8478] | 56851/56857（1.000） | 0.8449 | [0.8420, 0.8479] |
| internvl3.5-4b-hf | refexp_poly | cIoU（all） | 8000 | 0.1823 | [0.1765, 0.1888] | 7631/8000（0.954） | 0.1918 | [0.1855, 0.1979] |
| internvl3.5-4b-hf | refexp_poly | gIoU（all） | 8000 | 0.1653 | [0.1615, 0.1691] | 7631/8000（0.954） | 0.1733 | [0.1694, 0.1773] |
| internvl3.5-4b-hf | grid16 | gIoU（all） | 500 | 0.0037 | [0.0017, 0.0060] | 498/500（0.996） | 0.0037 | [0.0015, 0.0062] |
| internvl3.5-4b-hf | grid16 | cIoU（all） | 500 | 0.0188 | [0.0060, 0.0325] | 498/500（0.996） | 0.0189 | [0.0063, 0.0333] |
| internvl3.5-4b-hf | count | accuracy（all） | 442 | 0.7240 | [0.6810, 0.7625] | 442/442（1.000） | 0.7240 | [0.6810, 0.7625] |
| internvl3.5-4b-hf | ovd | F1@0.5（coco_val2017） | 500 | 0.1263 | [0.1122, 0.1412] | 459/500（0.918） | 0.1342 | [0.1201, 0.1508] |
| internvl3.5-4b-hf | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.1206 | — | 404/463（0.873） | 0.1230 | — |
| internvl3.5-4b-hf | rel_depth_nyu | accuracy（all） | 3270 | 0.7774 | [0.7633, 0.7924] | 3270/3270（1.000） | 0.7774 | [0.7633, 0.7924] |
| internvl3.5-4b-hf | rel_depth_blink | accuracy（all） | 112 | 0.7857 | [0.7054, 0.8571] | 112/112（1.000） | 0.7857 | [0.7054, 0.8571] |
| internvl3.5-4b-hf | spatial_cvbench | accuracy（all） | 2588 | 0.7299 | [0.7121, 0.7473] | 2588/2588（1.000） | 0.7299 | [0.7121, 0.7473] |
| internvl3.5-4b-hf | spatial_embspatial | accuracy（all） | 3590 | 0.6997 | [0.6847, 0.7145] | 3590/3590（1.000） | 0.6997 | [0.6847, 0.7145] |
| gemma-4-e2b-it | refexp_box | acc@0.5（all） | 56857 | 0.1455 | [0.1425, 0.1484] | 11576/56857（0.204） | 0.7148 | [0.7069, 0.7231] |
| gemma-4-e2b-it | refexp_poly | cIoU（all） | 8000 | 0.2405 | [0.2343, 0.2472] | 7901/8000（0.988） | 0.2423 | [0.2367, 0.2493] |
| gemma-4-e2b-it | refexp_poly | gIoU（all） | 8000 | 0.2293 | [0.2246, 0.2343] | 7901/8000（0.988） | 0.2321 | [0.2277, 0.2370] |
| gemma-4-e2b-it | grid16 | gIoU（all） | 500 | 0.0000 | [0.0000, 0.0000] | 0/500（0.000） | — | — |
| gemma-4-e2b-it | grid16 | cIoU（all） | 500 | 0.0000 | [0.0000, 0.0000] | 0/500（0.000） | — | — |
| gemma-4-e2b-it | count | accuracy（all） | 442 | 0.5271 | [0.4773, 0.5747] | 442/442（1.000） | 0.5271 | [0.4773, 0.5747] |
| gemma-4-e2b-it | ovd | F1@0.5（coco_val2017） | 500 | 0.2488 | [0.2246, 0.2754] | 346/500（0.692） | 0.3430 | [0.3164, 0.3775] |
| gemma-4-e2b-it | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.0367 | — | 25/463（0.054） | 0.4237 | — |
| gemma-4-e2b-it | rel_depth_nyu | accuracy（all） | 3270 | 0.5000 | [0.4841, 0.5183] | 3270/3270（1.000） | 0.5000 | [0.4841, 0.5183] |
| gemma-4-e2b-it | rel_depth_blink | accuracy（all） | 112 | 0.5357 | [0.4464, 0.6250] | 112/112（1.000） | 0.5357 | [0.4464, 0.6250] |
| gemma-4-e2b-it | spatial_cvbench | accuracy（all） | 2588 | 0.4231 | [0.4038, 0.4424] | 2588/2588（1.000） | 0.4231 | [0.4038, 0.4424] |
| gemma-4-e2b-it | spatial_embspatial | accuracy（all） | 3590 | 0.2585 | [0.2440, 0.2730] | 3590/3590（1.000） | 0.2585 | [0.2440, 0.2730] |
| gemma-4-e4b-it | refexp_box | acc@0.5（all） | 56857 | 0.2057 | [0.2024, 0.2089] | 14467/56857（0.254） | 0.8083 | [0.8021, 0.8146] |
| gemma-4-e4b-it | refexp_poly | cIoU（all） | 8000 | 0.0577 | [0.0521, 0.0636] | 1106/8000（0.138） | 0.3088 | [0.2871, 0.3316] |
| gemma-4-e4b-it | refexp_poly | gIoU（all） | 8000 | 0.0376 | [0.0349, 0.0404] | 1106/8000（0.138） | 0.2720 | [0.2571, 0.2880] |
| gemma-4-e4b-it | grid16 | gIoU（all） | 500 | 0.0271 | [0.0196, 0.0353] | 500/500（1.000） | 0.0271 | [0.0196, 0.0353] |
| gemma-4-e4b-it | grid16 | cIoU（all） | 500 | 0.0904 | [0.0652, 0.1169] | 500/500（1.000） | 0.0904 | [0.0652, 0.1169] |
| gemma-4-e4b-it | count | accuracy（all） | 442 | 0.5226 | [0.4774, 0.5679] | 442/442（1.000） | 0.5226 | [0.4774, 0.5679] |
| gemma-4-e4b-it | ovd | F1@0.5（coco_val2017） | 500 | 0.4366 | [0.4105, 0.4633] | 480/500（0.960） | 0.4425 | [0.4152, 0.4684] |
| gemma-4-e4b-it | ovd | F1@0.5（odinw13_mean_over_datasets） | 463 | 0.1529 | — | 133/463（0.287） | 0.5240 | — |
| gemma-4-e4b-it | rel_depth_nyu | accuracy（all） | 3270 | 0.5361 | [0.5199, 0.5547] | 3270/3270（1.000） | 0.5361 | [0.5199, 0.5547] |
| gemma-4-e4b-it | rel_depth_blink | accuracy（all） | 112 | 0.5268 | [0.4286, 0.6250] | 112/112（1.000） | 0.5268 | [0.4286, 0.6250] |
| gemma-4-e4b-it | spatial_cvbench | accuracy（all） | 2588 | 0.5402 | [0.5205, 0.5591] | 2588/2588（1.000） | 0.5402 | [0.5205, 0.5591] |
| gemma-4-e4b-it | spatial_embspatial | accuracy（all） | 3590 | 0.4744 | [0.4574, 0.4917] | 3590/3590（1.000） | 0.4744 | [0.4574, 0.4917] |

#### 定位、多边形、网格按划分分列（metrics.json 的 breakdown.split：严格读数、95% CI、n、解析失败率）

| 模型 | 任务 | 划分 | 指标 | n | 严格 | 95% CI | 解析失败率 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.7843 | [0.7744, 0.7951] | 0.0145 |
| qwen3.5-0.8b | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.6333 | [0.6198, 0.6474] | 0.0374 |
| qwen3.5-0.8b | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.7227 | [0.7148, 0.7312] | 0.0241 |
| qwen3.5-0.8b | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.8665 | [0.8573, 0.8754] | 0.0058 |
| qwen3.5-0.8b | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.7808 | [0.7694, 0.7922] | 0.0137 |
| qwen3.5-0.8b | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.8393 | [0.8327, 0.8460] | 0.0087 |
| qwen3.5-0.8b | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.7997 | [0.7920, 0.8079] | 0.0198 |
| qwen3.5-0.8b | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.7975 | [0.7860, 0.8086] | 0.0211 |
| qwen3.5-0.8b | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | grid16 | refcoco_unc/val | cIoU | 500 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-0.8b | grid16 | refcoco_unc/val | gIoU | 500 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.8313 | [0.8217, 0.8411] | 0.0082 |
| qwen3.5-2b | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.7122 | [0.6997, 0.7247] | 0.0227 |
| qwen3.5-2b | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.7852 | [0.7779, 0.7929] | 0.0118 |
| qwen3.5-2b | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.8989 | [0.8913, 0.9065] | 0.0041 |
| qwen3.5-2b | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.8324 | [0.8220, 0.8422] | 0.0110 |
| qwen3.5-2b | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.8715 | [0.8654, 0.8776] | 0.0064 |
| qwen3.5-2b | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.8466 | [0.8391, 0.8535] | 0.0132 |
| qwen3.5-2b | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.8492 | [0.8392, 0.8592] | 0.0117 |
| qwen3.5-2b | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 0.9990 |
| qwen3.5-2b | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 0.9990 |
| qwen3.5-2b | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.0003 | [0.0000, 0.0009] | 0.9990 |
| qwen3.5-2b | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.0005 | [0.0000, 0.0015] | 0.9990 |
| qwen3.5-2b | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.0015 | [0.0000, 0.0043] | 0.9980 |
| qwen3.5-2b | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.0005 | [0.0000, 0.0015] | 0.9980 |
| qwen3.5-2b | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-2b | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.0019 | [0.0000, 0.0041] | 0.9940 |
| qwen3.5-2b | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.0009 | [0.0000, 0.0023] | 0.9940 |
| qwen3.5-2b | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.0003 | [0.0000, 0.0008] | 0.9980 |
| qwen3.5-2b | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.0008 | [0.0000, 0.0023] | 0.9980 |
| qwen3.5-2b | grid16 | refcoco_unc/val | cIoU | 500 | 0.0045 | [0.0000, 0.0120] | 0.9900 |
| qwen3.5-2b | grid16 | refcoco_unc/val | gIoU | 500 | 0.0005 | [0.0000, 0.0013] | 0.9900 |
| qwen3.5-4b | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.8491 | [0.8404, 0.8577] | 0.0307 |
| qwen3.5-4b | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.7460 | [0.7333, 0.7576] | 0.0587 |
| qwen3.5-4b | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.8046 | [0.7973, 0.8122] | 0.0421 |
| qwen3.5-4b | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.9056 | [0.8982, 0.9130] | 0.0152 |
| qwen3.5-4b | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.8569 | [0.8469, 0.8661] | 0.0232 |
| qwen3.5-4b | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.8863 | [0.8798, 0.8920] | 0.0188 |
| qwen3.5-4b | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.8586 | [0.8510, 0.8653] | 0.0443 |
| qwen3.5-4b | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.8554 | [0.8458, 0.8652] | 0.0413 |
| qwen3.5-4b | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-4b | grid16 | refcoco_unc/val | cIoU | 500 | 0.1022 | [0.0549, 0.1483] | 0.0020 |
| qwen3.5-4b | grid16 | refcoco_unc/val | gIoU | 500 | 0.0143 | [0.0068, 0.0224] | 0.0020 |
| qwen3.5-9b | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.8833 | [0.8753, 0.8916] | 0.0157 |
| qwen3.5-9b | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.7811 | [0.7693, 0.7926] | 0.0358 |
| qwen3.5-9b | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.8342 | [0.8276, 0.8414] | 0.0224 |
| qwen3.5-9b | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.9275 | [0.9208, 0.9339] | 0.0072 |
| qwen3.5-9b | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.8724 | [0.8636, 0.8817] | 0.0153 |
| qwen3.5-9b | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.9054 | [0.8999, 0.9109] | 0.0117 |
| qwen3.5-9b | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.8776 | [0.8707, 0.8840] | 0.0222 |
| qwen3.5-9b | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.8795 | [0.8701, 0.8886] | 0.0211 |
| qwen3.5-9b | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| qwen3.5-9b | grid16 | refcoco_unc/val | cIoU | 500 | 0.0074 | [0.0000, 0.0225] | 0.1540 |
| qwen3.5-9b | grid16 | refcoco_unc/val | gIoU | 500 | 0.0015 | [0.0000, 0.0046] | 0.1540 |
| internvl3.5-2b-hf | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.7419 | [0.7311, 0.7534] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.6247 | [0.6105, 0.6380] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.6756 | [0.6666, 0.6848] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.7870 | [0.7767, 0.7974] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.7115 | [0.6985, 0.7237] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.7359 | [0.7278, 0.7442] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.7035 | [0.6946, 0.7126] | 0.0000 |
| internvl3.5-2b-hf | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.7106 | [0.6978, 0.7236] | 0.0002 |
| internvl3.5-2b-hf | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.0641 | [0.0545, 0.0745] | 0.2780 |
| internvl3.5-2b-hf | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.0521 | [0.0460, 0.0584] | 0.2780 |
| internvl3.5-2b-hf | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.0835 | [0.0719, 0.0950] | 0.2560 |
| internvl3.5-2b-hf | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.0620 | [0.0553, 0.0691] | 0.2560 |
| internvl3.5-2b-hf | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.0729 | [0.0597, 0.0867] | 0.2600 |
| internvl3.5-2b-hf | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.0502 | [0.0445, 0.0569] | 0.2600 |
| internvl3.5-2b-hf | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.0757 | [0.0671, 0.0852] | 0.2340 |
| internvl3.5-2b-hf | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.0648 | [0.0589, 0.0711] | 0.2340 |
| internvl3.5-2b-hf | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.0884 | [0.0756, 0.1031] | 0.2310 |
| internvl3.5-2b-hf | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.0676 | [0.0606, 0.0741] | 0.2310 |
| internvl3.5-2b-hf | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.0808 | [0.0691, 0.0938] | 0.2330 |
| internvl3.5-2b-hf | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.0645 | [0.0577, 0.0713] | 0.2330 |
| internvl3.5-2b-hf | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.0702 | [0.0605, 0.0806] | 0.2200 |
| internvl3.5-2b-hf | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.0561 | [0.0502, 0.0624] | 0.2200 |
| internvl3.5-2b-hf | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.0736 | [0.0634, 0.0836] | 0.2150 |
| internvl3.5-2b-hf | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.0576 | [0.0512, 0.0641] | 0.2150 |
| internvl3.5-2b-hf | grid16 | refcoco_unc/val | cIoU | 500 | 0.0000 | [0.0000, 0.0000] | 0.9980 |
| internvl3.5-2b-hf | grid16 | refcoco_unc/val | gIoU | 500 | 0.0000 | [0.0000, 0.0000] | 0.9980 |
| internvl3.5-4b-hf | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.8709 | [0.8626, 0.8797] | 0.0000 |
| internvl3.5-4b-hf | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.7691 | [0.7572, 0.7807] | 0.0000 |
| internvl3.5-4b-hf | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.8182 | [0.8107, 0.8254] | 0.0001 |
| internvl3.5-4b-hf | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.9001 | [0.8927, 0.9074] | 0.0000 |
| internvl3.5-4b-hf | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.8414 | [0.8306, 0.8516] | 0.0006 |
| internvl3.5-4b-hf | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.8734 | [0.8667, 0.8795] | 0.0001 |
| internvl3.5-4b-hf | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.8385 | [0.8313, 0.8457] | 0.0001 |
| internvl3.5-4b-hf | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.8371 | [0.8269, 0.8480] | 0.0000 |
| internvl3.5-4b-hf | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.1509 | [0.1340, 0.1692] | 0.0290 |
| internvl3.5-4b-hf | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.1294 | [0.1206, 0.1382] | 0.0290 |
| internvl3.5-4b-hf | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.2085 | [0.1917, 0.2243] | 0.0620 |
| internvl3.5-4b-hf | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.1886 | [0.1771, 0.2014] | 0.0620 |
| internvl3.5-4b-hf | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.1532 | [0.1378, 0.1713] | 0.0430 |
| internvl3.5-4b-hf | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.1376 | [0.1279, 0.1474] | 0.0430 |
| internvl3.5-4b-hf | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.1568 | [0.1451, 0.1691] | 0.0250 |
| internvl3.5-4b-hf | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.1445 | [0.1355, 0.1534] | 0.0250 |
| internvl3.5-4b-hf | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.2247 | [0.2096, 0.2403] | 0.0710 |
| internvl3.5-4b-hf | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.2122 | [0.1998, 0.2254] | 0.0710 |
| internvl3.5-4b-hf | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.1926 | [0.1759, 0.2095] | 0.0520 |
| internvl3.5-4b-hf | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.1835 | [0.1716, 0.1958] | 0.0520 |
| internvl3.5-4b-hf | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.1809 | [0.1609, 0.2033] | 0.0430 |
| internvl3.5-4b-hf | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.1631 | [0.1521, 0.1744] | 0.0430 |
| internvl3.5-4b-hf | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.1827 | [0.1658, 0.1996] | 0.0440 |
| internvl3.5-4b-hf | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.1634 | [0.1528, 0.1755] | 0.0440 |
| internvl3.5-4b-hf | grid16 | refcoco_unc/val | cIoU | 500 | 0.0188 | [0.0060, 0.0325] | 0.0040 |
| internvl3.5-4b-hf | grid16 | refcoco_unc/val | gIoU | 500 | 0.0037 | [0.0017, 0.0060] | 0.0040 |
| gemma-4-e2b-it | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.0498 | [0.0444, 0.0557] | 0.9263 |
| gemma-4-e2b-it | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.0522 | [0.0456, 0.0581] | 0.9053 |
| gemma-4-e2b-it | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.0490 | [0.0447, 0.0531] | 0.9185 |
| gemma-4-e2b-it | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.0783 | [0.0719, 0.0850] | 0.8922 |
| gemma-4-e2b-it | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.1134 | [0.1046, 0.1223] | 0.8243 |
| gemma-4-e2b-it | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.0874 | [0.0820, 0.0925] | 0.8719 |
| gemma-4-e2b-it | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.3704 | [0.3605, 0.3802] | 0.5109 |
| gemma-4-e2b-it | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.3644 | [0.3501, 0.3782] | 0.5172 |
| gemma-4-e2b-it | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.2179 | [0.2003, 0.2348] | 0.0110 |
| gemma-4-e2b-it | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.2013 | [0.1889, 0.2134] | 0.0110 |
| gemma-4-e2b-it | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.2193 | [0.2007, 0.2375] | 0.0180 |
| gemma-4-e2b-it | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.2052 | [0.1923, 0.2186] | 0.0180 |
| gemma-4-e2b-it | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.2028 | [0.1869, 0.2190] | 0.0190 |
| gemma-4-e2b-it | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.1841 | [0.1718, 0.1961] | 0.0190 |
| gemma-4-e2b-it | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.2587 | [0.2420, 0.2766] | 0.0050 |
| gemma-4-e2b-it | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.2513 | [0.2388, 0.2648] | 0.0050 |
| gemma-4-e2b-it | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.2711 | [0.2516, 0.2915] | 0.0150 |
| gemma-4-e2b-it | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.2687 | [0.2550, 0.2837] | 0.0150 |
| gemma-4-e2b-it | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.2497 | [0.2293, 0.2713] | 0.0110 |
| gemma-4-e2b-it | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.2455 | [0.2319, 0.2589] | 0.0110 |
| gemma-4-e2b-it | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.2567 | [0.2372, 0.2759] | 0.0100 |
| gemma-4-e2b-it | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.2366 | [0.2238, 0.2492] | 0.0100 |
| gemma-4-e2b-it | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.2553 | [0.2363, 0.2755] | 0.0100 |
| gemma-4-e2b-it | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.2416 | [0.2279, 0.2553] | 0.0100 |
| gemma-4-e2b-it | grid16 | refcoco_unc/val | cIoU | 500 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| gemma-4-e2b-it | grid16 | refcoco_unc/val | gIoU | 500 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| gemma-4-e4b-it | refexp_box | refcoco+_unc/testA | acc@0.5 | 5726 | 0.0899 | [0.0828, 0.0973] | 0.8922 |
| gemma-4-e4b-it | refexp_box | refcoco+_unc/testB | acc@0.5 | 4889 | 0.0916 | [0.0830, 0.0998] | 0.8542 |
| gemma-4-e4b-it | refexp_box | refcoco+_unc/val | acc@0.5 | 10558 | 0.0887 | [0.0833, 0.0943] | 0.8780 |
| gemma-4-e4b-it | refexp_box | refcoco_unc/testA | acc@0.5 | 5657 | 0.1635 | [0.1543, 0.1732] | 0.8105 |
| gemma-4-e4b-it | refexp_box | refcoco_unc/testB | acc@0.5 | 5095 | 0.2573 | [0.2447, 0.2701] | 0.6636 |
| gemma-4-e4b-it | refexp_box | refcoco_unc/val | acc@0.5 | 10634 | 0.1971 | [0.1893, 0.2046] | 0.7521 |
| gemma-4-e4b-it | refexp_box | refcocog_umd/test | acc@0.5 | 9602 | 0.3841 | [0.3742, 0.3939] | 0.5504 |
| gemma-4-e4b-it | refexp_box | refcocog_umd/val | acc@0.5 | 4696 | 0.3778 | [0.3637, 0.3912] | 0.5507 |
| gemma-4-e4b-it | refexp_poly | refcoco+_unc/testA | cIoU | 1000 | 0.0317 | [0.0234, 0.0410] | 0.8920 |
| gemma-4-e4b-it | refexp_poly | refcoco+_unc/testA | gIoU | 1000 | 0.0234 | [0.0176, 0.0298] | 0.8920 |
| gemma-4-e4b-it | refexp_poly | refcoco+_unc/testB | cIoU | 1000 | 0.0493 | [0.0342, 0.0654] | 0.8990 |
| gemma-4-e4b-it | refexp_poly | refcoco+_unc/testB | gIoU | 1000 | 0.0282 | [0.0206, 0.0353] | 0.8990 |
| gemma-4-e4b-it | refexp_poly | refcoco+_unc/val | cIoU | 1000 | 0.0373 | [0.0213, 0.0580] | 0.8930 |
| gemma-4-e4b-it | refexp_poly | refcoco+_unc/val | gIoU | 1000 | 0.0204 | [0.0145, 0.0270] | 0.8930 |
| gemma-4-e4b-it | refexp_poly | refcoco_unc/testA | cIoU | 1000 | 0.0494 | [0.0350, 0.0680] | 0.8750 |
| gemma-4-e4b-it | refexp_poly | refcoco_unc/testA | gIoU | 1000 | 0.0354 | [0.0281, 0.0430] | 0.8750 |
| gemma-4-e4b-it | refexp_poly | refcoco_unc/testB | cIoU | 1000 | 0.0601 | [0.0429, 0.0807] | 0.8830 |
| gemma-4-e4b-it | refexp_poly | refcoco_unc/testB | gIoU | 1000 | 0.0370 | [0.0285, 0.0461] | 0.8830 |
| gemma-4-e4b-it | refexp_poly | refcoco_unc/val | cIoU | 1000 | 0.0559 | [0.0405, 0.0723] | 0.8710 |
| gemma-4-e4b-it | refexp_poly | refcoco_unc/val | gIoU | 1000 | 0.0414 | [0.0321, 0.0507] | 0.8710 |
| gemma-4-e4b-it | refexp_poly | refcocog_umd/test | cIoU | 1000 | 0.0871 | [0.0711, 0.1057] | 0.7790 |
| gemma-4-e4b-it | refexp_poly | refcocog_umd/test | gIoU | 1000 | 0.0603 | [0.0513, 0.0707] | 0.7790 |
| gemma-4-e4b-it | refexp_poly | refcocog_umd/val | cIoU | 1000 | 0.0907 | [0.0718, 0.1140] | 0.8020 |
| gemma-4-e4b-it | refexp_poly | refcocog_umd/val | gIoU | 1000 | 0.0548 | [0.0457, 0.0650] | 0.8020 |
| gemma-4-e4b-it | grid16 | refcoco_unc/val | cIoU | 500 | 0.0904 | [0.0652, 0.1169] | 0.0000 |
| gemma-4-e4b-it | grid16 | refcoco_unc/val | gIoU | 500 | 0.0271 | [0.0196, 0.0353] | 0.0000 |

#### 解析失败（ParseTally，按原因）、截断率、生成 token 与墙钟（eval；compute.json 的墙钟只含 generate）

| 模型 | 任务 | n | 解析失败 n（率） | 按原因 | 截断率 | 平均生成 token | 生成 token 合计 | generate 墙钟 s |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 56857 | 1005（0.0177） | {"malformed": 2, "multiple_boxes": 824, "no_box": 4, "truncated": 175} | 0.0031 | 48.3 | 2746288 | 1481.1 |
| qwen3.5-0.8b | refexp_poly | 8000 | 8000（1.0000） | {"malformed": 7789, "truncated": 211} | 0.0264 | 85.6 | 685179 | 1446.0 |
| qwen3.5-0.8b | grid16 | 500 | 500（1.0000） | {"truncated": 500} | 1.0000 | 1024.0 | 512000 | 109.7 |
| qwen3.5-0.8b | count | 442 | 0（0.0000） | {} | 0.0000 | 2.4 | 1049 | 2.6 |
| qwen3.5-0.8b | ovd | 963 | 291（0.3022） | {"no_box": 7, "out_of_range": 3, "truncated": 281} | 0.2918 | 681.3 | 656110 | 465.3 |
| qwen3.5-0.8b | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 11.7 |
| qwen3.5-0.8b | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 0.4 |
| qwen3.5-0.8b | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 22.3 |
| qwen3.5-0.8b | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 13.5 |
| qwen3.5-2b | refexp_box | 56857 | 612（0.0108） | {"malformed": 2, "multiple_boxes": 536, "truncated": 74} | 0.0013 | 47.9 | 2721760 | 1374.6 |
| qwen3.5-2b | refexp_poly | 8000 | 7988（0.9985） | {"malformed": 7888, "no_polygon": 15, "truncated": 85} | 0.0106 | 53.7 | 429900 | 1034.2 |
| qwen3.5-2b | grid16 | 500 | 495（0.9900） | {"truncated": 492, "wrong_row_count": 3} | 0.9840 | 1012.0 | 506001 | 114.9 |
| qwen3.5-2b | count | 442 | 0（0.0000） | {} | 0.0000 | 2.1 | 922 | 2.8 |
| qwen3.5-2b | ovd | 963 | 143（0.1485） | {"inverted": 1, "out_of_range": 1, "truncated": 141} | 0.1464 | 414.9 | 399575 | 432.7 |
| qwen3.5-2b | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 17.9 |
| qwen3.5-2b | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 0.7 |
| qwen3.5-2b | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 24.9 |
| qwen3.5-2b | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 21.3 |
| qwen3.5-4b | refexp_box | 56857 | 1930（0.0339） | {"multiple_boxes": 1823, "no_box": 21, "truncated": 86} | 0.0015 | 48.7 | 2769743 | 2446.0 |
| qwen3.5-4b | refexp_poly | 8000 | 8000（1.0000） | {"ambiguous": 1, "malformed": 7966, "no_polygon": 6, "truncated": 27} | 0.0034 | 47.3 | 378080 | 940.6 |
| qwen3.5-4b | grid16 | 500 | 1（0.0020） | {"wrong_row_count": 1} | 0.0000 | 272.1 | 136034 | 48.6 |
| qwen3.5-4b | count | 442 | 0（0.0000） | {} | 0.0000 | 2.1 | 921 | 4.3 |
| qwen3.5-4b | ovd | 963 | 64（0.0665） | {"out_of_range": 1, "truncated": 63} | 0.0654 | 317.8 | 305999 | 733.6 |
| qwen3.5-4b | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 28.5 |
| qwen3.5-4b | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 1.1 |
| qwen3.5-4b | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 37.4 |
| qwen3.5-4b | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 34.3 |
| qwen3.5-9b | refexp_box | 56857 | 1056（0.0186） | {"malformed": 2, "multiple_boxes": 979, "no_box": 1, "truncated": 74} | 0.0013 | 47.8 | 2717053 | 2443.8 |
| qwen3.5-9b | refexp_poly | 8000 | 8000（1.0000） | {"ambiguous": 3, "malformed": 7925, "no_polygon": 11, "truncated": 61} | 0.0076 | 51.5 | 411804 | 1636.2 |
| qwen3.5-9b | grid16 | 500 | 77（0.1540） | {"wrong_row_count": 77} | 0.0000 | 274.7 | 137337 | 55.9 |
| qwen3.5-9b | count | 442 | 0（0.0000） | {} | 0.0000 | 2.1 | 925 | 5.6 |
| qwen3.5-9b | ovd | 963 | 55（0.0571） | {"out_of_range": 1, "truncated": 54} | 0.0561 | 294.1 | 283233 | 780.0 |
| qwen3.5-9b | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 37.0 |
| qwen3.5-9b | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 1.4 |
| qwen3.5-9b | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 40.8 |
| qwen3.5-9b | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 45.5 |
| internvl3.5-2b-hf | refexp_box | 56857 | 1（0.0000） | {"truncated": 1} | 0.0000 | 29.8 | 1693435 | 694.5 |
| internvl3.5-2b-hf | refexp_poly | 8000 | 1927（0.2409） | {"ambiguous": 259, "malformed": 227, "no_polygon": 2, "out_of_range": 6, "too_few_points": 1007, "truncated": 426} | 0.0532 | 265.0 | 2119626 | 3151.6 |
| internvl3.5-2b-hf | grid16 | 500 | 499（0.9980） | {"wrong_row_count": 499} | 0.0000 | 254.3 | 127153 | 45.8 |
| internvl3.5-2b-hf | count | 442 | 50（0.1131） | {"multiple_numbers": 45, "thinking_unclosed": 4, "truncated": 1} | 0.0090 | 22.7 | 10046 | 29.0 |
| internvl3.5-2b-hf | ovd | 963 | 864（0.8972） | {"no_box": 848, "truncated": 16} | 0.0166 | 200.0 | 192608 | 503.6 |
| internvl3.5-2b-hf | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 10.2 |
| internvl3.5-2b-hf | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 0.4 |
| internvl3.5-2b-hf | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 16.7 |
| internvl3.5-2b-hf | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 13.1 |
| internvl3.5-4b-hf | refexp_box | 56857 | 6（0.0001） | {"multiple_boxes": 2, "truncated": 4} | 0.0001 | 29.7 | 1689876 | 1202.4 |
| internvl3.5-4b-hf | refexp_poly | 8000 | 369（0.0461） | {"malformed": 49, "no_polygon": 6, "out_of_range": 1, "too_few_points": 7, "truncated": 306} | 0.0382 | 108.8 | 870093 | 5492.5 |
| internvl3.5-4b-hf | grid16 | 500 | 2（0.0040） | {"multiple_blocks": 2} | 0.0000 | 272.0 | 136000 | 79.2 |
| internvl3.5-4b-hf | count | 442 | 0（0.0000） | {} | 0.0000 | 8.1 | 3589 | 4.9 |
| internvl3.5-4b-hf | ovd | 963 | 100（0.1038） | {"no_box": 51, "truncated": 49} | 0.0509 | 357.6 | 344370 | 1122.2 |
| internvl3.5-4b-hf | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 17.2 |
| internvl3.5-4b-hf | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 0.6 |
| internvl3.5-4b-hf | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 23.2 |
| internvl3.5-4b-hf | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 22.6 |
| gemma-4-e2b-it | refexp_box | 56857 | 45281（0.7964） | {"empty": 2, "malformed": 342, "multiple_boxes": 577, "no_box": 40122, "out_of_range": 3, "truncated": 4235} | 0.0745 | 72.4 | 4117889 | 4788.5 |
| gemma-4-e2b-it | refexp_poly | 8000 | 99（0.0124） | {"malformed": 74, "no_polygon": 10, "too_few_points": 4, "truncated": 11} | 0.0014 | 73.7 | 589547 | 1113.4 |
| gemma-4-e2b-it | grid16 | 500 | 500（1.0000） | {"empty": 1, "wrong_row_count": 499} | 0.0000 | 248.0 | 124023 | 44.3 |
| gemma-4-e2b-it | count | 442 | 0（0.0000） | {} | 0.0000 | 2.1 | 946 | 4.7 |
| gemma-4-e2b-it | ovd | 963 | 592（0.6147） | {"no_box": 591, "truncated": 1} | 0.0010 | 120.7 | 116198 | 200.3 |
| gemma-4-e2b-it | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 32.9 |
| gemma-4-e2b-it | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 1.2 |
| gemma-4-e2b-it | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 37.4 |
| gemma-4-e2b-it | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 38.1 |
| gemma-4-e4b-it | refexp_box | 56857 | 42390（0.7456） | {"ambiguous": 2, "empty": 1, "malformed": 156, "multiple_boxes": 1368, "no_box": 39715, "truncated": 1148} | 0.0202 | 52.5 | 2986550 | 7770.3 |
| gemma-4-e4b-it | refexp_poly | 8000 | 6894（0.8618） | {"malformed": 6786, "no_polygon": 79, "too_few_points": 28, "truncated": 1} | 0.0001 | 68.3 | 546451 | 857.1 |
| gemma-4-e4b-it | grid16 | 500 | 0（0.0000） | {} | 0.0000 | 272.0 | 136000 | 88.0 |
| gemma-4-e4b-it | count | 442 | 0（0.0000） | {} | 0.0000 | 2.2 | 966 | 5.9 |
| gemma-4-e4b-it | ovd | 963 | 350（0.3634） | {"ambiguous": 5, "malformed": 1, "no_box": 344} | 0.0000 | 115.5 | 111190 | 350.9 |
| gemma-4-e4b-it | rel_depth_nyu | 3270 | 0（0.0000） | {} | 0.0000 | 1.0 | 3270 | 40.2 |
| gemma-4-e4b-it | rel_depth_blink | 112 | 0（0.0000） | {} | 0.0000 | 1.0 | 112 | 1.4 |
| gemma-4-e4b-it | spatial_cvbench | 2588 | 0（0.0000） | {} | 0.0000 | 1.0 | 2588 | 44.8 |
| gemma-4-e4b-it | spatial_embspatial | 3590 | 0（0.0000） | {} | 0.0000 | 1.0 | 3590 | 47.7 |

#### 拒答：定位、多边形、网格按 person 与非 person 分列（refusal_by_person）

| 模型 | 任务 | 指标 | 类 | n | 拒答 | 拒答率 | 有效 | 严格 | 条件 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | acc@0.5 | person | 26649 | 0 | 0.0000 | 26442 | 0.8509 | 0.8575 |
| qwen3.5-0.8b | refexp_box | acc@0.5 | non_person | 30208 | 0 | 0.0000 | 29410 | 0.7207 | 0.7403 |
| qwen3.5-0.8b | refexp_poly | cIoU | person | 3602 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-0.8b | refexp_poly | cIoU | non_person | 4398 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-0.8b | grid16 | gIoU | person | 250 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-0.8b | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-2b | refexp_box | acc@0.5 | person | 26649 | 0 | 0.0000 | 26529 | 0.8867 | 0.8908 |
| qwen3.5-2b | refexp_box | acc@0.5 | non_person | 30208 | 0 | 0.0000 | 29716 | 0.7816 | 0.7946 |
| qwen3.5-2b | refexp_poly | cIoU | person | 3602 | 0 | 0.0000 | 4 | 0.0003 | 0.5213 |
| qwen3.5-2b | refexp_poly | cIoU | non_person | 4398 | 0 | 0.0000 | 8 | 0.0006 | 0.1201 |
| qwen3.5-2b | grid16 | gIoU | person | 250 | 0 | 0.0000 | 1 | 0.0000 | 0.0000 |
| qwen3.5-2b | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 4 | 0.0010 | 0.0627 |
| qwen3.5-4b | refexp_box | acc@0.5 | person | 26649 | 0 | 0.0000 | 26097 | 0.8968 | 0.9158 |
| qwen3.5-4b | refexp_box | acc@0.5 | non_person | 30208 | 0 | 0.0000 | 28830 | 0.8038 | 0.8422 |
| qwen3.5-4b | refexp_poly | cIoU | person | 3602 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-4b | refexp_poly | cIoU | non_person | 4398 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-4b | grid16 | gIoU | person | 250 | 0 | 0.0000 | 249 | 0.0172 | 0.0173 |
| qwen3.5-4b | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 250 | 0.0114 | 0.0114 |
| qwen3.5-9b | refexp_box | acc@0.5 | person | 26649 | 0 | 0.0000 | 26405 | 0.9223 | 0.9308 |
| qwen3.5-9b | refexp_box | acc@0.5 | non_person | 30208 | 0 | 0.0000 | 29396 | 0.8270 | 0.8499 |
| qwen3.5-9b | refexp_poly | cIoU | person | 3602 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-9b | refexp_poly | cIoU | non_person | 4398 | 0 | 0.0000 | 0 | 0.0000 | — |
| qwen3.5-9b | grid16 | gIoU | person | 250 | 0 | 0.0000 | 194 | 0.0000 | 0.0000 |
| qwen3.5-9b | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 229 | 0.0031 | 0.0033 |
| internvl3.5-2b-hf | refexp_box | acc@0.5 | person | 26649 | 0 | 0.0000 | 26649 | 0.7633 | 0.7633 |
| internvl3.5-2b-hf | refexp_box | acc@0.5 | non_person | 30208 | 0 | 0.0000 | 30207 | 0.6650 | 0.6650 |
| internvl3.5-2b-hf | refexp_poly | cIoU | person | 3602 | 1 | 0.0003 | 2694 | 0.0709 | 0.0906 |
| internvl3.5-2b-hf | refexp_poly | cIoU | non_person | 4398 | 1 | 0.0002 | 3379 | 0.0807 | 0.1007 |
| internvl3.5-2b-hf | grid16 | gIoU | person | 250 | 0 | 0.0000 | 1 | 0.0000 | 0.0000 |
| internvl3.5-2b-hf | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 0 | 0.0000 | — |
| internvl3.5-4b-hf | refexp_box | acc@0.5 | person | 26649 | 0 | 0.0000 | 26649 | 0.8953 | 0.8953 |
| internvl3.5-4b-hf | refexp_box | acc@0.5 | non_person | 30208 | 0 | 0.0000 | 30202 | 0.8003 | 0.8004 |
| internvl3.5-4b-hf | refexp_poly | cIoU | person | 3602 | 5 | 0.0014 | 3509 | 0.1448 | 0.1493 |
| internvl3.5-4b-hf | refexp_poly | cIoU | non_person | 4398 | 1 | 0.0002 | 4122 | 0.2107 | 0.2252 |
| internvl3.5-4b-hf | grid16 | gIoU | person | 250 | 0 | 0.0000 | 248 | 0.0014 | 0.0015 |
| internvl3.5-4b-hf | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 250 | 0.0060 | 0.0060 |
| gemma-4-e2b-it | refexp_box | acc@0.5 | person | 26649 | 15531 | 0.5828 | 4554 | 0.1363 | 0.7978 |
| gemma-4-e2b-it | refexp_box | acc@0.5 | non_person | 30208 | 9148 | 0.3028 | 7022 | 0.1537 | 0.6611 |
| gemma-4-e2b-it | refexp_poly | cIoU | person | 3602 | 1 | 0.0003 | 3563 | 0.2416 | 0.2433 |
| gemma-4-e2b-it | refexp_poly | cIoU | non_person | 4398 | 0 | 0.0000 | 4338 | 0.2397 | 0.2416 |
| gemma-4-e2b-it | grid16 | gIoU | person | 250 | 0 | 0.0000 | 0 | 0.0000 | — |
| gemma-4-e2b-it | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 0 | 0.0000 | — |
| gemma-4-e4b-it | refexp_box | acc@0.5 | person | 26649 | 17009 | 0.6383 | 5548 | 0.1854 | 0.8904 |
| gemma-4-e4b-it | refexp_box | acc@0.5 | non_person | 30208 | 9069 | 0.3002 | 8919 | 0.2236 | 0.7573 |
| gemma-4-e4b-it | refexp_poly | cIoU | person | 3602 | 13 | 0.0036 | 494 | 0.0516 | 0.2923 |
| gemma-4-e4b-it | refexp_poly | cIoU | non_person | 4398 | 5 | 0.0011 | 612 | 0.0626 | 0.3209 |
| gemma-4-e4b-it | grid16 | gIoU | person | 250 | 0 | 0.0000 | 250 | 0.0256 | 0.0256 |
| gemma-4-e4b-it | grid16 | gIoU | non_person | 250 | 0 | 0.0000 | 250 | 0.0287 | 0.0287 |

#### free：自由作答 vs 同样本上的字母读数（eval）

| 模型 | 任务 | n | free 严格 [CI] | free 有效率 | 字母读数 严格 [CI]（同样本） | 差（free − 字母） | 配对：都对 / 只字母对 / 只 free 对 / 都错 | free 截断率 | free 平均生成 token |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | rel_depth_nyu | 200 | 0.6950 [0.6250, 0.7550] | 0.975 | 0.7650 [0.7000, 0.8250] | -0.0700 | 117 / 36 / 22 / 25 | 0.010 | 240.4 |
| qwen3.5-0.8b | rel_depth_blink | 112 | 0.6696 [0.5804, 0.7500] | 0.920 | 0.7589 [0.6786, 0.8393] | -0.0893 | 60 / 25 / 15 / 12 | 0.027 | 271.2 |
| qwen3.5-0.8b | spatial_cvbench | 200 | 0.7150 [0.6549, 0.7750] | 0.960 | 0.4250 [0.3550, 0.4900] | 0.2900 | 65 / 20 / 78 / 37 | 0.015 | 146.5 |
| qwen3.5-0.8b | spatial_embspatial | 200 | 0.5200 [0.4549, 0.5850] | 0.770 | 0.4800 [0.4150, 0.5500] | 0.0400 | 58 / 38 / 46 / 58 | 0.005 | 127.3 |
| qwen3.5-2b | rel_depth_nyu | 200 | 0.8250 [0.7700, 0.8750] | 0.990 | 0.7250 [0.6600, 0.7850] | 0.1000 | 122 / 23 / 43 / 12 | 0.005 | 198.8 |
| qwen3.5-2b | rel_depth_blink | 112 | 0.7857 [0.7054, 0.8571] | 0.991 | 0.5357 [0.4375, 0.6252] | 0.2500 | 51 / 9 / 37 / 15 | 0.009 | 211.0 |
| qwen3.5-2b | spatial_cvbench | 200 | 0.7550 [0.6900, 0.8150] | 0.955 | 0.5150 [0.4500, 0.5800] | 0.2400 | 88 / 15 / 63 / 34 | 0.025 | 165.7 |
| qwen3.5-2b | spatial_embspatial | 200 | 0.5350 [0.4700, 0.6050] | 0.725 | 0.5950 [0.5250, 0.6600] | -0.0600 | 87 / 32 / 20 / 61 | 0.005 | 120.5 |
| qwen3.5-4b | rel_depth_nyu | 200 | 0.8550 [0.8050, 0.9000] | 0.995 | 0.4500 [0.3800, 0.5151] | 0.4050 | 78 / 12 / 93 / 17 | 0.005 | 185.4 |
| qwen3.5-4b | rel_depth_blink | 112 | 0.8304 [0.7589, 0.8929] | 0.964 | 0.5179 [0.4283, 0.6161] | 0.3125 | 52 / 6 / 41 / 13 | 0.009 | 196.4 |
| qwen3.5-4b | spatial_cvbench | 200 | 0.8550 [0.8000, 0.9050] | 0.980 | 0.4550 [0.3900, 0.5250] | 0.4000 | 84 / 7 / 87 / 22 | 0.000 | 166.2 |
| qwen3.5-4b | spatial_embspatial | 200 | 0.5700 [0.5000, 0.6400] | 0.655 | 0.7550 [0.6900, 0.8150] | -0.1850 | 109 / 42 / 5 / 44 | 0.010 | 119.0 |
| qwen3.5-9b | rel_depth_nyu | 200 | 0.8400 [0.7850, 0.8900] | 0.985 | 0.9000 [0.8599, 0.9400] | -0.0600 | 161 / 19 / 7 / 13 | 0.010 | 171.5 |
| qwen3.5-9b | rel_depth_blink | 112 | 0.8304 [0.7589, 0.8929] | 0.991 | 0.8750 [0.8125, 0.9286] | -0.0446 | 89 / 9 / 4 / 10 | 0.009 | 186.3 |
| qwen3.5-9b | spatial_cvbench | 200 | 0.8400 [0.7850, 0.8900] | 0.985 | 0.7700 [0.7150, 0.8250] | 0.0700 | 143 / 11 / 25 / 21 | 0.000 | 145.2 |
| qwen3.5-9b | spatial_embspatial | 200 | 0.6600 [0.5950, 0.7201] | 0.805 | 0.8100 [0.7550, 0.8601] | -0.1500 | 125 / 37 / 7 / 31 | 0.005 | 141.9 |
| internvl3.5-2b-hf | rel_depth_nyu | 200 | 0.6100 [0.5450, 0.6750] | 0.980 | 0.5700 [0.5050, 0.6350] | 0.0400 | 60 / 54 / 62 / 24 | 0.000 | 165.8 |
| internvl3.5-2b-hf | rel_depth_blink | 112 | 0.5804 [0.4911, 0.6696] | 0.982 | 0.6161 [0.5266, 0.7054] | -0.0357 | 31 / 38 / 34 / 9 | 0.000 | 202.3 |
| internvl3.5-2b-hf | spatial_cvbench | 200 | 0.7000 [0.6350, 0.7600] | 0.925 | 0.6850 [0.6200, 0.7450] | 0.0150 | 108 / 29 / 32 / 31 | 0.000 | 149.9 |
| internvl3.5-2b-hf | spatial_embspatial | 200 | 0.3600 [0.2950, 0.4251] | 0.620 | 0.5050 [0.4350, 0.5800] | -0.1450 | 59 / 42 / 13 / 86 | 0.015 | 136.0 |
| internvl3.5-4b-hf | rel_depth_nyu | 200 | 0.7500 [0.6900, 0.8100] | 0.995 | 0.7850 [0.7250, 0.8400] | -0.0350 | 136 / 21 / 14 / 29 | 0.000 | 135.7 |
| internvl3.5-4b-hf | rel_depth_blink | 112 | 0.7321 [0.6518, 0.8125] | 1.000 | 0.7857 [0.7054, 0.8571] | -0.0536 | 69 / 19 / 13 / 11 | 0.000 | 223.8 |
| internvl3.5-4b-hf | spatial_cvbench | 200 | 0.7700 [0.7150, 0.8250] | 0.980 | 0.7000 [0.6350, 0.7601] | 0.0700 | 127 / 13 / 27 / 33 | 0.000 | 117.5 |
| internvl3.5-4b-hf | spatial_embspatial | 200 | 0.7100 [0.6500, 0.7750] | 0.930 | 0.7600 [0.7000, 0.8200] | -0.0500 | 141 / 11 / 1 / 47 | 0.000 | 31.4 |
| gemma-4-e2b-it | rel_depth_nyu | 200 | 0.7050 [0.6400, 0.7651] | 0.980 | 0.4450 [0.3750, 0.5150] | 0.2600 | 68 / 21 / 73 / 38 | 0.000 | 263.8 |
| gemma-4-e2b-it | rel_depth_blink | 112 | 0.7232 [0.6339, 0.8036] | 0.929 | 0.5357 [0.4464, 0.6250] | 0.1875 | 49 / 11 / 32 / 20 | 0.000 | 256.0 |
| gemma-4-e2b-it | spatial_cvbench | 200 | 0.6750 [0.6100, 0.7400] | 0.900 | 0.4050 [0.3400, 0.4750] | 0.2700 | 56 / 25 / 79 / 40 | 0.005 | 172.1 |
| gemma-4-e2b-it | spatial_embspatial | 200 | 0.1150 [0.0700, 0.1600] | 0.200 | 0.2400 [0.1850, 0.3000] | -0.1250 | 4 / 44 / 19 / 133 | 0.010 | 340.7 |
| gemma-4-e4b-it | rel_depth_nyu | 200 | 0.8050 [0.7500, 0.8550] | 0.965 | 0.4800 [0.4100, 0.5450] | 0.3250 | 83 / 13 / 78 / 26 | 0.000 | 253.6 |
| gemma-4-e4b-it | rel_depth_blink | 112 | 0.7500 [0.6696, 0.8304] | 0.991 | 0.5268 [0.4286, 0.6250] | 0.2232 | 43 / 16 / 41 / 12 | 0.000 | 168.9 |
| gemma-4-e4b-it | spatial_cvbench | 200 | 0.6550 [0.5950, 0.7151] | 0.870 | 0.5200 [0.4550, 0.5900] | 0.1350 | 82 / 22 / 49 / 47 | 0.000 | 141.5 |
| gemma-4-e4b-it | spatial_embspatial | 200 | 0.3300 [0.2700, 0.3950] | 0.415 | 0.4700 [0.4000, 0.5450] | -0.1400 | 44 / 50 / 22 / 84 | 0.000 | 199.8 |

#### extra：附加口径 vs 同样本上的主口径（eval）

| 模型 | 口径 | 任务 | 指标（范围） | n | extra 严格 [CI] | extra 有效率 | 主口径 严格 [CI]（同样本） | 主口径有效率 | 差（extra − 主口径） | extra 视觉 token 均值 / 主口径 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | qwen-1024 | ovd | F1@0.5（coco_val2017） | 100 | 0.2426 [0.1788, 0.3173] | 0.540 | 0.1960 [0.1417, 0.2623] | 0.520 | 0.0466 | 1055.5 / 313.6 |
| qwen3.5-0.8b | qwen-1024 | refexp_box | acc@0.5（all） | 500 | 0.8360 [0.8020, 0.8680] | 0.980 | 0.8160 [0.7820, 0.8520] | 0.988 | 0.0200 | 1052.6 / 309.8 |
| qwen3.5-9b | qwen-1024 | ovd | F1@0.5（coco_val2017） | 100 | 0.5825 [0.5118, 0.6503] | 0.930 | 0.5417 [0.4756, 0.6141] | 0.880 | 0.0409 | 1055.5 / 313.6 |
| qwen3.5-9b | qwen-1024 | refexp_box | acc@0.5（all） | 500 | 0.9020 [0.8740, 0.9280] | 0.984 | 0.9080 [0.8820, 0.9320] | 0.986 | -0.0060 | 1052.6 / 309.8 |
| internvl3.5-2b-hf | internvl-native | ovd | F1@0.5（coco_val2017） | 100 | 0.0000 [0.0000, 0.0000] | 0.000 | 0.0000 [0.0000, 0.0000] | 0.000 | 0.0000 | 2391.0 / 256.0 |
| internvl3.5-2b-hf | internvl-native | refexp_box | acc@0.5（all） | 500 | 0.8720 [0.8400, 0.9020] | 1.000 | 0.7220 [0.6820, 0.7580] | 1.000 | 0.1500 | 2467.8 / 256.0 |
| internvl3.5-4b-hf | internvl-native | ovd | F1@0.5（coco_val2017） | 100 | 0.1151 [0.0873, 0.1437] | 0.800 | 0.1166 [0.0906, 0.1421] | 0.880 | -0.0015 | 2391.0 / 256.0 |
| internvl3.5-4b-hf | internvl-native | refexp_box | acc@0.5（all） | 500 | 0.9240 [0.9000, 0.9460] | 1.000 | 0.8940 [0.8659, 0.9200] | 1.000 | 0.0300 | 2467.8 / 256.0 |

#### 坐标诊断（D22-2）：全部坐标 ≤ 1 且含小数的条数（diagnostics.coord_all_le1_with_decimal）

| 模型 | 任务 | 子集 | 条数 | 比例 |
|---|---|---|---|---|
| qwen3.5-0.8b | ovd | eval | 0 | 0.00000 |
| qwen3.5-0.8b | ovd | extra__qwen-1024 | 0 | 0.00000 |
| qwen3.5-0.8b | refexp_box | eval | 0 | 0.00000 |
| qwen3.5-0.8b | refexp_box | extra__qwen-1024 | 0 | 0.00000 |
| qwen3.5-0.8b | refexp_poly | eval | 0 | 0.00000 |
| qwen3.5-2b | ovd | eval | 0 | 0.00000 |
| qwen3.5-2b | refexp_box | eval | 0 | 0.00000 |
| qwen3.5-2b | refexp_poly | eval | 0 | 0.00000 |
| qwen3.5-4b | ovd | eval | 0 | 0.00000 |
| qwen3.5-4b | refexp_box | eval | 0 | 0.00000 |
| qwen3.5-4b | refexp_poly | eval | 0 | 0.00000 |
| qwen3.5-9b | ovd | eval | 0 | 0.00000 |
| qwen3.5-9b | ovd | extra__qwen-1024 | 0 | 0.00000 |
| qwen3.5-9b | refexp_box | eval | 0 | 0.00000 |
| qwen3.5-9b | refexp_box | extra__qwen-1024 | 0 | 0.00000 |
| qwen3.5-9b | refexp_poly | eval | 0 | 0.00000 |
| internvl3.5-2b-hf | ovd | eval | 0 | 0.00000 |
| internvl3.5-2b-hf | ovd | extra__internvl-native | 0 | 0.00000 |
| internvl3.5-2b-hf | refexp_box | eval | 0 | 0.00000 |
| internvl3.5-2b-hf | refexp_box | extra__internvl-native | 0 | 0.00000 |
| internvl3.5-2b-hf | refexp_poly | eval | 0 | 0.00000 |
| internvl3.5-4b-hf | ovd | eval | 0 | 0.00000 |
| internvl3.5-4b-hf | ovd | extra__internvl-native | 0 | 0.00000 |
| internvl3.5-4b-hf | refexp_box | eval | 0 | 0.00000 |
| internvl3.5-4b-hf | refexp_box | extra__internvl-native | 0 | 0.00000 |
| internvl3.5-4b-hf | refexp_poly | eval | 0 | 0.00000 |
| gemma-4-e2b-it | ovd | eval | 0 | 0.00000 |
| gemma-4-e2b-it | refexp_box | eval | 0 | 0.00000 |
| gemma-4-e2b-it | refexp_poly | eval | 0 | 0.00000 |
| gemma-4-e4b-it | ovd | eval | 0 | 0.00000 |
| gemma-4-e4b-it | refexp_box | eval | 3 | 0.00005 |
| gemma-4-e4b-it | refexp_poly | eval | 0 | 0.00000 |

#### 用量：每个模型每部分的墙钟（驱动日志里每条 run_e1 命令的起止，含加载与预热；GPU 小时）与预注册 §9 的估算

| 模型 | eval 实际 / 估算 | free 实际 / 估算 | extra 实际 / 估算 | 合计实际 / 估算 | rc（eval / free / extra） |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 1.11 / 1.21 | 0.05 / 0.10 | 0.03 / 0.06 | 1.18 / 1.37 | 0 / 0 / 0 |
| qwen3.5-2b | 0.95 / 1.00 | 0.05 / 0.10 | — / 0.00 | 1.00 / 1.10 | 0 / 0 / — |
| qwen3.5-4b | 1.30 / 3.13 | 0.07 / 0.14 | — / 0.00 | 1.37 / 3.27 | 0 / 0 / — |
| qwen3.5-9b | 1.53 / 2.34 | 0.07 / 0.12 | 0.07 / 0.09 | 1.66 / 2.55 | 0 / 0 / 0 |
| internvl3.5-2b-hf | 1.33 / 1.69 | 0.05 / 0.07 | 0.07 / 0.11 | 1.46 / 1.87 | 0 / 0 / 0 |
| internvl3.5-4b-hf | 2.31 / 2.68 | 0.04 / 0.08 | 0.21 / 0.24 | 2.56 / 3.00 | 0 / 0 / 0 |
| gemma-4-e2b-it | 1.86 / 1.93 | 0.07 / 0.18 | — / 0.00 | 1.93 / 2.11 | 0 / 0 / — |
| gemma-4-e4b-it | 2.68 / 3.06 | 0.11 / 0.21 | — / 0.00 | 2.79 / 3.27 | 0 / 0 / — |
| **合计（已完成的模型）** | | | | **13.95 / 18.54** | |
