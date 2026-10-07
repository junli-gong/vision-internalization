# 去重报告（硬约束 7；D11-7）

- 生成：2026-10-07T04:23:37+00:00，git 6e77ab064d
- 感知哈希阈值：汉明距离 ≤ 8（src/data/constants.PHASH_MAX_HAMMING）

## 输入

- train-root /dev/shm/visint/train_data（host metallic-matrix-fin-03，2026-10-07T04:23:36+00:00）
- eval experiments/P0/0023_eval_hashes/eval_hashes_b200.json（host metallic-matrix-fin-03，2026-10-02T13:28:53+00:00）

## 集合

| 集合 | 角色 | 图像数 | 有感知哈希 |
|---|---|---:|---:|
| ade20k/train | train | 20210 | 20210 |
| ade20k/val | eval | 2000 | 2000 |
| blink_reldepth/val | eval | 124 | 124 |
| coco_val2017/e1sub500 | eval | 500 | 500 |
| coco_val2017_full/val2017 | eval | 5000 | 0 |
| countbench/eval | eval | 491 | 491 |
| cvbench/test | eval | 2638 | 2638 |
| embspatial/test | eval | 3640 | 3640 |
| nyu_depth_train/train | train | 22848 | 22848 |
| nyu_normals/test | eval | 654 | 654 |
| nyu_normals/train | train | 795 | 795 |
| nyuv2/dev_train | eval | 50 | 50 |
| nyuv2/test | eval | 654 | 654 |
| odinw13/eval | eval | 463 | 463 |
| probe/D20 | eval | 2 | 2 |
| refcoco/eval | eval | 6549 | 6549 |
| refcoco_train/train | train | 24407 | 24407 |

## D11-1、D11-2 过滤前后

- D11-1（RefCOCO 训练图）：三者 train 图像并集 28158，评测划分图像并集 6549，剔除 3751，保留 24407；交叉 {'refcoco train ∩ refcoco eval': 0, 'refcoco train ∩ refcoco+ eval': 0, 'refcoco train ∩ refcocog eval': 1891, 'refcoco+ train ∩ refcoco eval': 0, 'refcoco+ train ∩ refcoco+ eval': 0, 'refcoco+ train ∩ refcocog eval': 1891, 'refcocog train ∩ refcoco eval': 1860, 'refcocog train ∩ refcoco+ eval': 1860, 'refcocog train ∩ refcocog eval': 0}
- D11-2（NYU 深度训练集）：BTS 清单 24231 对，按场景去掉 1383，保留 22848；去掉的目录 {'bookstore_0001d': 242, 'bookstore_0001e': 158, 'bookstore_0001f': 348, 'bookstore_0001g': 189, 'bookstore_0001h': 119, 'bookstore_0001i': 114, 'bookstore_0001j': 213}

## 每对集合的重合

| 训练集 | 评测集 | 同键 | ≤0 的训练图 | ≤2 的训练图 | ≤4 的训练图 | ≤6 的训练图 | ≤8 的训练图 | ≤10 的训练图 | ≤12 的训练图 | 剔除 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ade20k/train | ade20k/val | 0 | 13 | 14 | 14 | 16 | 20 | 35 | 188 | 20 |
| ade20k/train | blink_reldepth/val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| ade20k/train | coco_val2017/e1sub500 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 36 | 1 |
| ade20k/train | coco_val2017_full/val2017 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ade20k/train | countbench/eval | 0 | 0 | 0 | 0 | 0 | 2 | 4 | 38 | 2 |
| ade20k/train | cvbench/test | 0 | 2 | 2 | 2 | 2 | 6 | 20 | 178 | 6 |
| ade20k/train | embspatial/test | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 108 | 0 |
| ade20k/train | nyu_normals/test | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 33 | 0 |
| ade20k/train | nyuv2/dev_train | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| ade20k/train | nyuv2/test | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 33 | 0 |
| ade20k/train | odinw13/eval | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 22 | 0 |
| ade20k/train | probe/D20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ade20k/train | refcoco/eval | 0 | 0 | 0 | 0 | 0 | 1 | 28 | 281 | 1 |
| nyu_depth_train/train | ade20k/val | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 145 | 0 |
| nyu_depth_train/train | blink_reldepth/val | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 9 | 0 |
| nyu_depth_train/train | coco_val2017/e1sub500 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 30 | 0 |
| nyu_depth_train/train | coco_val2017_full/val2017 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nyu_depth_train/train | countbench/eval | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 47 | 0 |
| nyu_depth_train/train | cvbench/test | 0 | 0 | 0 | 0 | 0 | 1 | 13 | 145 | 1 |
| nyu_depth_train/train | embspatial/test | 0 | 0 | 0 | 0 | 0 | 1 | 10 | 128 | 1 |
| nyu_depth_train/train | nyu_normals/test | 0 | 0 | 0 | 0 | 0 | 1 | 26 | 166 | 1 |
| nyu_depth_train/train | nyuv2/dev_train | 0 | 13 | 43 | 68 | 95 | 126 | 161 | 199 | 126 |
| nyu_depth_train/train | nyuv2/test | 0 | 0 | 0 | 0 | 0 | 1 | 26 | 168 | 1 |
| nyu_depth_train/train | odinw13/eval | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 32 | 0 |
| nyu_depth_train/train | probe/D20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nyu_depth_train/train | refcoco/eval | 0 | 0 | 0 | 0 | 1 | 3 | 24 | 373 | 3 |
| nyu_normals/train | ade20k/val | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 6 | 0 |
| nyu_normals/train | blink_reldepth/val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nyu_normals/train | coco_val2017/e1sub500 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 |
| nyu_normals/train | coco_val2017_full/val2017 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nyu_normals/train | countbench/eval | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| nyu_normals/train | cvbench/test | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 9 | 0 |
| nyu_normals/train | embspatial/test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| nyu_normals/train | nyu_normals/test | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 8 | 0 |
| nyu_normals/train | nyuv2/dev_train | 50 | 44 | 47 | 50 | 50 | 50 | 50 | 50 | 50 |
| nyu_normals/train | nyuv2/test | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 8 | 0 |
| nyu_normals/train | odinw13/eval | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nyu_normals/train | probe/D20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nyu_normals/train | refcoco/eval | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 0 |
| refcoco_train/train | ade20k/val | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 141 | 0 |
| refcoco_train/train | blink_reldepth/val | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 |
| refcoco_train/train | coco_val2017/e1sub500 | 0 | 0 | 1 | 1 | 1 | 1 | 3 | 35 | 1 |
| refcoco_train/train | coco_val2017_full/val2017 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| refcoco_train/train | countbench/eval | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 28 | 0 |
| refcoco_train/train | cvbench/test | 0 | 0 | 1 | 1 | 1 | 2 | 13 | 145 | 2 |
| refcoco_train/train | embspatial/test | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 95 | 0 |
| refcoco_train/train | nyu_normals/test | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 37 | 0 |
| refcoco_train/train | nyuv2/dev_train | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| refcoco_train/train | nyuv2/test | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 38 | 0 |
| refcoco_train/train | odinw13/eval | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 28 | 0 |
| refcoco_train/train | probe/D20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| refcoco_train/train | refcoco/eval | 0 | 1 | 2 | 2 | 4 | 8 | 43 | 401 | 8 |

### 例子（每对最多 10 个）

- ade20k/train × ade20k/val：同键 []；感知哈希 ade20k:ADE_train_00000399 ~ ade20k:ADE_val_00000045（0），ade20k:ADE_train_00001912 ~ ade20k:ADE_val_00000049（0），ade20k:ADE_train_00001936 ~ ade20k:ADE_val_00000048（0），ade20k:ADE_train_00002880 ~ ade20k:ADE_val_00001164（0），ade20k:ADE_train_00002932 ~ ade20k:ADE_val_00001166（0），ade20k:ADE_train_00004117 ~ ade20k:ADE_val_00000162（0），ade20k:ADE_train_00006244 ~ ade20k:ADE_val_00001278（0），ade20k:ADE_train_00009791 ~ ade20k:ADE_val_00000440（0），ade20k:ADE_train_00009795 ~ ade20k:ADE_val_00001440（0），ade20k:ADE_train_00011248 ~ ade20k:ADE_val_00001516（0）
- ade20k/train × blink_reldepth/val：同键 []；感知哈希 ade20k:ADE_train_00007886 ~ blink:val_Relative_Depth_66（12），ade20k:ADE_train_00009047 ~ blink:val_Relative_Depth_115（12），ade20k:ADE_train_00011470 ~ blink:val_Relative_Depth_118（12），ade20k:ADE_train_00015258 ~ blink:val_Relative_Depth_94（12），ade20k:ADE_train_00016393 ~ blink:val_Relative_Depth_89（12），ade20k:ADE_train_00017559 ~ blink:val_Relative_Depth_72（12）
- ade20k/train × coco_val2017/e1sub500：同键 []；感知哈希 ade20k:ADE_train_00002187 ~ coco:398652（8），ade20k:ADE_train_00003355 ~ coco:456143（10），ade20k:ADE_train_00012474 ~ coco:553094（10），ade20k:ADE_train_00019140 ~ coco:134856（10），ade20k:ADE_train_00000120 ~ coco:534664（12），ade20k:ADE_train_00002679 ~ coco:522638（12），ade20k:ADE_train_00003324 ~ coco:577539（12），ade20k:ADE_train_00003541 ~ coco:557501（12），ade20k:ADE_train_00003938 ~ coco:66635（12），ade20k:ADE_train_00004639 ~ coco:338624（12）
- ade20k/train × countbench/eval：同键 []；感知哈希 ade20k:ADE_train_00012059 ~ countbench:363（8），ade20k:ADE_train_00014453 ~ countbench:224（8），ade20k:ADE_train_00005765 ~ countbench:203（10），ade20k:ADE_train_00014126 ~ countbench:266（10），ade20k:ADE_train_00001486 ~ countbench:215（12），ade20k:ADE_train_00004019 ~ countbench:441（12），ade20k:ADE_train_00005018 ~ countbench:414（12），ade20k:ADE_train_00005603 ~ countbench:438（12），ade20k:ADE_train_00006091 ~ countbench:521（12），ade20k:ADE_train_00006128 ~ countbench:521（12）
- ade20k/train × cvbench/test：同键 []；感知哈希 ade20k:ADE_train_00011349 ~ ade20k:ADE_val_00000159（0），ade20k:ADE_train_00017026 ~ ade20k:ADE_val_00001839（0），ade20k:ADE_train_00000329 ~ ade20k:ADE_val_00001005（8），ade20k:ADE_train_00002187 ~ coco:398652（8），ade20k:ADE_train_00005507 ~ coco:205647（8），ade20k:ADE_train_00018718 ~ nuscenes:n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151270012404.jpg（8），ade20k:ADE_train_00000303 ~ nyuv2:211（10），ade20k:ADE_train_00000949 ~ ade20k:ADE_val_00001144（10），ade20k:ADE_train_00001647 ~ nuscenes:n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400201912460.jpg（10），ade20k:ADE_train_00003803 ~ ade20k:ADE_val_00000174（10）
- ade20k/train × embspatial/test：同键 []；感知哈希 ade20k:ADE_train_00006960 ~ embspatial:mp3d_355（10），ade20k:ADE_train_00013907 ~ embspatial:mp3d_275（10），ade20k:ADE_train_00014679 ~ embspatial:mp3d_417（10），ade20k:ADE_train_00000008 ~ embspatial:mp3d_333（12），ade20k:ADE_train_00000186 ~ embspatial:mp3d_981（12），ade20k:ADE_train_00000195 ~ embspatial:ai2thor_532（12），ade20k:ADE_train_00000422 ~ embspatial:ai2thor_1114（12），ade20k:ADE_train_00000529 ~ embspatial:mp3d_76（12），ade20k:ADE_train_00000558 ~ embspatial:scannet_488（12），ade20k:ADE_train_00000955 ~ embspatial:mp3d_969（12）
- ade20k/train × nyu_normals/test：同键 []；感知哈希 ade20k:ADE_train_00005475 ~ nyuv2:134（10），ade20k:ADE_train_00013990 ~ nyuv2:908（10），ade20k:ADE_train_00000733 ~ nyuv2:525（12），ade20k:ADE_train_00001600 ~ nyuv2:1032（12），ade20k:ADE_train_00003161 ~ nyuv2:41（12），ade20k:ADE_train_00003174 ~ nyuv2:520（12），ade20k:ADE_train_00003740 ~ nyuv2:153（12），ade20k:ADE_train_00004436 ~ nyuv2:47（12），ade20k:ADE_train_00004982 ~ nyuv2:592（12），ade20k:ADE_train_00005440 ~ nyuv2:1180（12）
- ade20k/train × nyuv2/dev_train：同键 []；感知哈希 ade20k:ADE_train_00002565 ~ nyuv2:905（12），ade20k:ADE_train_00019229 ~ nyuv2:158（12），ade20k:ADE_train_00019747 ~ nyuv2:51（12），ade20k:ADE_train_00020073 ~ nyuv2:1359（12）
- ade20k/train × nyuv2/test：同键 []；感知哈希 ade20k:ADE_train_00005475 ~ nyuv2:134（10），ade20k:ADE_train_00013990 ~ nyuv2:908（10），ade20k:ADE_train_00000733 ~ nyuv2:525（12），ade20k:ADE_train_00001600 ~ nyuv2:1032（12），ade20k:ADE_train_00003174 ~ nyuv2:520（12），ade20k:ADE_train_00003740 ~ nyuv2:153（12），ade20k:ADE_train_00003871 ~ nyuv2:1306（12），ade20k:ADE_train_00003892 ~ nyuv2:634（12），ade20k:ADE_train_00004982 ~ nyuv2:592（12），ade20k:ADE_train_00005440 ~ nyuv2:1180（12）
- ade20k/train × odinw13/eval：同键 []；感知哈希 ade20k:ADE_train_00007335 ~ odinw:pothole:img-122_jpg.rf.90260f1c979ae127d37bb26cf5d1b3cf.jpg（10），ade20k:ADE_train_00017224 ~ odinw:EgoHands_generic:JENGA_COURTYARD_B_H_frame_1292_jpg.rf.351ba7a0847d4f2d84ed78ed737ea44f.jpg（10），ade20k:ADE_train_00000436 ~ odinw:PascalVOC:2007_005107_jpg.rf.11eea282ed3d574e57da6e5e28f74272.jpg（12），ade20k:ADE_train_00001244 ~ odinw:pothole:img-24_jpg.rf.42018f151db0049eda0c504eabced76b.jpg（12），ade20k:ADE_train_00001939 ~ odinw:PascalVOC:2010_005978_jpg.rf.9536027060d0234e81830df817d02a41.jpg（12），ade20k:ADE_train_00003841 ~ odinw:PascalVOC:2008_008615_jpg.rf.ea3861885f297db57ee7b000f1d27b11.jpg（12），ade20k:ADE_train_00004255 ~ odinw:VehiclesOpenImages_416x416:07d61c5699b6f4da_jpg.rf.c91a31a6ac5fcc72f98ba6b6e7585168.jpg（12），ade20k:ADE_train_00004501 ~ odinw:AerialMaritimeDrone_large:DJI_0263_JPG.rf.46456e1604da41debe69ed67f7486ed2.jpg（12），ade20k:ADE_train_00008667 ~ odinw:Raccoon_Raccoon.v2-raw.coco:raccoon-92_jpg.rf.273cd8aa0ccf4882d34914089efaae28.jpg（12），ade20k:ADE_train_00008710 ~ odinw:Raccoon_Raccoon.v2-raw.coco:raccoon-99_jpg.rf.df3f011940cffad5dd115620bd324583.jpg（12）
- ade20k/train × refcoco/eval：同键 []；感知哈希 ade20k:ADE_train_00003154 ~ coco:189646（8），ade20k:ADE_train_00000694 ~ coco:213005（10），ade20k:ADE_train_00001574 ~ coco:127388（10），ade20k:ADE_train_00002092 ~ coco:442641（10），ade20k:ADE_train_00003700 ~ coco:572960（10），ade20k:ADE_train_00003921 ~ coco:313437（10），ade20k:ADE_train_00005521 ~ coco:307475（10），ade20k:ADE_train_00006108 ~ coco:346250（10），ade20k:ADE_train_00006936 ~ coco:265766（10），ade20k:ADE_train_00007709 ~ coco:581563（10）
- nyu_depth_train/train × ade20k/val：同键 []；感知哈希 nyuraw:bathroom_0010/00013 ~ ade20k:ADE_val_00001947（10），nyuraw:bedroom_0050/00004 ~ ade20k:ADE_val_00000486（10），nyuraw:bedroom_0056a/00059 ~ ade20k:ADE_val_00000119（10），nyuraw:home_office_0005/00132 ~ ade20k:ADE_val_00000727（10），nyuraw:kitchen_0028a/00172 ~ ade20k:ADE_val_00001865（10），nyuraw:living_room_0058/00008 ~ ade20k:ADE_val_00000660（10），nyuraw:living_room_0071/00001 ~ ade20k:ADE_val_00000983（10），nyuraw:office_0006/00109 ~ ade20k:ADE_val_00001132（10），nyuraw:bathroom_0013/00043 ~ ade20k:ADE_val_00000036（12），nyuraw:bathroom_0014a/00018 ~ ade20k:ADE_val_00000577（12）
- nyu_depth_train/train × blink_reldepth/val：同键 []；感知哈希 nyuraw:living_room_0022/00336 ~ blink:val_Relative_Depth_85（10），nyuraw:bathroom_0051/00033 ~ blink:val_Relative_Depth_54（12），nyuraw:dining_room_0001b/00038 ~ blink:val_Relative_Depth_81（12），nyuraw:kitchen_0010/00024 ~ blink:val_Relative_Depth_98（12），nyuraw:living_room_0022/00335 ~ blink:val_Relative_Depth_85（12），nyuraw:living_room_0070/00021 ~ blink:val_Relative_Depth_83（12），nyuraw:living_room_0070/00022 ~ blink:val_Relative_Depth_83（12），nyuraw:student_lounge_0001/00119 ~ blink:val_Relative_Depth_98（12），nyuraw:student_lounge_0001/00121 ~ blink:val_Relative_Depth_98（12）
- nyu_depth_train/train × coco_val2017/e1sub500：同键 []；感知哈希 nyuraw:bathroom_0024/00056 ~ coco:802（10），nyuraw:reception_room_0001a/00051 ~ coco:447465（10），nyuraw:bathroom_0007/00120 ~ coco:426297（12），nyuraw:bedroom_0010/00105 ~ coco:86956（12），nyuraw:bedroom_0010/00106 ~ coco:86956（12），nyuraw:bedroom_0010/00107 ~ coco:86956（12），nyuraw:bedroom_0014/00073 ~ coco:189828（12），nyuraw:bedroom_0051/00100 ~ coco:312340（12），nyuraw:bedroom_0052/00089 ~ coco:350054（12），nyuraw:bedroom_0098/00046 ~ coco:508730（12）
- nyu_depth_train/train × countbench/eval：同键 []；感知哈希 nyuraw:bathroom_0001/00006 ~ countbench:230（10），nyuraw:bathroom_0028/00031 ~ countbench:311（10），nyuraw:bathroom_0035/00031 ~ countbench:122（10），nyuraw:bedroom_0096/00000 ~ countbench:26（10），nyuraw:conference_room_0002/00052 ~ countbench:131（10），nyuraw:bathroom_0001/00007 ~ countbench:230（12），nyuraw:bathroom_0016/00020 ~ countbench:22（12），nyuraw:bathroom_0042/00008 ~ countbench:18（12），nyuraw:bedroom_0004/00192 ~ countbench:326（12），nyuraw:bedroom_0019/00091 ~ countbench:422（12）
- nyu_depth_train/train × cvbench/test：同键 []；感知哈希 nyuraw:dining_room_0013/00077 ~ coco:79229（8），nyuraw:bathroom_0006/00056 ~ coco:267537（10），nyuraw:bathroom_0024/00020 ~ nyuv2:1443（10），nyuraw:bedroom_0010/00118 ~ hypersim:hypersim/ai_048_010/images/scene_cam_00_final_preview/frame.0018.tonemap.jpg（10），nyuraw:bedroom_0010/00119 ~ hypersim:hypersim/ai_048_010/images/scene_cam_00_final_preview/frame.0018.tonemap.jpg（10），nyuraw:bedroom_0021/00092 ~ ade20k:ADE_val_00001124（10），nyuraw:dining_room_0019/00096 ~ coco:475904（10），nyuraw:home_office_0005/00132 ~ ade20k:ADE_val_00000727（10），nyuraw:kitchen_0028a/00172 ~ ade20k:ADE_val_00001865（10），nyuraw:kitchen_0033/00143 ~ ade20k:ADE_val_00001251（10）
- nyu_depth_train/train × embspatial/test：同键 []；感知哈希 nyuraw:bedroom_0086/00040 ~ embspatial:mp3d_475（8），nyuraw:bathroom_0010/00015 ~ embspatial:ai2thor_1004（10），nyuraw:bedroom_0020/00117 ~ embspatial:scannet_1028（10），nyuraw:dining_room_0012/00040 ~ embspatial:ai2thor_58（10），nyuraw:dining_room_0014/00094 ~ embspatial:ai2thor_595（10），nyuraw:dining_room_0031/00215 ~ embspatial:scannet_1061（10），nyuraw:dining_room_0031/00221 ~ embspatial:scannet_1061（10），nyuraw:kitchen_0006/00014 ~ embspatial:ai2thor_606（10），nyuraw:living_room_0022/00378 ~ embspatial:ai2thor_1172（10），nyuraw:living_room_0058/00026 ~ embspatial:scannet_862（10）
- nyu_depth_train/train × nyu_normals/test：同键 []；感知哈希 nyuraw:bedroom_0126/00030 ~ nyuv2:1166（8），nyuraw:bathroom_0053/00039 ~ nyuv2:280（10），nyuraw:bathroom_0054/00019 ~ nyuv2:782（10），nyuraw:bedroom_0004/00093 ~ nyuv2:1226（10），nyuraw:bedroom_0004/00095 ~ nyuv2:1226（10），nyuraw:bedroom_0004/00130 ~ nyuv2:1226（10），nyuraw:bedroom_0051/00070 ~ nyuv2:1289（10），nyuraw:bedroom_0051/00071 ~ nyuv2:1289（10），nyuraw:bedroom_0056b/00006 ~ nyuv2:550（10），nyuraw:bedroom_0060/00090 ~ nyuv2:221（10）
- nyu_depth_train/train × nyuv2/dev_train：同键 []；感知哈希 nyuraw:bathroom_0011/00040 ~ nyuv2:655（0），nyuraw:bedroom_0051/00102 ~ nyuv2:983（0），nyuraw:bedroom_0067a/00020 ~ nyuv2:1030（0），nyuraw:bedroom_0094/00001 ~ nyuv2:1105（0），nyuraw:dinette_0001/00001 ~ nyuv2:344（0），nyuraw:furniture_store_0002c/00029 ~ nyuv2:245（0），nyuraw:kitchen_0029b/00058 ~ nyuv2:797（0），nyuraw:kitchen_0053/00227 ~ nyuv2:905（0），nyuraw:living_room_0004/00054 ~ nyuv2:147（0），nyuraw:living_room_0004/00056 ~ nyuv2:147（0）
- nyu_depth_train/train × nyuv2/test：同键 []；感知哈希 nyuraw:bedroom_0126/00030 ~ nyuv2:1166（8），nyuraw:bathroom_0053/00039 ~ nyuv2:280（10），nyuraw:bathroom_0054/00019 ~ nyuv2:782（10），nyuraw:bedroom_0004/00093 ~ nyuv2:1226（10），nyuraw:bedroom_0004/00095 ~ nyuv2:1226（10），nyuraw:bedroom_0004/00130 ~ nyuv2:1226（10），nyuraw:bedroom_0051/00070 ~ nyuv2:1289（10），nyuraw:bedroom_0051/00071 ~ nyuv2:1289（10），nyuraw:bedroom_0056b/00006 ~ nyuv2:550（10），nyuraw:bedroom_0060/00090 ~ nyuv2:221（10）
- nyu_depth_train/train × odinw13/eval：同键 []；感知哈希 nyuraw:bedroom_0015/00058 ~ odinw:Raccoon_Raccoon.v2-raw.coco:raccoon-73_jpg.rf.52cf57605e6f485be44a3b9f8aa9fd8e.jpg（10），nyuraw:living_room_0038/00046 ~ odinw:ShellfishOpenImages_raw:34257f98c20621a3_jpg.rf.e84f781a5e8cd628ae1db53160efff8c.jpg（10），nyuraw:living_room_0040/00260 ~ odinw:EgoHands_generic:JENGA_COURTYARD_H_B_frame_2087_jpg.rf.6fc057c64d44c6546e5465f383b476e8.jpg（10），nyuraw:bathroom_0007/00037 ~ odinw:Aquarium_Aquarium_Combined.v2-raw-1024.coco:IMG_8317_jpg.rf.8692db77ddbc0e8a3c10a4202af99ced.jpg（12），nyuraw:bathroom_0007/00038 ~ odinw:Aquarium_Aquarium_Combined.v2-raw-1024.coco:IMG_8317_jpg.rf.8692db77ddbc0e8a3c10a4202af99ced.jpg（12），nyuraw:bathroom_0028/00145 ~ odinw:pistols_export:armas (2210).jpg.rf.825359b4b509936373efb873c531f4a3.jpg（12），nyuraw:bedroom_0004/00004 ~ odinw:EgoHands_generic:CARDS_COURTYARD_B_T_frame_1542_jpg.rf.405b3869c8b45a78d61be80104c95255.jpg（12），nyuraw:bedroom_0104/00109 ~ odinw:CottontailRabbits:7823435568_3121de3006_b_jpg.rf.fb476d05b70cf1d53a3dfad99983fb12.jpg（12），nyuraw:bedroom_0120/00081 ~ odinw:Raccoon_Raccoon.v2-raw.coco:raccoon-80_jpg.rf.1a8b801d529f6c41e42c6ed15f402ad6.jpg（12），nyuraw:dining_room_0001b/00101 ~ odinw:PascalVOC:2010_003667_jpg.rf.9d48e35fe50f100e98b80daed42870a7.jpg（12）
- nyu_depth_train/train × refcoco/eval：同键 []；感知哈希 nyuraw:bedroom_0106/00128 ~ coco:265766（6），nyuraw:dining_room_0031/00357 ~ coco:263516（8），nyuraw:kitchen_0035b/00294 ~ coco:221053（8），nyuraw:bedroom_0004/00175 ~ coco:35322（10），nyuraw:bedroom_0014/00053 ~ coco:61877（10），nyuraw:bedroom_0016/00160 ~ coco:225210（10），nyuraw:bedroom_0016/00212 ~ coco:453260（10），nyuraw:bedroom_0047/00057 ~ coco:214265（10），nyuraw:bedroom_0086/00013 ~ coco:443944（10），nyuraw:bedroom_0106/00100 ~ coco:473352（10）
- nyu_normals/train × ade20k/val：同键 []；感知哈希 nyuv2:69 ~ ade20k:ADE_val_00000652（10），nyuv2:115 ~ ade20k:ADE_val_00000962（12），nyuv2:409 ~ ade20k:ADE_val_00001667（12），nyuv2:624 ~ ade20k:ADE_val_00001487（12），nyuv2:818 ~ ade20k:ADE_val_00001251（12），nyuv2:848 ~ ade20k:ADE_val_00001513（12）
- nyu_normals/train × coco_val2017/e1sub500：同键 []；感知哈希 nyuv2:452 ~ coco:447465（10），nyuv2:54 ~ coco:86956（12），nyuv2:427 ~ coco:479448（12），nyuv2:675 ~ coco:135410（12）
- nyu_normals/train × countbench/eval：同键 []；感知哈希 nyuv2:990 ~ countbench:230（12）
- nyu_normals/train × cvbench/test：同键 []；感知哈希 nyuv2:652 ~ nuscenes:n008-2018-09-18-15-12-01-0400__CAM_FRONT__1537298355112404.jpg（10），nyuv2:818 ~ ade20k:ADE_val_00001251（10），nyuv2:108 ~ hypersim:hypersim/ai_054_007/images/scene_cam_00_final_preview/frame.0089.tonemap.jpg（12），nyuv2:161 ~ hypersim:hypersim/ai_054_007/images/scene_cam_00_final_preview/frame.0053.tonemap.jpg（12），nyuv2:425 ~ nuscenes:n015-2018-10-08-15-44-23+0800__CAM_FRONT__1538985040512460.jpg（12），nyuv2:499 ~ coco:228144（12），nyuv2:624 ~ ade20k:ADE_val_00001487（12），nyuv2:997 ~ ade20k:ADE_val_00000387（12），nyuv2:1111 ~ coco:216419（12）
- nyu_normals/train × embspatial/test：同键 []；感知哈希 nyuv2:140 ~ embspatial:scannet_112（12），nyuv2:251 ~ embspatial:scannet_1017（12），nyuv2:338 ~ embspatial:ai2thor_1079（12），nyuv2:346 ~ embspatial:mp3d_524（12），nyuv2:498 ~ embspatial:mp3d_557（12），nyuv2:1141 ~ embspatial:ai2thor_351（12）
- nyu_normals/train × nyu_normals/test：同键 []；感知哈希 nyuv2:320 ~ nyuv2:836（10），nyuv2:1037 ~ nyuv2:1398（10），nyuv2:263 ~ nyuv2:441（12），nyuv2:856 ~ nyuv2:663（12），nyuv2:904 ~ nyuv2:927（12），nyuv2:997 ~ nyuv2:550（12），nyuv2:1030 ~ nyuv2:209（12），nyuv2:1113 ~ nyuv2:1338（12）
- nyu_normals/train × nyuv2/dev_train：同键 ['nyuv2:10', 'nyuv2:1017', 'nyuv2:103', 'nyuv2:1030', 'nyuv2:1036', 'nyuv2:1105', 'nyuv2:1159', 'nyuv2:1160', 'nyuv2:1237', 'nyuv2:1281']；感知哈希 nyuv2:7 ~ nyuv2:7（0），nyuv2:10 ~ nyuv2:10（0），nyuv2:51 ~ nyuv2:51（0），nyuv2:103 ~ nyuv2:103（0），nyuv2:147 ~ nyuv2:147（0），nyuv2:158 ~ nyuv2:158（0），nyuv2:216 ~ nyuv2:216（0），nyuv2:217 ~ nyuv2:217（0），nyuv2:218 ~ nyuv2:218（0），nyuv2:231 ~ nyuv2:231（0）
- nyu_normals/train × nyuv2/test：同键 []；感知哈希 nyuv2:320 ~ nyuv2:836（10），nyuv2:263 ~ nyuv2:441（12），nyuv2:438 ~ nyuv2:635（12），nyuv2:856 ~ nyuv2:663（12），nyuv2:904 ~ nyuv2:927（12），nyuv2:997 ~ nyuv2:550（12），nyuv2:1037 ~ nyuv2:1398（12），nyuv2:1113 ~ nyuv2:1338（12）
- nyu_normals/train × refcoco/eval：同键 []；感知哈希 nyuv2:52 ~ coco:111992（12），nyuv2:99 ~ coco:238618（12），nyuv2:146 ~ coco:503777（12），nyuv2:659 ~ coco:221053（12），nyuv2:660 ~ coco:155618（12），nyuv2:662 ~ coco:474461（12），nyuv2:665 ~ coco:496732（12），nyuv2:794 ~ coco:567008（12），nyuv2:924 ~ coco:32818（12），nyuv2:981 ~ coco:212757（12）
- refcoco_train/train × ade20k/val：同键 []；感知哈希 coco:25549 ~ ade20k:ADE_val_00001412（10），coco:88210 ~ ade20k:ADE_val_00001570（10），coco:98543 ~ ade20k:ADE_val_00000595（10），coco:150932 ~ ade20k:ADE_val_00000995（10），coco:279385 ~ ade20k:ADE_val_00000018（10），coco:282503 ~ ade20k:ADE_val_00001238（10），coco:304023 ~ ade20k:ADE_val_00001157（10），coco:351855 ~ ade20k:ADE_val_00000831（10），coco:422528 ~ ade20k:ADE_val_00000555（10），coco:492131 ~ ade20k:ADE_val_00001411（10）
- refcoco_train/train × blink_reldepth/val：同键 []；感知哈希 coco:76578 ~ blink:val_Relative_Depth_92（10），coco:112785 ~ blink:val_Relative_Depth_19（12），coco:219357 ~ blink:val_Relative_Depth_64（12），coco:393027 ~ blink:val_Relative_Depth_3（12），coco:519652 ~ blink:val_Relative_Depth_3（12）
- refcoco_train/train × coco_val2017/e1sub500：同键 []；感知哈希 coco:334631 ~ coco:423123（2），coco:414647 ~ coco:208901（10），coco:556599 ~ coco:250127（10），coco:2886 ~ coco:172877（12），coco:59080 ~ coco:228436（12），coco:66708 ~ coco:555005（12），coco:73814 ~ coco:208901（12），coco:89629 ~ coco:179642（12），coco:103338 ~ coco:481386（12），coco:106293 ~ coco:577864（12）
- refcoco_train/train × countbench/eval：同键 []；感知哈希 coco:10890 ~ countbench:123（10），coco:233617 ~ countbench:283（10），coco:4278 ~ countbench:289（12），coco:5811 ~ countbench:514（12），coco:21097 ~ countbench:164（12），coco:44679 ~ countbench:532（12），coco:64093 ~ countbench:491（12），coco:65095 ~ countbench:77（12），coco:79156 ~ countbench:248（12），coco:169217 ~ countbench:422（12）
- refcoco_train/train × cvbench/test：同键 []；感知哈希 coco:334631 ~ coco:423123（2），coco:81961 ~ coco:392481（8），coco:11856 ~ nuscenes:n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151830112404.jpg（10），coco:56005 ~ coco:367082（10），coco:98543 ~ ade20k:ADE_val_00000595（10），coco:118745 ~ ade20k:ADE_val_00001965（10），coco:201260 ~ nuscenes:n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151271912404.jpg（10），coco:248769 ~ sunrgbd:SUNRGBD/kv2/kinect2data/002955_2014-06-08_17-34-45_094959634447_rgbf000150-resize/image/0000150.jpg（10），coco:324252 ~ coco:377723（10），coco:410966 ~ hypersim:hypersim/ai_048_010/images/scene_cam_03_final_preview/frame.0098.tonemap.jpg（10）
- refcoco_train/train × embspatial/test：同键 []；感知哈希 coco:126651 ~ embspatial:scannet_1064（10），coco:199619 ~ embspatial:scannet_413（10），coco:225634 ~ embspatial:scannet_68（10），coco:323427 ~ embspatial:mp3d_334（10），coco:329966 ~ embspatial:scannet_361（10），coco:363549 ~ embspatial:ai2thor_634（10），coco:425216 ~ embspatial:ai2thor_1195（10），coco:511062 ~ embspatial:scannet_869（10），coco:563447 ~ embspatial:mp3d_118（10），coco:578466 ~ embspatial:ai2thor_1004（10）
- refcoco_train/train × nyu_normals/test：同键 []；感知哈希 coco:169095 ~ nyuv2:1412（10），coco:365911 ~ nyuv2:928（10），coco:8602 ~ nyuv2:522（12），coco:43371 ~ nyuv2:192（12），coco:49424 ~ nyuv2:1196（12），coco:90317 ~ nyuv2:537（12），coco:95360 ~ nyuv2:1397（12），coco:105589 ~ nyuv2:843（12），coco:112272 ~ nyuv2:29（12），coco:136043 ~ nyuv2:1038（12）
- refcoco_train/train × nyuv2/dev_train：同键 []；感知哈希 coco:363341 ~ nyuv2:1160（12）
- refcoco_train/train × nyuv2/test：同键 []；感知哈希 coco:169095 ~ nyuv2:1412（10），coco:365911 ~ nyuv2:928（10），coco:8602 ~ nyuv2:522（12），coco:43371 ~ nyuv2:192（12），coco:49424 ~ nyuv2:1196（12），coco:90317 ~ nyuv2:537（12），coco:95360 ~ nyuv2:1397（12），coco:105589 ~ nyuv2:843（12），coco:112272 ~ nyuv2:29（12），coco:136043 ~ nyuv2:1038（12）
- refcoco_train/train × odinw13/eval：同键 []；感知哈希 coco:290110 ~ odinw:ShellfishOpenImages_raw:0a41b29ef96a4188_jpg.rf.18b38c5146081bc86ccd0149b6d5ca1c.jpg（10），coco:578094 ~ odinw:EgoHands_generic:PUZZLE_COURTYARD_B_S_frame_2549_jpg.rf.43b5ef33429b5da63c3a9c5cb6415da3.jpg（10），coco:72258 ~ odinw:ShellfishOpenImages_raw:0ef2b7b92267c0d1_jpg.rf.8bd3e289b443fc9297dd0908381996a0.jpg（12），coco:73075 ~ odinw:CottontailRabbits:rabbit-5736119_1280_jpg.rf.127a89c150e1ff45be50b7aa492e7073.jpg（12），coco:97778 ~ odinw:pistols_export:armas (540).jpg.rf.6d0f6d67199bc564a4f4bd04c59a5768.jpg（12），coco:101050 ~ odinw:AerialMaritimeDrone_large:DJI_0263_JPG.rf.46456e1604da41debe69ed67f7486ed2.jpg（12），coco:115342 ~ odinw:pistols_export:armas (575).jpg.rf.203bd693a80da99aa87377ee499c1ba6.jpg（12），coco:119750 ~ odinw:pistols_export:armas (1584).jpg.rf.a4608d0cf6c9ac402046862e136e5958.jpg（12），coco:125822 ~ odinw:Aquarium_Aquarium_Combined.v2-raw-1024.coco:IMG_2636_jpeg_jpg.rf.75197e967fba5462d754b0e72e88fa80.jpg（12），coco:138961 ~ odinw:pistols_export:armas (2210).jpg.rf.825359b4b509936373efb873c531f4a3.jpg（12）
- refcoco_train/train × probe/D20：同键 []；感知哈希 coco:366967 ~ coco:39769（12）
- refcoco_train/train × refcoco/eval：同键 []；感知哈希 coco:252787 ~ coco:395169（0），coco:176298 ~ coco:173484（2），coco:202270 ~ coco:412002（6），coco:416652 ~ coco:108920（6），coco:333066 ~ coco:209185（8），coco:403718 ~ coco:246342（8），coco:508741 ~ coco:388991（8），coco:572745 ~ coco:95455（8），coco:9095 ~ coco:521366（10），coco:85143 ~ coco:73671（10）

## 阈值标定

- 已知重合 1492 对，距离 ≤ 8 的占 0.954424（68 对超出阈值）；随机不同键图对 199996 对，距离 ≤ 8 的比例 0.0
- 已知重合 ade20k/val ~ cvbench/test：n=633，最大 6，中位 0.0，p95 2.0
- 已知重合 coco_val2017/e1sub500 ~ cvbench/test：n=83，最大 4，中位 0.0，p95 2.0
- 已知重合 cvbench/test ~ nyu_normals/test：n=36，最大 20，中位 14.0，p95 18.0
- 已知重合 cvbench/test ~ nyuv2/test：n=36，最大 18，中位 14.0，p95 18.0
- 已知重合 nyu_normals/test ~ nyuv2/test：n=654，最大 8，中位 0.0，p95 2.0
- 已知重合 nyu_normals/train ~ nyuv2/dev_train：n=50，最大 4，中位 0.0，p95 3.0999999999999943
- 随机不同键图对：n=199996，最小 10，1% 分位 22.0，各档比例 {'0': 0.0, '2': 0.0, '4': 0.0, '6': 0.0, '8': 0.0, '10': 5.00010000200004e-06, '12': 1.500030000600012e-05}

## 训练剔除清单

- ade20k/train：26 条
- nyu_depth_train/train：132 条
- nyu_normals/train：50 条
- refcoco_train/train：10 条
- 参与比较的评测集（剔除清单的 eval_sets，13 个）：['ade20k/val', 'blink_reldepth/val', 'coco_val2017/e1sub500', 'coco_val2017_full/val2017', 'countbench/eval', 'cvbench/test', 'embspatial/test', 'nyu_normals/test', 'nyuv2/dev_train', 'nyuv2/test', 'odinw13/eval', 'probe/D20', 'refcoco/eval']

训练加载器（src/data/train_loaders.py）默认读 `<训练数据根>/dedupe/train_exclusions.json`，自动排除这些样本；训练集在清单里没有键就报错（没比较过）。
