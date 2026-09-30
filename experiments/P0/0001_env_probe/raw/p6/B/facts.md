# RUN-0001 §6 数据源事实 · B 组（NYU Depth V2 / BLINK Relative Depth / Hypersim）

- 采集时间：2026-09-30（UTC），采集者：服务器 CC 子代理 p6_B。只陈述事实，不做推荐。
- 下文所说的"证据"均为 `$PROJ/logs/0001/p6/B/` 下的相对路径（`$PROJ=/home/ubuntu/workspace/wzg/gjl/vision-internalization`）。脚本放在 `$PROJ/scripts/p6_B_*.py`。
- 字节数写原值，括号里给出 GB（1e9）或 GiB（2^30），并注明单位。"HEAD"指 HTTP HEAD 的 Content-Length；"HF API"指 `HfApi().dataset_info(files_metadata=True)` 汇总的 siblings 大小。
- 没有做任何大体量下载。range read 只读进内存，不落盘；所有网络读取都记在 `downloads.tsv`。

---

## 1. NYU Depth V2

### 1.1 官方来源与许可

| 项 | 事实 | 证据 |
|---|---|---|
| 官方主页 | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html（`~silberman/...` 301 跳转到这里） | `nyu_official_page.html`、`.txt` |
| 引用 | Silberman, Hoiem, Kohli, Fergus, "Indoor Segmentation and Support Inference from RGBD Images", ECCV 2012 | 同上 |
| 主页自述 | "1449 densely labeled pairs of aligned RGB and depth images / 464 new scenes taken from 3 cities / 407,024 new unlabeled frames" | 同上 |
| **许可** | **官方主页没有任何 license、terms 或 copyright 文字**（grep `licen|terms|copyright|non-commercial` 命中 0 次）。页面只要求 "If you use the dataset, please cite the following work"。 | `nyu_official_page.html` |
| Labeled 集 | http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat → 302 到 https，HEAD 为 **2,972,037,809 B**（2.97 GB / 2.77 GiB），Last-Modified 2020-07-14，Accept-Ranges bytes。主页标注 "~2.8 GB"。 | `nyu_official_head.jsonl` |
| Raw 集（单文件） | `.../nyu_depth_v2/nyu_depth_v2_raw.zip`，HEAD 为 **428,798,889,146 B**（428.8 GB / 399.4 GiB）。主页标注 "~428 GB"。 | `nyu_official_head.jsonl` |
| Raw 集（分卷） | 主页列出 37 个分卷 zip，HEAD 全部 200，合计 **407,558,846,978 B**（407.6 GB / 379.6 GiB）。最小的是 cafe.zip，1.67 GB；最大的是 living_rooms_part4.zip，27.05 GB。 | `nyu_official_head.jsonl` |
| Toolbox | https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip，30,990 B | `nyu_toolbox_excerpts.txt` |
| 官方 splits.mat | http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat，2,626 B。它的 sha256 `6e0814…0cee` 与本地 `/home/ubuntu/workspace/wzg/gjl/datasets/nyuv2/splits.mat` **完全一致**。注意 `.../nyu_depth_v2/splits.mat` 返回 404。 | `nyu_splits_mat.json`、`nyu_official_head.jsonl` |

### 1.2 Labeled `.mat` 的变量（MATLAB v7.3 / HDF5；range read 元数据共 3.34 MB，只进内存）

文件头原文："MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Fri Jul 13 11:34:28 2012 HDF5 schema 1.00 ."。h5py 看到的形状是 MATLAB 维度的转置。

| 变量 | h5py shape | dtype / MATLAB_class | 压缩 |
|---|---|---|---|
| images | (1449, 3, 640, 480) | uint8 | gzip，chunk (1,3,42,480) |
| depths（Levin 着色法补洞，单位米） | (1449, 640, 480) | float32 / single | gzip |
| rawDepths（投影后未补洞，单位米） | (1449, 640, 480) | float32 / single | gzip |
| labels（0 = unlabeled，1..C） | (1449, 640, 480) | uint16 | gzip |
| instances | (1449, 640, 480) | uint8 | gzip |
| accelData | (4, 1449) | float32 | gzip |
| names | (1, 894) | cell，**C = 894 类** | — |
| namesToIds | (1, 6) | containers.Map | — |
| scenes / sceneTypes / rawRgbFilenames / rawDepthFilenames | (1, 1449) | cell | gzip |
| `#refs#`（8484 个成员）、`#subsystem#` | group | — | — |

- 变量的语义以官方主页原文为准，见 `nyu_official_page.txt` 的 "Labeled Dataset" 一节。证据：`nyu_labeled_mat_h5meta.json`。
- 解码 scenes、sceneTypes、names 另外 range read 了 1.64 MB，只进内存。结果：**1449 张图来自 464 个不同的 scene；sceneTypes 有 27 种取值**，其中包含拼写 "excercise_room"。names 的前 10 个依次是 book、bottle、cabinet、ceiling、chair、cone、counter、dishwasher、faucet、fire extinguisher。证据：`nyu_labeled_scenes_names.json`。
- Toolbox 的 `get_projection_mask.m` 原文是 `mask = false(480, 640); mask(45:471, 41:601) = 1; sz = [427 561];`（MATLAB 1-based、闭区间）。`camera_params.m` 给出 fx_rgb=518.8579、fy_rgb=519.4696、cx_rgb=325.5824、cy_rgb=253.7362、maxDepth=10。证据：`nyu_toolbox_excerpts.txt`。

### 1.3 官方 795/654 划分（本地 `splits.mat`，只读，用 scipy.io.loadmat）

| 项 | 值 |
|---|---|
| 文件头 | "MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Wed Jun 6 14:22:01 2012" |
| 变量 | `trainNdxs`：uint16，shape (795, 1)；`testNdxs`：uint16，shape (654, 1) |
| 取值范围 | train 为 min 3 / max 1440；test 为 min 1 / max 1449。两者都严格升序、无重复。 |
| 索引基 | **1-based**：并集等于 {1..1449}，不等于 {0..1448}。 |
| train∩test | **空集（0）** |
| train∪test | **1449**，等于 {1..1449} |
| test 前 10 个 | 1, 2, 9, 14, 15, 16, 17, 18, 21, 28 |
| 场景层面（结合 `.mat` 的 scenes 变量） | train 覆盖 249 个 scene，test 覆盖 215 个 scene，**两者没有共同的 scene**（249 + 215 = 464） |

- 证据：`nyu_splits_mat.json`、`nyu_labeled_scenes_names.json`，脚本 `p6_B_splits.py`、`p6_B_h5cells.py`。

### 1.4 常用的大训练集（Eigen / BTS "sync" / DenseDepth / FastDepth 及 HF 镜像）

| 来源 | 帧数 / 划分 | 大小（来源类型） | 格式 | 许可 / 条款（原文） | 证据 |
|---|---|---|---|---|---|
| **BTS sync.zip**（官方链接：Google Drive `1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP`，出自 cleinc/bts README） | 训练清单 `train_test_inputs/nyudepthv2_train_files_with_gt.txt` 有 **24,231 对**，分布在 284 个 raw 目录；测试清单 `nyudepthv2_test_files_with_gt.txt` 有 **654 行** | Drive 页面显示 "sync.zip **(5.9G)**"（Drive 自己的显示单位）。HEAD 返回的是 HTML，拿不到字节数。 | 目录形如 `/<scene>/rgb_XXXXX.jpg` 加 `sync_depth_XXXXX.png`，每行还带 focal 518.8579 | 仓库代码 GPL-3.0，README 原文："This Software is licensed under GPL-3.0-or-later."；**数据本身没有单独的许可说明** | `gh/cleinc_bts_*`、`gdrive_bts_sync_meta.txt`、`gh/github_repo_licenses.jsonl` |
| HF `aradhye/nyu_depth_v2`（非官方；目录布局与 BTS 一致） | `sync/` 下有 **36,396 对** RGB-D，284 个目录，BTS 的 24,231 条训练清单**全部都在里面**；`official_splits/` 下有 train 795 对、test 654 对、val 654 对（BTS 654 条测试清单在 test 和 val 里都有） | HF API：6,857,933,947 B；zip 本体 6,857,931,462 B（6.39 GiB）。其中 sync 解压后 6,366,984,533 B（5.93 GiB）；BTS 24,231 对 4,240,085,416 B；official test 159,056,062 B。 | zip，内含 jpg/png | 卡片只写 `license: mit`，没有其他正文 | `zips/aradhye__nyu_depth_v2.summary.json`、`zips/aradhye_subset_bytes.json` |
| **DenseDepth nyu_data.zip**（ialhashim/DenseDepth） | README 原文："NYU Depth V2 (50K) (4.1 GB)" | **未能核实**：S3 地址 `s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip` 与 `nyu_test.zip`（README 标 1.4 GB）HEAD 都返回 **403**；README 链接 tinyurl.com/nyu-data-zip 跳转到一个 Drive 文件夹，该文件夹返回 **404** | README 称训练时直接把 zip 整个读进内存 | 仓库代码 GPL-3.0；数据许可未说明 | `head_misc1.jsonl`、`gh/ialhashim_DenseDepth_master_README.md` |
| **FastDepth / sparse-to-dense h5**（dwofk/fast-depth，原始地址 `datasets.lids.mit.edu/{fastdepth,sparse-to-dense}/data/nyudepthv2.tar.gz`） | TFDS 目录页：train **47,584**，validation **654** | 原始地址**连接超时**，两次，每次 60 s，**未能核实**。TFDS 页面写 "Download size: 31.92 GiB, Dataset size: 74.03 GiB"。 | TFDS 特征：depth (480,640) float16，image (480,640,3) uint8 | fast-depth 仓库 MIT；TFDS 页面内容 CC BY 4.0（指文档本身） | `head_misc1.jsonl`、`tfds_nyu_excerpt.txt` |
| HF `sayakpaul/nyu_depth_v2`（上一行 FastDepth 数据的镜像） | 卡片写 train 47,584 / validation 654 | HF API：**35,151,139,339 B**（12 个 train tar、2 个 val tar，外加一个加载脚本）；val 两个 tar 共 1,016,340,480 B | tar 里是 `.h5`，例如 `val/official/00001.h5`、`train/basement_0001a/00001.h5`。实测 h5 键：`rgb` (3,480,640) uint8，`depth` (480,640) float32。两个 val 样本的 depth 零值像素数都是 0（min 1.80/1.12 m，max 3.62/2.61 m）。训练样本帧号步长为 5（00001、00006、00011）。 | YAML 写 `apache-2.0`，脚本 `_LICENSE = "Apace 2.0 License"`，卡片正文写 "licensed under a MIT License"（指向 fast-depth LICENSE）——**三处不一致** | `hf/hfinfo_sayakpaul__nyu_depth_v2.json`、`hf/sayakpaul_*_tar_peek.json`、`hf/cards/sayakpaul__nyu_depth_v2/*` |
| HF `SobanHM/nyu_depth_v2`、`heuliuh/nyu_depth_v2` | 与 sayakpaul 相同 | 总字节同为 35,151,139,339 B，卡片与脚本也相同（sha 不同） | 同上 | 同上 | `hf/hfinfo_*.json` |
| HF `vikhyatk/nyu_depth_v2` | datasets-server：train 47,584 / validation 654 | parquet 32,624,898,041 B | 特征 `image`: Image、`depth_map`: Image。datasets-server 取行报错 "Image cannot be written as JPEG or PNG"，所以 first-rows 拿不到。 | 卡片无 license | `dsserver/vikhyatk_*` |
| HF `adams-story/nyu-depthv2-wds` | 卡片写 47,584 / 654，并说明来自 sayakpaul，四边各裁 16 px，depth 存 float16 | HF API：36,141,612,641 B（49 个 tar） | webdataset（jpg + npy） | 卡片无 license | `hf/cards/adams-story__nyu-depthv2-wds/README.md` |
| Eigen et al. 2014 原始训练集 | — | **未能核实**：作者页 cs.nyu.edu/~deigen/depth/ 现在返回 403。Wayback 快照（2022-09-28）只列出 predictions、inference code 和 weights，**没有训练数据下载链接**。 | — | — | `wayback/eigen_depth.txt` |

Labeled 集的其他 HF 镜像（与 1449 张相关，不是 24K 帧集）：

| repo | 内容 | 大小（HF API） | 许可 |
|---|---|---|---|
| `0jl/NYUv2` | 只有加载脚本，从官方下载 `.mat`；`datasets>=4.0` 已不支持 | 9,892 B | 无 |
| `kronecker0122/nyuv2` | `nyu_depth_v2_labeled.mat` 的 tar.zst 包 | 2,963,990,529 B | `other` |
| `xcll/NYU-V2` | zip 里有完整的 `nyu_depth_v2_labeled.mat`（2,972,037,809 B，与官方同大小），外加 depth/、depth1/、labels/ 三个目录，各 1449 个 png | 4,764,119,859 B | 无卡片 |
| `jagennath-hari/nyuv2` | **train 1014 / val 290 / test 145**（不是官方 795/654）；字段 id、rgb、depth（mm, uint16 tiff）、semantic、instance | 1,285,148,700 B | mit |
| `WyettZ/nyuv2-raw-depth` | **train 797 / test 652**（不是 795/654）；448×448，字段 depth float32 + mask bool | 1,555,500,671 B | 无 |
| `tanganke/nyuv2` | train 795 / val 654；288×384；字段 image、segmentation、depth、**normal**、noise | 2,971,254,901 B | 无（卡片写来源为 ForkMerge Tsinghua Cloud） |
| `idsia-robotics/micro-nyuv2` | 48×48，自定义划分 22,600 / 2,726 / 2,948 | 988,750,919 B | 卡片称 MIT |
| `Zhonghua/NYUDv2` | **不是数据**：zip 里是 `ACNet_NYUDv2_best.pth`（模型权重） | 867,157,681 B | mit |

HF 检索的原始结果见 `hf_search_nyu.json`、`hf_search_misc.json`。

### 1.5 Eigen 654 测试集与 crop 的定义出处（只给出处，不引用论文数值）

- 测试集：Eigen, Puhrsch, Fergus, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NIPS 2014（arXiv:1406.2283；HEAD 6,579,016 B，**PDF 未下载，论文正文未核实**）。作者页的 Wayback 快照见 `wayback/eigen_depth.txt`。
- 实测：BTS 的 654 条测试清单，文件名编号 0..1448（0-based），加 1 后**与 `splits.mat` 的 testNdxs 完全相同**。BTS 生成测试 GT 的脚本 `utils/extract_official_train_test_set_from_mat.py` 读的是 `h5_file['rawDepths']`，不是补洞后的 `depths`；它按 `sceneTypes` 分目录，并把 RGB 的 `[7:474, 7:632]` 以外置黑。证据：`gh/cleinc_bts_master_utils_extract_official_train_test_set_from_mat.py`。
- crop 在代码里的定义：BTS `pytorch/bts_eval.py` 的参数 `--eigen_crop`（help 原文 "crops according to Eigen NIPS14"），NYU 分支写 `eval_mask[45:471, 41:601] = 1`，Python 0-based 半开区间。官方 toolbox 写 `mask(45:471, 41:601)`，MATLAB 1-based 闭区间，`sz=[427 561]`。证据：`gh/cleinc_bts_master_pytorch_bts_eval.py`、`nyu_toolbox_excerpts.txt`。
- Eigen & Fergus ICCV 2015 的页面（Wayback 2022-10-10）提供 NYUv2 测试集的 predictions_depth/normals `.mat` 文件。见 `wayback/eigen_dnl.txt`。

### 1.6 NYUv2 表面法向 GT 来源

| 来源 | 划分 / 张数 | 下载地址 | 大小 | 格式（实测或原文） | 许可 | 证据 |
|---|---|---|---|---|---|---|
| Ladický, Zeisl, Pollefeys, ECCV 2014 | 未能核实 | 作者页（Wayback 2017-08-11 快照）有链接 `./nyu_normals_gt.zip`，标注 "[NYU_gt]" | **未能核实**：现地址 www.inf.ethz.ch/… 返回 404，people.inf.ethz.ch/… 显示 "Website not found"，Wayback 没有该 zip 的存档 | 未能核实 | 未说明 | `wayback/ladicky_2017.txt`、`head_normals_wayback.jsonl` |
| Eigen & Fergus, ICCV 2015（dnl 页） | 原文："Ground truth normals: Precomputed ground truth normals. These were calculated using Nathan Silberman's code"。张数未能核实。 | https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz（现返回 **403**）；Wayback：`web.archive.org/web/20220412033406id_/…/normals_gt.tgz` | Wayback HEAD **496,214,722 B** | tar.gz。前 2 MB 可见 `normals_gt/normals/0000.png … 0006.png`，第一张是 640×480、RGB、8-bit，取值 0..254。成员总数未能核实，因为 gzip 不能随机访问。 | 页面未说明 | `wayback/eigen_dnl.txt`、`wayback/eigen_normals_gt_*.json` |
| Wang, Fouhey, Gupta, CVPR 2015 | — | 项目页 http://www.cs.cmu.edu/~xiaolonw/deep3d.html 只提供 "results for NYU Depth V2"（normals.tar.gz，是**预测结果**），地址 ladoga.graphics.cs.cmu.edu **DNS 无法解析**；代码 README 给了一个 Dropbox 链接，同样是 results | 未能核实 | — | caffe-3dnormal 仓库：NOASSERTION | `wayback/wang_deep3d.txt`、`gh/xiaolonw_caffe-3dnormal_README.md` |
| Qi et al. GeoNet, CVPR 2018 | 训练集：Bae 2021 README 原文 "a larger training set consisting of **30907** images"；测试集为官方 654 | 训练数据在 HKU SharePoint（`hkuhk-my.sharepoint.com/…/xjqi_hku_hk/Ek0Vm…`，HEAD 返回 200 HTML，**大小未能核实**）；预训练模型在 Drive `1o2t8735…` | 未能核实 | — | 仓库**没有 LICENSE**（GitHub API 返回 license 为 null） | `gh/xjqi_GeoNet_master_README.md`、`head_normals_links.jsonl` |
| GeoNet 测试集的 HF 镜像 `ema-ioana-xyz/nyu-v2-geonet-test`（非官方） | **654 个 `.mat`**，文件名编号 1..1449，**与 testNdxs（1-based）完全相同**；另有 `test_split.txt`，654 行 | HF API：4,621,182,485 B，每个文件 7,065,976 B | MAT v5 未压缩，变量：`img` uint8 [480,640,3]、`depth` single [640,480]、`norm` single [3,640,480]、`mask` single [640,480]（range read 1,184 B） | 无卡片，无 license | `hf/ema_geonet_test_1mat_vars.json`、`hf/hfinfo_ema-ioana-xyz__nyu-v2-geonet-test.json` |
| Bae, Budvytis, Cipolla, ICCV 2021（surface_normal_uncertainty） | 原文 "NYUv2 (official): The official train/test split contains 795/654 images"；"We used the ground truth and data split provided by GeoNet" | Drive 文件夹 `1Ku25Am69h_HrbtcCptXn4aetjo7sB33F` 里的 `nyu_dataset.zip`（文件 id `1Rl-x6Gw…`） | Drive 显示 **"962M"** | zip，解压后为 `./datasets/nyu/train`、`./datasets/nyu/test/` | 仓库 MIT | `gh/baegwangbin_surface_normal_uncertainty_README.md`、`gdrive_normals_meta.txt` |
| Bae & Davison, CVPR 2024（DSINE） | `dsine_eval.zip` 含 6 个评测集，其中有 nyuv2；README 原文 "There are only 795 images here"（指 nyuv2/train） | Drive 文件夹 `1t3LMJII…` 里的 `dsine_eval.zip`（id `1u6_-Zvg…`） | Drive 显示 **"21G"**，是整包大小 | — | 仓库 NOASSERTION；README 原文 "By downloading the dataset, you are agreeing to the respective LICENSE of each dataset" | `gh/baegwangbin_DSINE_main_README.md`、`gdrive_normals_meta.txt` |
| HF `obukhovai/marigold_normals_eval` | `nyuv2/test` 有 **654 张**，编号 0..1448，**等于 testNdxs−1** | nyuv2 子目录合计 5,100,451,816 B（整个 repo 14,041,306,426 B，另含 ibims、scannet、sintel） | 每张一个 `NNNNNN_img.png` 加一个 `NNNNNN_normal.npy`，后者为 float64 `<f8` (480,640,3) | 无卡片 license；readme 原文 "check the relevant LICENSE before using this dataset. GT surface normals are from GeoNet (CVPR 2018)" | `hf/marigold_normals_eval_nyuv2_readme.txt` |
| HF `tanganke/nyuv2` | 795 / 654，288×384，`normal` float32 [3,288,384] | 2,971,254,901 B | parquet | 无 | `dsserver/tanganke_nyuv2_info.json` |
| FrameNet（Huang et al., ICCV 2019） | README 只提到 ScanNet 法向，**没有 NYUv2 法向下载** | — | — | MIT | `gh/hjwdzh_FrameNet_master_README.md` |
| Fouhey et al., ICCV 2013（3DP） | 页面只有预测代码、模型和一个 "Data (926MB .tar.gz)"，**没有标明 NYU 法向 GT**；原文 "precomputed results … Please contact David Fouhey" | — | — | 页面 Copyright Notice 原文："…included by the contributing authors as a means to ensure timely dissemination of scholarly and technical work on a non-commercial basis…" | `gh/web_www.cs.cmu.edu__dfouhey_3DP_.html` |

---

## 2. BLINK：Relative Depth 子任务

| 项 | 事实 | 证据 |
|---|---|---|
| 官方来源 | HF `BLINK-Benchmark/BLINK`（repo sha 见 hfinfo），GitHub `zeyofu/BLINK_Benchmark`，论文 arXiv:2404.12390（ECCV 2024） | `hf/hfinfo_BLINK-Benchmark__BLINK.json` |
| 许可 | HF 卡片 `license: apache-2.0`；GitHub 仓库 LICENSE 为 Apache-2.0。卡片 Disclaimers 原文："Blink makes use of data from existing image datasets … if you are the copyright owner of any image … please contact us"。 | `hf/cards/BLINK-Benchmark__BLINK/README.md`、`gh/zeyofu_BLINK_Benchmark_main_LICENSE` |
| 图像来源 | 论文原文："We curate our samples using human annotations from the Depth in the Wild [19] dataset. Each question contains an image and two specified points."（DIW 本身的许可**未核实**） | `blink/blink_paper_excerpt.txt` |
| configs | 14 个：Art_Style、Counting、Forensic_Detection、Functional_Correspondence、IQ_Test、Jigsaw、Multi-view_Reasoning、Object_Localization、**Relative_Depth**、Relative_Reflectance、Semantic_Correspondence、Spatial_Relation、Visual_Correspondence、Visual_Similarity。每个 config 都有 val 和 test。 | `blink/splits.json` |
| Relative_Depth 行数 | **val 124，test 124**（datasets-server info，partial=False） | `blink/info_Relative_Depth.json` |
| 大小 | parquet：val 4,604,789 B，test 4,599,186 B；info 中 download_size 为 9,203,975 B。整个 BLINK repo 805,181,939 B。 | 同上、hfinfo |
| schema（datasets-server features 原样） | `idx`: string；`question`: string；`sub_task`: string；`image_1..image_4`: Image；`choices`: list[string]；`answer`: string；`prompt`: string；`explanation`: string | `blink/info_Relative_Depth.json`、`blink/firstrows_*.json` |
| 实测字段内容（全部 248 行） | 只有 image_1 有值，image_2..4 在 124/124 行都是 null；choices 都是 `["A is closer","B is closer"]`；question 为 "Which point is closer to the camera?"；prompt 里写 "Two points are circled on the image, labeled by A and B beside each circle…"；explanation 都为空；图像尺寸不一，例如 500×375、500×334 | `blink/rows_*.json` |
| **val 答案** | **公开**：`answer` 取值 "(A)" 64 行，"(B)" 60 行 | `blink/rows_val_*.json` |
| **test 答案** | **不公开**：`answer` 在 124/124 行都是字面值 **"hidden"** | `blink/rows_test_*.json` |
| 官方关于 test 的原文 | HF 卡片与 GitHub README 都写："We have released a full suite comprising 1,901 validation samples, the prompts we used, and model predictions … However, the 1,907 test questions are available without their answers. You can submit your model's predictions for the test set on EvalAI." | 两个 README |
| **EvalAI 评测服务器状态** | challenge 2287 "BLINK-Benchmark Evaluation Challenge"：start 2024-04-19，**end_date 2026-01-31T23:59:59Z，is_active=False**。唯一 phase 4540 "Test Set Evaluation"，end 2026-01-31，is_active=False，每日上限 20 次，每月上限 1000 次。 | `blink/evalai_challenge_2287.json`、`blink/evalai_2287_phases.json` |

---

## 3. Hypersim（可选）

| 项 | 事实 | 证据 |
|---|---|---|
| 官方来源 | GitHub `apple/ml-hypersim` **已 301 跳转**到 `apple-aiml-research/ml-hypersim`（默认分支 main）；论文 arXiv:2011.02523 | `gh/github_repo_licenses.jsonl` |
| **数据许可（原文）** | "The Hypersim Dataset is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License (http://creativecommons.org/licenses/by-sa/3.0/)." | `hypersim/README.md` 第 7 行 |
| 代码许可 | `LICENSE.txt` 是 Apple 示例代码许可（原文开头："Copyright (C) 2020 Apple Inc. All Rights Reserved. IMPORTANT: This Apple software is supplied to you by Apple Inc. …"），并注明子组件见 ACKNOWLEDGEMENTS.txt。GitHub API 识别为 NOASSERTION。 | `hypersim/LICENSE.txt` |
| Evermotion 资产相关原文 | "To obtain the ground truth triangle meshes for each scene, you must purchase the asset files here (TurboSquid, Evermotion)."；"we use Evermotion Archinteriors Volumes 1-55 excluding 20,25,40,49. All the Evermotion Archinteriors volumes are available for purchase"；"The exported OBJ file for each scene (which can be obtained by purchasing the original scene assets)"。README 没有写出针对渲染图像的额外 Evermotion 使用限制条款。 | `hypersim/README.md` 第 37、151、344 行 |
| 规模（原文） | "we generate 77,400 images of 461 indoor scenes"；"our public release contains 74,619 images, rather than 77,400"（去掉了含人物或显著 logo 的图）；"our dataset is roughly 1.9TB … a few hundred separate ZIP files, where each ZIP file is between 1GB and 20GB" | README 第 5、33、35 行 |
| 下载粒度 | **每个 scene 一个 zip**，地址形如 `https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_VVV_NNN.zip`。官方下载脚本 `code/python/tools/dataset_download_images.py` 中有 **457 个**不同的 zip URL。`contrib/99991/download.py` 可以只取 zip 内的指定文件（`--list`、`--contains`）。 | `hypersim/scene_zip_urls.txt` |
| **实测总大小** | 457 个 zip **全部 HEAD 200**，合计 **1,914,288,906,139 B**（1.914 TB / 1.741 TiB）。均值 4,188,815,987 B，中位数 3,036,535,483 B，最小 77,947,604 B，最大 20,591,130,595 B。Last-Modified 为 2020-12-29。 | `hypersim/head_scene_zips.jsonl`、`hypersim/scene_zip_sizes_summary.json` |
| 官方 v1 划分（`evermotion_dataset/analysis/metadata_images_split_scene_v1.csv`，2,962,706 B） | 列：scene_name、camera_name、frame_id、included_in_public_release、exclude_reason、split_partition_name。共 82,900 行，其中 True 74,619 行、False 8,281 行（False 的行 split 为空）。公开图像：**train 59,543 / val 7,386 / test 7,690**；scene：**365 / 46 / 46（共 457）**，与 457 个 zip 一一对应，双向差集都为空。 | `hypersim/split_v1_summary.json` |
| 按划分合计 zip 大小（HEAD 精确值） | train 1,520,262,631,540 B；val 193,257,517,392 B；test 200,768,757,207 B | `hypersim/scene_zip_sizes_summary.json` |
| 按文件类型拆分（3 个 scene 的 zip 中央目录，range read 共 2.13 MB） | ai_001_001（98 帧）、ai_010_005（100 帧）、ai_035_001（200 帧）。四类文件每帧压缩后字节：`final_hdf5/color.hdf5` 约 3.16–3.46 MB，`geometry_hdf5/depth_meters.hdf5` 约 0.18–0.24 MB，`geometry_hdf5/normal_cam.hdf5` 约 0.67–1.18 MB，`final_preview/tonemap.jpg` 约 0.05–0.08 MB。这四类合计占 zip 字节的 **17.4% / 22.6% / 18.1%**。 | `hypersim/zip_filetype_breakdown.json`、`hypersim/zip_listing_*.tsv` |
| 子集大小（**外推**，不是实测） | 3 个样本 scene 的四类文件每帧合计 4.08–4.95 MB，乘以 74,619 张得约 **305–370 GB**。按占比 17.4–22.6% 乘以 1.914 TB 得约 **333–432 GB**。如果只取 tonemap.jpg、depth_meters、normal_cam 三类，每帧 0.92–1.49 MB，乘以 74,619 得约 **69–111 GB**。 | 由上面两行计算 |
| HF 非官方镜像（示例） | `ritianyu/Hypersim`：API 列出的文件数恰好是 100,000（是否触到 API 列举上限未核实），合计 20,435,119,249 B，无卡片，无 license | `hf/hfinfo_ritianyu__Hypersim.json` |

---

## 未能核实汇总

- DenseDepth `nyu_data.zip` 与 `nyu_test.zip` 的真实字节数：S3 返回 403，tinyurl 指向的 Drive 文件夹返回 404。
- FastDepth/sparse-to-dense 原始 tar.gz（datasets.lids.mit.edu）：连接超时。TFDS 页面的数字是间接来源。
- BTS sync.zip 的精确字节数：Drive 只显示 "5.9G"。aradhye 镜像是否与它逐字节相同：未验证。
- Eigen 作者页（~deigen/depth、~deigen/dnl）现在 403，只能看 Wayback 快照。Eigen NIPS14 论文正文（crop 的文字定义）没有读，因为 PDF 有 6.6 MB，为节省预算没有下载。
- Ladický 2014 的 `nyu_normals_gt.zip`：原站失效，Wayback 没有存档。
- Eigen `normals_gt.tgz` 的成员总数。GeoNet 30,907 张训练集在 SharePoint 上的大小（只在 Bae 2021 README 里看到这个张数）。
- DIW（BLINK 图像来源）的许可。FastDepth 47,584 帧所属 scene 与官方 test scene 是否重叠：需要逐个遍历 tar 头，没有做。

## 值得注意的事实（只陈述不解读）

1. NYU 官方主页**没有任何许可或条款文字**，只要求引用。多个 HF 镜像各自标了 mit、apache-2.0、other，或者不标。
2. `sayakpaul/nyu_depth_v2` 的许可有三种写法：YAML 为 apache-2.0，脚本为 "Apace 2.0 License"，正文为 MIT。它的 47,584 帧来自 FastDepth/sparse-to-dense，**不是 BTS 的 24,231 帧**。
3. "~24K 训练帧"这一说法与 BTS 清单相符，实测 24,231 对。BTS 的 sync 目录（按 aradhye 镜像）共有 36,396 对，比清单多。DenseDepth 自称 "50K"，FastDepth 为 47,584。这四个数互不相同。
4. BTS 训练清单里有 1,383 帧来自 raw 目录 `bookstore_0001d…j`。去掉字母后缀后，名字 `bookstore_0001` 是官方 **test** 划分里的 scene（官方 train 用的是 bookstore_0000 和 0002）。BTS 清单中没有任何目录名与官方 test scene 名完全相同。
5. BTS 的测试 GT 取自 `rawDepths`（未补洞），而 sayakpaul/FastDepth 的 val 样本 depth 没有零值（只看了 2 个样本）。
6. crop 的下标写法：toolbox 是 MATLAB `mask(45:471, 41:601)`，即 1-based 闭区间，427×561；BTS 是 Python `eval_mask[45:471, 41:601]`，即 0-based 半开区间，426×560。两者的起点差 1 像素，尺寸差 1×1。
7. 同一份 654 张测试集，不同来源的编号基不同：`splits.mat` 为 1-based（1..1449），BTS 文件名和 Marigold 为 0-based（0..1448），GeoNet-test 的 HF 镜像为 1-based。
8. 有些 HF "NYUv2" 镜像用的不是官方 795/654 划分：jagennath-hari 为 1014/290/145，WyettZ 为 797/652，idsia 为自定义 48×48 划分。`Zhonghua/NYUDv2` 是模型权重，不是数据。
9. BLINK Relative_Depth 的 **test 答案一律是 "hidden"**。官方唯一的 test 评测渠道 EvalAI challenge 2287 已于 **2026-01-31 结束**（is_active=False）。val 124 题的答案是公开的。
10. Hypersim README 写每个 zip 在 "between 1GB and 20GB"，但实测有 4 个小于 1e9 B（最小 77.9 MB），有 2 个大于 20e9 B（最大 20.59 GB）。README 写 461 个 scene，而下载脚本和 v1 划分都是 457 个 scene。
11. GitHub `apple/ml-hypersim` 已迁移到 `apple-aiml-research/ml-hypersim`。
12. NYU labeled 集的 sceneTypes 实测有 27 种（含拼写 "excercise_room"），而 sayakpaul 卡片转引的论文原文是 "26 scene classes"。
