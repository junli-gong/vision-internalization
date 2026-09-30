# RUN-0001 §6 数据源事实 · 组 C

范围：ADE20K、CV-Bench、EmbSpatial-Bench、ODinW13，以及"哪些基准只有 test/eval split"。
采集时间：2026-09-30 05:49–06:11 UTC，机器 b200。本文只记录事实，不给建议。

**证据目录：** `$PROJ/logs/0001/p6/C/`，下文简称 `C/`。表中每个数字后面标了来源文件或 URL。
**方法：**
- HF Hub API：`dataset_info(files_metadata=True)` 和 `list_repo_tree`。
- datasets-server JSON API：`/splits`、`/size`、`/info`、`/first-rows`、`/rows`、`/statistics`。
- HTTP HEAD。
- 远程 zip 中央目录：用 HTTP Range 读取（`scripts/p6_C_zip.py`，由 `scripts/remote_zip.py` 改写，增加了网络字节计数）。
- 少量小文件下载，逐条记在 `C/downloads.tsv`；用完已全部删除。

证据脚本共 11 个，均为 `$PROJ/scripts/p6_C_*.py`：hfsearch、hfmeta、html2txt、zip、cvbench_rows、cvbench_tab、sunsplit、dsrows、odinw_members、odinw_fetch、tables。

---

## C1. ADE20K

### C1.1 候选来源

| 来源 | 性质 | URL / repo | 大小 | 访问方式 |
|---|---|---|---|---|
| **MIT Scene Parsing Benchmark（SceneParse150）ADEChallengeData2016** | **官方**（150 类语义分割 benchmark） | https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip（网页 http://sceneparsing.csail.mit.edu/ 上的 "train/val (922MB)" 链接；http 会 301 跳到 https） | HEAD Content-Length **967,382,037 B**，Last-Modified 2022-07-22（`C/url_head_checks.txt`） | 公开直链，无需注册；支持 Accept-Ranges |
| SceneParse150 测试集 release_test.zip | 官方 | https://data.csail.mit.edu/places/ADEchallenge/release_test.zip | HEAD **211,820,497 B** | 公开直链。只有图像，没有标注（C1.3） |
| ADE20K 完整数据集 2016 版 ADE20K_2016_07_26.zip | 官方（完整 ADE20K，不是 150 类 benchmark） | https://ade20k.csail.mit.edu/ADE20K_2016_07_26.zip（旧地址 groups.csail.mit.edu/vision/datasets/ADE20K/… 会 301 跳到这里） | HEAD **4,051,732,024 B**，Last-Modified 2021-02-07 | HEAD 返回 200，未要求登录（本次未下载） |
| ADE20K 完整数据集 2021 版（ADE20K_2021_17_01） | 官方 | 官网 "Register here"：https://ade20k.csail.mit.edu/request_data/ | 猜测的直链 `…/ADE20K_2021_17_01.zip` 返回 **404**（`C/url_head_checks.txt`） | 需注册并同意 Terms of Use。GitHub README 原文："To download the dataset, register in this link. Once you are approved you will be able to download the data" |
| ADE20K index_ade20k.pkl | 官方工具索引 | https://ade20k.csail.mit.edu/toolkit/index_ade20k.pkl | HEAD 817,537,016 B | 公开 |
| HF `zhoubolei/scene_parse_150` | HF 上的 loader 脚本（仓库本身不存数据，只有 README.md 和 scene_parse_150.py，共 73,010 B） | https://huggingface.co/datasets/zhoubolei/scene_parse_150 | 卡片 download_size 1,179,202,534 | 需执行 Python 脚本。datasets-server 返回 501 "runs arbitrary Python code"（`C/hf/zhoubolei__scene_parse_150.json`） |
| HF `1aurent/ADE20K` | 第三方 parquet 镜像，内容是**完整 ADE20K**（不是 150 类） | https://huggingface.co/datasets/1aurent/ADE20K | HF API 合计 5,935,269,198 B；train 25,574 行，validation 2,000 行（`C/hf/1aurent__ADE20K.json`） | 公开，license tag `bsd` |
| 其他 HF 镜像 | 第三方 | 见 `C/hf_search.txt`（如 `qubvel-hf/ade20k-mini`、`nateraw/ade20k-tiny`、`uva-cv-lab/ADE20k-150` 等） | — | 本次未逐个核实 |

### C1.2 许可 / 条款（原文引用）

- **SceneParse150 页面** http://sceneparsing.csail.mit.edu/ 页脚：写 "License BSD 3-Clause License"，链接到 https://github.com/CSAILVision/sceneparsing/blob/master/LICENSE（`C/ade_sceneparsing_site.txt`）。该 LICENSE 首行为 "BSD 3-Clause License / Copyright (c) 2019, MIT CSAIL Computer Vision"。
- **ADE20K Terms of Use** https://ade20k.csail.mit.edu/terms/（`C/ade_terms_page.txt`）：
  - 图像部分："MIT, CSAIL does not own the copyright of the images. … Researcher shall use the Database only for non-commercial research and educational purposes. … If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions"
  - 标注部分："Software and Annotations — This website, image annotations and the software provided belongs to MIT CSAIL and is licensed under a Creative Commons BSD-3 License Agreement"
- HF `zhoubolei/scene_parse_150` 卡片："The MIT Scene Parsing Benchmark dataset is licensed under a BSD 3-Clause License"。

### C1.3 Split 与计数

数据来自 ADEChallengeData2016.zip 的中央目录（`C/ade_scene_parsing_ziplist.txt`，统计结果在 `C/ade_scene_parsing_split_counts.txt`）：

| 目录 | 文件数 | 未压缩字节 | 压缩后字节 |
|---|---:|---:|---:|
| images/training/*.jpg | **20,210** | 793,871,532 | 788,022,012 |
| annotations/training/*.png | **20,210** | 84,362,657 | 80,204,814 |
| images/validation/*.jpg | **2,000** | 80,272,583 | 79,663,754 |
| annotations/validation/*.png | **2,000** | 8,702,235 | 8,265,910 |
| 其他 | objectInfo150.txt（5,689 B）、sceneCategories.txt（646,429 B） | | |

- release_test.zip（`C/ade_release_test_ziplist.txt`）含 **3,352** 张 `release_test/testing/*.jpg`，另有 list.txt 和 readme.txt，**没有标注 png**。
- HF `scene_parse_150` 卡片记的 split：train 20,210 / validation 2,000 / test 3,352。
- 完整 ADE20K（2021）按 GitHub README：
  - "27,574 images (25,574 for training and 2,000 for testing) spanning 365 different scenes"
  - "707,868 unique objects from 3,688 categories"
  - 来源：`C/ade_ADE20K_github_README.md`

### C1.4 类别数与标签编码（官方文档）

- sceneparsing README（`C/ade_sceneparsing_README.md`）原文："There are in total 150 semantic categories included in the benchmark for evaluation"；"Note: annotations masks contain labels ranging from 0 to 150, where 0 refers to "other objects". We do not consider those pixels in our evaluation."；"As mentioned before, we ignore pixels labeled with 0's."
- objectInfo150.txt（`C/ade_objectInfo150.txt`）共 150 行，列为 `Idx Ratio Train Val Name`，Idx 从 1 到 150。第 1 类是 wall，第 150 类是 flag。
- 实测两张标注 PNG（`C/ade_annotation_png_probe.txt`）：
  - 格式：mode `L`，uint8，单通道。
  - ADE_val_00000001.png：值域 0–18，0 占 1.03%。
  - ADE_train_00000001.png：值域 0–150，0 占 9.84%。
- 完整 ADE20K 的 `*_seg.png` 用另一套编码。官网原文："The channels R and G encode the objects class masks. The channel B encodes the instance object masks"，而且类别表有 3,688 类，不是 150 类（`C/ade_homepage.txt`、`C/ade_ADE20K_github_README.md`）。

### C1.5 我们需要的子集大小

- train（20,210 张图 + 20,210 张标注）和 val（2,000 + 2,000）都在同一个 zip 里：**967,382,037 B（HEAD）**。
- 分开算（压缩后）：train 图像约 788.0 MB、标注约 80.2 MB；val 图像约 79.7 MB、标注约 8.3 MB。

---

## C2. CV-Bench（nyu-visionx/CV-Bench）

### C2.1 来源与许可

- **官方：** HF `nyu-visionx/CV-Bench`（sha `bc284db5…`，未 gated），出自 Cambrian-1 论文 arXiv:2406.16860。
- 卡片 license：**apache-2.0**（`C/hf/nyu-visionx__CV-Bench.README.md`）。
- 仓库文件（HF API，`C/hf/nyu-visionx__CV-Bench.json`）：

  | 文件 | 字节 |
  |---|---:|
  | test_2d.parquet | 184,906,137 |
  | test_3d.parquet | 219,902,227 |
  | test_2d.jsonl | 820,585 |
  | test_3d.jsonl | 1,069,448 |
  | build_img.py | 1,909 |
  | 合计 | 406,708,749 |

- 图像上游许可：卡片只写 "By repurposing standard vision benchmarks, ADE20k, COCO and OMNI3D"，没有另外声明上游许可。
- Omni3D 标注的许可：Omni3D_json.zip 里的 LICENSE.md 写的是 "Omni3D and Cube R-CNN License … Attribution-NonCommercial 4.0 International"（`C/omni3d_LICENSE.md`）。
- COCO 的许可本次未核实。

### C2.2 Split、计数与任务

- datasets-server `/splits` 和 `/size`（`C/cvbench_splits.json`、`C/cvbench_size.json`）：3 个 config，**每个都只有 `test` split**。
  - default：2,638 行
  - 2D：1,438 行
  - 3D：1,200 行
- 按 `/rows` 分页取回全部 2,638 行，只保留非图像列（`C/cvbench_rows_meta.jsonl`，脚本 `p6_C_cvbench_rows.py`，`partial: False`），统计见 `C/cvbench_source_tab.txt` / `.json`：

| type / task | 行数 |
|---|---:|
| 2D / Count | 788 |
| 2D / Relation | 650 |
| 3D / Depth | 600 |
| 3D / Distance | 600 |

与论文 Table 1 一致：Spatial Relationship 650、Object Count 788、Depth Order 600、Relative Distance 600。

### C2.3 图像来源与所用 split（逐行字段统计）

| source | source_dataset（原值） | 行数 | 去重后的源图像数 | 源文件名模式 |
|---|---|---:|---:|---|
| ADE20K | "ADE20K Validation Set" | 633 | 633 | `ADE_val_#.jpg`（633/633） |
| COCO | "COCO 2017 Validation Set" | 805 | 805 | `/coco/val2017/#.jpg`（805/805） |
| Omni3D | "Omni3D_Hypersim" | 400 | 242 | `hypersim/ai_XXX_YYY/images/scene_cam_NN_final_preview/frame.NNNN.tonemap.jpg` |
| Omni3D | "Omni3D_SUNRGBD" | 400 | 149 | `SUNRGBD/<sensor>/<subset>/…/image/*.jpg` |
| Omni3D | "Omni3D_nuScenes" | 400 | 283 | `nuScenes/samples/CAM_FRONT/…jpg` |

- 各 source_dataset 与 task 的交叉：ADE 为 Count 342 / Relation 291；COCO 为 Count 446 / Relation 359；Hypersim、SUNRGBD、nuScenes 各为 Depth 200 / Distance 200。
- 全表去重后的 (source_dataset, source_filename) 共 **2,112** 个；CV-Bench 自己的 `filename` 有 2,638 个，每行一个。
- 交叉核对：datasets-server 返回的 parquet 行，与仓库里的 test_2d.jsonl + test_3d.jsonl（下载后已删除）在 (type, task, filename, source, source_dataset, source_filename, question, answer) 上**完全一致**：jsonl 2,638 行，两边各自独有的行都是 0。
- 卡片原文："`source` | Source of the image: `ADE20K`, `COCO`, or `Omni3D`"；"`source_dataset` | More detailed source of the image"。
- 论文（https://arxiv.org/html/2406.16860v2）原文：
  - "The examples are sourced from ADE20K [155], COCO [80], and Omni3D [18]"
  - 脚注 "Omni3D assets are sourced from [47, 20, 122, 13, 112, 3]"，对应 KITTI、nuScenes、SUN RGB-D、ARKitScenes、Hypersim、Objectron。
  - 论文正文**没有写** Omni3D 用的是哪个 split。

**Omni3D 部分各自来自哪个 split（本次自行核对）：**

- **SUN RGB-D**（`C/cvbench_sunrgbd_split_check.json`，脚本 `p6_C_sunsplit.py`）：
  - 149 个去重序列**全部**在 SUN RGB-D 官方 `allsplit.mat` 的 `alltest` 中（5,050 个序列），覆盖 400/400 行；在 `alltrain` 中为 0。`allsplit.mat` 取自 SUNRGBDtoolbox.zip 的 `traintestSUNRGBD/allsplit.mat`。
  - 这些图像在 Omni3D 自己的 `SUNRGBD_val.json`（356 张）中为 0。
  - 按传感器目录分：kv1/NYUdata 89 行、kv1/b3dodata 1、kv2/kinect2data 203、kv2/align_kv2 17、realsense/lg 3、xtion/sun3ddata 85、xtion/xtion_align_data 2。
- **Hypersim**（`C/cvbench_hypersim_split_check.txt`）：242 张去重图像**全部**标为 `test`，覆盖 400/400 行，来自 23 个场景。对照表是 Hypersim 官方 `metadata_images_split_scene_v1.csv`（apple/ml-hypersim，按 (scene, camera, frame) 匹配）。
- **nuScenes**：**未能核实**所属 split。原因：需要 nuScenes 的 scene/log 元数据（v1.0-trainval_meta，远超预算），或 Omni3D 的 `nuScenes_test.json`（75,433,550 B，超过单文件 50 MB 上限）。

**是否含 NYUv2 图像：含。**

- Omni3D_SUNRGBD 行中，路径形如 `SUNRGBD/kv1/NYUdata/NYU####/…` 的共 **89 行**，对应 **36 张**去重图像。按任务分：Depth 53 行，Distance 36 行。
- 示例：`SUNRGBD/kv1/NYUdata/NYU0021//image/NYU0021.jpg`、`…/NYU0035//image/NYU0035.jpg`、`…/NYU0076//…`，完整 id 列表在 `C/cvbench_nyu_split_map.json`。
- **判定方法：** 在 source_filename 中匹配正则 `NYU(\d{4})`。
- 把 `NYU####` 的编号当作 NYUv2 labeled 集（1,449 张）的 1-based 索引，与 NYUv2 官方 `splits.mat` 对照（trainNdxs 795 / testNdxs 654，来自 https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat）：**36/36 张都在 NYUv2 官方 test（654）中，在 train 中为 0**；89/89 行同样全在 test。
- 编号映射的旁证（`C/sunrgbd_vs_nyu_split_check.txt`）：SUN RGB-D `alltrain` 中的 NYU id 集合（795 个），与 NYUv2 `trainNdxs` **完全相等**；`alltest` 中的 NYU id 集合（654 个）与 `testNdxs` **完全相等**。
- 逐张图像像素级的同一性**未核实**（没有下载图像）。

### C2.4 真实 schema（datasets-server `/rows` 返回的 features，`C/cvbench_rows_run.txt`）

```
idx int32 | type string | task string | image Image | question string | choices List<string> | answer string |
prompt string | filename string | source string | source_dataset string | source_filename string |
target_class string | target_size int32 | bbox List<List<float32>>
```
- answer 形如 `"(C)"`。
- bbox 只在 Omni3D 行有值，例如 `[[373.91,432.78,670.32,583.83],[177.79,274.94,351.70,343.55]]`。
- target_class 和 target_size 按卡片说明只对 COCO 行有值。

### C2.5 大小

- 需要的就是整个 test，parquet 合计 **404,808,364 B**。default config 与 2D+3D 是同一批数据（`/size`）。
- 只要文本和元数据的话：test_2d.jsonl + test_3d.jsonl 共 1,890,033 B。

---

## C3. EmbSpatial-Bench

### C3.1 来源与许可

| 来源 | 性质 | 内容 | 许可 |
|---|---|---|---|
| GitHub https://github.com/mengfeidu/EmbSpatial-Bench | **官方**代码与说明（`C/embspatial_github_README.md`） | 数据链接指向 HF `Phineas476/EmbSpatial-Bench`，原文："We share the datset in: EmbSpatial-Bench (containing our benchmark data and sft data) for model evaluation and training." | GitHub API 返回 `license: null`，仓库没有 LICENSE 文件 |
| HF `Phineas476/EmbSpatial-Bench` | **官方数据发布**（作者 README 指向此处） | `embspatial_bench.json`：253,512,493 B（图像以 base64 内嵌）<br>`embspatial_sft.json`：1,201,627,311 B | 卡片 `license: cc-by-4.0`（`C/hf/Phineas476__EmbSpatial-Bench.json`） |
| HF `FlagEval/EmbSpatial-Bench` | 第三方转成的 parquet | `data/test-00000-of-00001.parquet`：408,685,444 B | 卡片**没有 license 字段**，原文："This dataset is organized and adapted from Phineas476/EmbSpatial-Bench" |
| 论文 | arXiv:2406.05756 | | — |

- datasets-server 对 Phineas476 仓库给出一个自动命名的 `default/train` split，但 size/info 计算失败（`failed: config-size`）。这个 "train" 名称是自动生成的，不对应论文里的任何 split。

### C3.2 Split 与计数

- FlagEval parquet：**只有 `test`，3,640 行**。来源是 datasets-server `/size`（`C/hf/FlagEval__EmbSpatial-Bench.json`）和 `/statistics`（`C/embspatial_flageval_statistics.json`，`partial: False`）。
- `/statistics` 结果：
  - `data_source`：mp3d 1,201 / ai2thor 1,239 / scannet 1,200
  - `relation`：left 616 / right 620 / above 596 / under 602 / close 612 / far 594
  - `scene_id` 去重 294 个：ScanNet 175 个 scan（来自 90 个不同的 scene 编号）、MP3D 26、AI2-THOR 93
  - `answer` 取值 0–3；`answer_options` 恒为 4 个
- 行顺序：offset 0–1200 为 mp3d，1201–2439 为 ai2thor，2440–3639 为 scannet。依据 `/rows` 在边界 offset 上的抽样（`C/embspatial_flageval_rows_off*.json`）。
- 论文 Table 1 原文："Matterport3D 1,201 928 133 26 | AI2-THOR 1,239 683 95 93 | ScanNet 1,200 570 35 175 | Overall 3,640 2,181 294 277"，列依次为 #QA Pairs、#Image、#Object、#Scene。
- **训练集：** 有，是单独的 EmbSpatial-SFT（`embspatial_sft.json`，1.2 GB）。论文原文："EmbSpatial-SFT is solely built on the training split of MP3D"；"We construct 25K training samples for spatial relation identification"。

### C3.3 图像来源（原文引用）与场景 split 核对

- 论文："The images used for evaluation are directly collected from embodied 3D scenes, namely MP3D (Chang et al., 2017), AI2-THOR (Kolve et al., 2017) and ScanNet (Dai et al., 2017)"；"Specifically, we utilize the test scenes from MP3D and validation scenes from ScanNet and A[I2-THOR]"。arXiv HTML 在这里原文被截断成 "and A."。另有："In AI2-THOR, we select 7 types of household tasks from ALFRED"。
- 用官方 split 列表对照（`C/embspatial_scene_split_check.txt`）：
  - **ScanNet**：175/175 个 scan 都在 `scannetv2_val.txt`（312 个）中，在 `scannetv2_train.txt` 中为 0，覆盖 1,200/1,200 行。列表来自 ScanNet/ScanNet GitHub 的 `Tasks/Benchmark/`。
  - **MP3D**：26 个场景中，**17 个**在 `scenes_test.txt`（18 个）中，**9 个**在 `scenes_val.txt`（11 个）中，在 train（61 个）中为 0。按行算，1,131 行来自 test 场景，70 行来自 val 场景。列表来自 niessner/Matterport GitHub 的 `tasks/benchmark/`。
  - **AI2-THOR**：共 93 个 FloorPlan，编号覆盖 1–30、201–230、301–330、401–430 中的大部分，完整列表见上面的 check 输出。ALFRED 的 split 定义**未核实**。

### C3.4 真实 schema（FlagEval，datasets-server `/first-rows` 的 features，`C/embspatial_flageval_first_rows.json`）

```
data_source string | scene_id string | question_id string | question string | relation string | image Image |
answer_options Sequence<string> | answer int32 | objects Sequence{bbox: Sequence<int32, length 4>, name: string} |
image_base64 string
```
- FlagEval 卡片说 bbox 是 "(x_min, y_min, x_max, y_max)"。实测抽样见 C3.5 和末节。
- 图像尺寸：mp3d 640×480，ai2thor 300×300，scannet 1296×968。依据抽样行和 `/statistics` 的 image 列（min 300，max 1296）。
- `image_base64` 列长度 6,576–285,428 字符，均值 69,133。

### C3.5 bbox 格式实测（抽样 10 行，`C/embspatial_flageval_rows_off*.json`、`C/embspatial_flageval_first_rows.json`）

- mp3d 行：`[343,406,102,74]`、`[361,1,279,53]`（图宽 640、高 480）。按 xywh 读，y+h=480、x+w=640，恰好落在边缘。官方 GitHub README 的 mp3d 示例 `[123,256,130,73]` 按 xyxy 读会出现 y2<y1。
- scannet 行：`[596,508,444,319]`，按 xyxy 读 x2<x1；按 xywh 读 `[186,538,357,429]` 得 y+h=967（图高 968）。
- ai2thor 行：`[68,245,210,269]` 按 xywh 读 y+h=514，超过图高 300；按 xyxy 读则落在图内。`[277,1,299,47]`、`[253,1,299,29]` 的 x2=299，即图宽减 1。

### C3.6 大小

- 需要的是整个 test：FlagEval parquet 408,685,444 B，或 Phineas476 的 `embspatial_bench.json` 253,512,493 B。
- SFT 1,201,627,311 B。

---

## C4. ODinW13（GLIP 使用的 13 个数据集）

### C4.1 数据集清单与官方来源

- **13 个数据集的定义：** GLIP 仓库 `configs/odinw_13/` 下的 13 个 yaml（microsoft/GLIP main 分支，tree sha `9dda9558…`，路径列表见 `C/glip_git_tree_odinw.txt`；yaml 原件在 `C/odinw_glip_configs13/`）。
- GLIP README 原文："[4] Zero-shot performance on the 13 ODinW datasets"。
- **官方下载脚本** `odinw/download.py`（`C/odinw_glip_download.py`）：
  - `root = "https://huggingface.co/GLIPModel/GLIP/tree/main/odinw_35"`
  - 执行 `wget root/<name>.zip`
  - 列出 33 个数据集名
- 该 `…/tree/main/…zip` 形式的 URL，HEAD 返回 **HTTP 404 text/html**（`C/url_head_checks.txt`）。实际可用的直链是 `…/resolve/<sha>/odinw_35/<name>.zip`，会 302 到 HF CDN。
- GLIP README 里 Aquarium 标注的旧直链 `penzhanwu2bbs.blob.core.windows.net/...`：**DNS 无法解析**（`C/url_head_checks.txt`）。
- HF `GLIPModel/GLIP`（model repo，sha `66ee3ae9…`，last_modified 2023-07-06）的 `odinw_35/` 下有 34 个 zip，合计 27,392,576,563 B（`C/odinw_glip_hf_tree.txt`）。仓库卡片没有 license。
- 其他 HF 镜像：
  - `AdemMarshall/odinw-13-mirror`：README 原文 "Byte-identical copy of the thirteen ODinW-13 dataset archives from GLIPModel/GLIP at revision 66ee3ae9…"。13 个 zip 的字节数与 GLIPModel/GLIP 完全相同（`C/hf/AdemMarshall__odinw-13-mirror.json`）。license: other。
  - `kcz358/ODinW-13`：只有一个 parquet split `test`，4,938 行，214,297,376 B；schema 为 `image, dataset_name, categories, bboxes, labels`。

### C4.2 各数据集的 split 计数与大小

- 图像数和框数来自 GLIP config 注册的 `annotations_without_background.json`：从 zip 中按 Range 读出，统计后已删除，统计结果在 `C/odinw13_ann_stats.json`。
- zip 大小来自 HF API（`C/odinw_glip_hf_tree.txt`）。
- 表由 `p6_C_tables.py` 生成（`C/tables_rendered.md`）。
- 单元格格式为"图像数/框数"。

| # | GLIP config（configs/odinw_13） | zip | zip 字节（HF API） | train | valid | test | minival | GLIP `DATASETS.TEST` | 类别数 | License（README.dataset.txt） |
|---|---|---|---:|---|---|---|---|---|---:|---|
| 1 | AerialMaritimeDrone_large | AerialMaritimeDrone.zip | 47,681,829 | 52/873 | 15/200 | 7/78 | — | ("val",) | 5 | Public Domain |
| 2 | Aquarium_Aquarium_Combined.v2-raw-1024.coco | Aquarium.zip | 70,037,905 | 448/3324 | 127/909 | 63/584 | — | ("val",) | 7 | CC BY 4.0 |
| 3 | CottontailRabbits | CottontailRabbits.zip | 153,305,581 | 1980/2070 | 19/21 | 10/11 | — | ("val",) | 1 | Public Domain |
| 4 | EgoHands_generic | EgoHands.zip | 720,990,333 | 3840/12015 | 480/1515 | 480/1514 | 200/621 | ("minival",) | 1 | CC BY 4.0 |
| 5 | NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco | NorthAmericaMushrooms.zip | 9,804,594 | 41/67 | 5/14 | 5/9 | — | ("val",) | 2 | Public Domain |
| 6 | Packages_Raw | Packages.zip | 10,768,880 | 19/31 | 4/6 | 3/5 | — | ("val",) | 1 | Public Domain |
| 7 | PascalVOC | PascalVOC.zip | 740,774,846 | 13690/31356 | 3422/7835 | 3422/7835 | — | ("val",) | 20 | CC BY 4.0 |
| 8 | Raccoon_Raccoon.v2-raw.coco | Raccoon.zip | 27,092,904 | 150/164 | 29/29 | 17/20 | — | ("val",) | 1 | MIT |
| 9 | ShellfishOpenImages_raw | ShellfishOpenImages.zip | 153,436,493 | 407/859 | 116/240 | 58/116 | — | ("val",) | 3 | CC BY 4.0 |
| 10 | VehiclesOpenImages_416x416 | VehiclesOpenImages.zip | 87,006,326 | 878/1676 | 250/454 | 126/258 | 200/366 | ("minival",) | 5 | CC BY 4.0 |
| 11 | pistols_export | pistols.zip | 121,434,350 | 2377/2728 | 297/346 | 297/358 | — | ("val",) | 1 | Public Domain |
| 12 | pothole | pothole.zip | 96,656,058 | 465/1256 | 133/330 | 67/154 | — | ("val",) | 1 | ODbL v1.0 |
| 13 | thermalDogsAndPeople | thermalDogsAndPeople.zip | 10,072,076 | 142/181 | 41/49 | 20/27 | — | ("val",) | 2 | undefined |
| Σ | | 13 个 zip | **2,249,062,175** | 24489/56600 | 4938/11948 | 4575/10969 | 400/987 | | | |

补充说明：

- **PascalVOC：** GLIP config 里 `test.ann_file` 与 `val.ann_file` 是同一个文件，`PascalVOC/valid/annotations_without_background.json`。zip 中只有 train/valid/minival 三个图像目录，没有 test 目录；minival 目录有 3,422 张，与 valid 相同（`C/odinw13_zip_image_counts.txt`）。
- **pistols：** 全部 2,971 张图都在 `pistols/export/` 一个目录下（2,377+297+297=2,971）。三个 split 用不同的 json 区分：`train_/val_/test_annotations_without_background.json`。
- **EgoHands：** zip 中 `mini_val/` 目录有 471 张图，而 minival json 只登记了 200 张。
- **kcz358/ODinW-13 的 test（4,938 行）：** 按 dataset_name 统计，其中 EgoHands 480、VehiclesOpenImages 250、PascalVOC 3,422 等，**逐个数据集都等于上表 valid 列的图像数**，总和 4,938（`C/odinw_kcz358_statistics.json`）。
- 每个 zip 里还带有其他变体目录和 few-shot json，例如 AerialMaritimeDrone/tiled、Raccoon.v38-416x416-resize、ShellfishOpenImages/416x416，以及 `fewshot_{train,val}_shot{1,3,5,10}_seed{3,30,300}.json`（`C/odinw_ziplists/`）。

### C4.3 许可（Roboflow `README.dataset.txt` 原文，完整文本存于 `C/odinw13_ann_stats.json` 的 readmes 字段）

- **Public Domain：** AerialMaritimeDrone、CottontailRabbits、NorthAmericaMushrooms、Packages、pistols，共 5 个。标注 json 的 `licenses` 字段指向 CC0 1.0 URL。
- **CC BY 4.0：** Aquarium、EgoHands、PascalVOC、ShellfishOpenImages、VehiclesOpenImages，共 5 个。
  - EgoHands 原文："It is released under a Creative Commons by Attribution 4.0 license. You may use it for academic or commercial purposes but must cite the original paper"
- **MIT：** Raccoon。原文："originally collected by Dat Tran, released with MIT license"
- **ODbL v1.0：** pothole。
- **undefined：** thermalDogsAndPeople。README 为 "License: undefined"，json 里 `licenses: [{"name": "Unknown"}]`，contributor 为空，url 为 `https://public.roboflow.ai/object-detection/undefined`。
- AerialMaritimeDrone 的 README 里同时出现 "License: Public Domain" 和 "This dataset was originally collected by the Roboflow team, released with MIT license"。

### C4.4 标注格式

- 格式为 **COCO detection json**。GLIP README 原文："Currently, the code supports evaluation on COCO-formatted dataset"；"Remove the background class from the annotation file … remove the entry with "id:0" from "categories""。`annotations_without_background.json` 就是按这个做法去掉背景类后的文件。
- 实测字段：
  - 顶层：`categories, images, annotations, info, licenses`
  - images：`id, file_name, width, height, license, date_captured`
  - annotations：`id, image_id, category_id, bbox, area, segmentation([]), iscrowd`
  - bbox 为 COCO 的 `[x, y, w, h]`，可以是浮点，例如 `[870, 342, 30.5, 25.5]`
- GLIP 的 13 个 config 都设 `TRAIN: ("train",)`。

---

## C5. 各基准可用的 split（HF / 官方元数据）

| 基准 | 可用 split（计数） | 依据 |
|---|---|---|
| CV-Bench | **仅 test**：default 2,638 / 2D 1,438 / 3D 1,200 | datasets-server `/splits` `/size`（`C/cvbench_*.json`） |
| EmbSpatial-Bench | benchmark **仅 test** 3,640（FlagEval parquet）；官方 Phineas476 仓库另有单独的训练集 **EmbSpatial-SFT**（json，1.2 GB，来自 MP3D 训练场景，论文称 25K 条关系识别样本） | `C/hf/FlagEval__EmbSpatial-Bench.json`、`C/hf/Phineas476__EmbSpatial-Bench.json`、论文 |
| BLINK（BLINK-Benchmark/BLINK，apache-2.0） | 14 个子任务，**每个都只有 val + test，没有 train**；val 合计 1,901，test 合计 **1,906**；test 的 `answer` 字段值为 `"hidden"`。卡片原文："the 1,907 test questions are available without their answers. You can submit … on EvalAI" | `C/hf/BLINK-Benchmark__BLINK.json`、`C/blink_counting_test_first_rows.json` |
| CountBench | 官方只有一个文件，**不分 split**：`CountBench.json`，540 条 `{image_url, text, number}`，计数 2–10 各 60 条。仓库 teaching-clip-to-count/teaching-clip-to-count.github.io，GitHub license 为 null。HF `nielsr/countbench` 的 split 名为 **`train`**（540 行，其中 49 行 image 为 null）；`vikhyatk/CountBenchQA` 为 **`test`** 491 行（原文："only includes 491 images from the original CountBench dataset, since some of the original URLs can no longer be accessed"） | `C/countbench_*_statistics.json`、上面的 GitHub raw 文件（已删除） |
| ODinW13 | 每个数据集都有 train/valid/test，EgoHands 和 VehiclesOpenImages 另有 minival（C4.2）；GLIP config 的 TEST 用 val 或 minival | C4 |
| RefCOCO（UNC split） | 官方 refer README 原文："They are with different train/val/test split by UNC, Google and UC Berkeley"；HF `jxu124/refcoco`：train 42,404 / validation 3,811 / test 1,975 / testB 1,810（按 ref 计行；**没有名为 testA 的 split**，test 的行数与 lmms 版 testA 相同）；`lmms-lab-encoder/RefCOCO`（原名 lmms-lab/RefCOCO，已改名）：val 8,811 / test 5,000 / testA 1,975 / testB 1,810，**没有 train**。官方 UNC 服务器 bvisionweb1.cs.unc.edu **DNS 无法解析**，refer README 原文："As the webserver is broken (sry about this)"；Wayback 快照 20220413 的 refcoco.zip 为 53,993,993 B | `C/hf_meta_summary.txt`、`C/refcoco_refer_README.md`、`C/url_head_checks.txt` |
| ADE20K（SceneParse150） | train 20,210 / val 2,000 带标注；test 3,352 只有图像 | C1.3 |
| NYUv2 | 官方 labeled 集 1,449 张：`nyu_depth_v2_labeled.mat` HEAD 2,972,037,809 B，官方划分 `splits.mat` 为 train 795 / test 654；另有 407,024 帧未标注 raw（官网原文 "1449 densely labeled pairs … 407,024 new unlabeled frames"）。HF `tanganke/nyuv2`：train 795 / val 654。HF `sayakpaul/nyu_depth_v2`：卡片写 train 47,584 / validation 654，datasets-server 只转换了一部分（train 3,100 行，`partial: True`）。官网页面文本中**没有找到** license 或 terms 字样 | `C/nyuv2_official_page.txt`、`C/hf_meta_summary.txt`、`C/url_head_checks.txt` |

---

## 值得注意的事实（只陈述不解读）

1. **CV-Bench 含 NYUv2 图像。** 共 36 张 NYUv2 图（89 行），都在 NYUv2 官方 **test**（654 张）中。CV-Bench 的全部 400 行 SUN RGB-D 数据，都来自 SUN RGB-D 官方 test 序列；400 行 Hypersim 数据都来自 Hypersim 官方 test。见 C2.3。
2. CV-Bench 的 ADE20K 部分全部来自 **ADE20K val**（633 张），COCO 部分全部来自 **COCO val2017**（805 张）。
3. Cambrian-1 论文 Table 21 原文："CV-Bench 2,638 0 (0.00%) 758 (28.73%) 336 (12.74%)"。三列依次是与 Data Engine 161k、Cambrian10M、LLaVA-665k 的**图像哈希重合数**。
4. CV-Bench 卡片 license 为 apache-2.0；它 Omni3D 部分的 Omni3D 标注许可是 **CC BY-NC 4.0**。
5. **EmbSpatial 的 MP3D 部分：** 论文写 "test scenes from MP3D"，但按 Matterport 官方 `scenes_*.txt` 对照，26 个场景中有 **9 个属于官方 val**（70 行），17 个属于 test（1,131 行）。
6. **EmbSpatial 的 bbox 格式：** FlagEval 卡片说是 xyxy。抽样中 mp3d 和 scannet 行的数值只能按 **xywh** 解释，ai2thor 行只能按 **xyxy** 解释（C3.5）。官方 GitHub README 的 mp3d 示例同样是 xywh 形态。
7. **EmbSpatial 场景总数：** 论文 Table 1 的 "Overall #Scene" 写 277，但分项 26+93+175=294。datasets-server 统计的 `scene_id` 去重数也是 294。
8. **EmbSpatial 许可来源不一致：** 官方数据仓库 Phineas476 标 cc-by-4.0；FlagEval 转换版没有 license；官方 GitHub 没有 LICENSE。
9. **BLINK test 行数：** datasets-server 合计 test 为 **1,906** 行，卡片写 "Test (1,907)"。val 两边一致，都是 1,901。
10. **CountBench 图像来源：** 官方发布只有 540 个 LAION 图像 URL，没有图像文件。HF 镜像里有 49 个 URL 已失效，剩 491 张，失效后 2–10 各类计数不再均衡：54/57/55/44/58/52/57/60/54。HF `nielsr/countbench` 把整个基准放在名为 `train` 的 split 里。
11. **ODinW13 评测 split：** GLIP 官方 13 个 config 的 `DATASETS.TEST` 是 **val**（11 个）或 **minival**（EgoHands、VehiclesOpenImages），没有一个是 test。其中 PascalVOC 的 test 标注文件就是 valid 文件。HF `kcz358/ODinW-13` 叫 "test" 的 4,938 行，实际等于 13 个数据集 **valid** split 的图像数之和。
12. **ODinW 官方下载链接：** `odinw/download.py` 用的 `…/tree/main/odinw_35/<name>.zip` 形式 URL，HEAD 返回 404；GLIP README 里的 Azure blob 直链域名无法解析。可用的是 `GLIPModel/GLIP` 的 `/resolve/` 直链。
13. **ODinW13 许可：** 并不统一，包括 5 个 Public Domain、5 个 CC BY 4.0、1 个 MIT、1 个 ODbL v1.0、1 个 "undefined"（thermalDogsAndPeople）。镜像 `AdemMarshall/odinw-13-mirror` 的 README 写 "six Public Domain, five CC BY 4.0, one MIT, one ODbL"，与 zip 内 README 不一致。
14. **ADE20K 的 split 说法不统一：**
    - 官方 GitHub README 写完整 ADE20K 为 "25,574 for training and 2,000 for testing"，官网写同一批 2,000 张为 "Validation set"。
    - sceneparsing README 写 SceneParse150 测试集 "contains 2000 images"，实际 release_test.zip 中有 **3,352** 张。
15. **ADE20K 的使用条款：** 图像为"仅限非商业研究与教育用途"（MIT 不拥有图像版权），标注和软件为 BSD-3。SceneParse150 页面只标 BSD 3-Clause。
16. **HF 镜像结构与官方不同：**
    - `1aurent/ADE20K` 的行数（25,574/2,000）与完整 ADE20K 相同，schema 含 RGB `segmentations`、`objects`（带 polygon、parts）等字段，不是 150 类的 ADEChallengeData2016。本次未核实它的类别表。
    - `zhoubolei/scene_parse_150` 是脚本型仓库，datasets-server 不支持。
17. **sayakpaul/nyu_depth_v2 许可自相矛盾：** YAML 写 license apache-2.0，正文写 "licensed under a MIT License"（fast-depth）。

---

## 下载记账

- 逐条记录在 `C/downloads.tsv`，含 URL、成员名、字节数、用途和删除行。
- **写入磁盘的下载文件合计 23,680,354 B（约 23.7 MB）**，其中 ODinW 标注和 README 为 15,070,110 B（66 个文件）。**全部已删除**：
  - `tmp/p6_C/odinw/` 于 06:06:56 删除。
  - 其余文件于 06:09 删除，删除清单在 `C/tmp_deleted_list.txt`。
  - 删除后 `tmp/p6_C/` 为空。
- **网络流量（Range 读，未落盘）：**
  - zip 中央目录与成员读取：ADE list 和 3 次 get 共约 4×6.9 MB，Omni3D 3.5 MB，ODinW list 6.5 MB。
  - ODinW 成员 get 共 37.5 MB。每次 get 都会重读一遍中央目录，这是主要开销。
  - 另有 datasets-server 和网页 JSON/HTML 流量，约数 MB。
  - 合计约 **80 MB**。
- **diskcheck：**
  - 开始：`/avail=5.901GB`
  - ODinW 拉取前：5.925 GB
  - 结束：`/avail=5.923GB`，shm_visint=21.087GB 未变
  - `C/` 目录合计 11.6 MB，其中没有超过 5 MB 的文件。
