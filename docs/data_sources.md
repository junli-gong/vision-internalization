# data_sources — 数据源的来源、许可、体积与 schema（RUN-0001 §6）

记录者：服务器 CC，2026-09-30。**只记事实，不下结论，不替项目选择来源。** 本次没有落地任何大数据。

## 0. 方法与记账

- 只用以下几种不占盘或低占盘的方式：
  - HTTP HEAD（Content-Length）；
  - HF Hub API（`dataset_info(files_metadata=True)`、`list_repo_tree`）；
  - HF datasets-server JSON API（`/splits`、`/size`、`/info`、`/first-rows`、`/rows`、`/statistics`；图像以 URL 返回，不落盘）；
  - 远端 zip 的 HTTP Range 读取（先读中央目录，再定点取成员；`probe_scripts/remote_zip.py`）；
  - 少量小于 50 MB 的标注样本文件（放在 `$PROJ/tmp`，看完即删，删除记录见各组的 `downloads.tsv`）。
- 执行方式：服务器 CC 开了三个并行子代理，分三组采集。A 组：RefCOCO 系列、COCO val2017、CountBench；B 组：NYUv2、BLINK、Hypersim；C 组：ADE20K、CV-Bench、EmbSpatial-Bench、ODinW13，以及只有测试划分的清单。服务器 CC 审读后合成本文件。原始证据在 `experiments/P0/0001_env_probe/raw/p6/{A,B,C}/`，脚本在 `probe_scripts/p6_{A,B,C}_*.py`。
- 数字的来源格式：字节数写原值。"HEAD" 指 HTTP Content-Length；"HF API" 指 siblings 大小之和；Drive 只有页面显示值（例如 "5.9G"），会原样注明。
- **§6 的下载记账**：

  | 谁 | 写入磁盘的文件（均已删除） | 网络读取（含只进内存的 Range 读） |
  |---|---|---|
  | 服务器 CC（§5.5 / §6 schema 与文档核对） | `instances_val2017.json` 19,987,840 B；Qwen3-VL `2d_grounding.ipynb` 8,691,442 B；README 若干（KB 级） | 约 16.1 MB（instances 只取回 7,340,760 B） |
  | A 组 | 168,736 B | 58,272,991 B，另有约 3 MB 元数据 JSON 未计量 |
  | B 组 | 临时文件 4,643,871 B；另有 7.50 MB 证据文本写入 `raw/p6/B/`（保留） | 30,207,628 B |
  | C 组 | 23,680,354 B（其中 ODinW 标注 15,070,110 B，66 个文件） | 约 80 MB（ODinW 取成员时每次都重读中央目录，是主要开销） |
  | **合计** | **约 57 MB**（单个文件都小于 50 MB） | **约 188 MB** |

  任务书的上限是"单个小于 50 MB、总量小于 200 MB"。无论按落盘字节还是按网络字节算，都在上限内。

## 1. 总表（细节与证据见 §4–§6）

| 数据集（简报用途） | 官方来源 | 许可（原文，或"未声明"） | 划分与条数 | 总体积 | 我们需要的子集 | schema |
|---|---|---|---|---|---|---|
| RefCOCO / RefCOCO+ / RefCOCOg（T 文本定位、多边形分割；T 像素 mask；D 指代分割） | lichengunc/refer。官方主机 `bvisionweb1.cs.unc.edu` 今天 DNS 解析失败；可用的是 issue #14 给出的 2022-04-13 Wayback 快照 | refer 仓库 Apache-2.0（代码）；**数据许可未声明**（未能核实）。RefCOCOg-google（Google Refexp）为 CC BY 4.0，但原始 zip 返回 403 | RefCOCO(unc)：refs 50,000，train/val/testA/testB = 42,404/3,811/1,975/1,810；RefCOCO+(unc)：49,856；RefCOCOg(umd)：49,822，train/val/test = 42,226/2,573/5,023 | 标注 zip：53,993,993 / 45,613,210 / 56,712,951 B | 图像来自 COCO train2014。三者去重并集 **30,956 张，5,252,007,403 B**（逐张 HEAD，全部 200）；只取评测划分为 6,549 张，1,105,415,725 B；逐张 URL 可下载 | refs pickle（A §1.2）与 HF 镜像的 first-rows（A §1.7） |
| COCO val2017（开放词汇检测，500 张子集） | cocodataset.org | 标注为 CC BY 4.0；图像须遵守 Flickr 条款，逐图许可中 69.5% 为 NC 类 | 5,000 张，36,781 条标注，80 类 | val2017.zip 815,585,330 B；annotations_trainval2017.zip 252,907,541 B | 500 张约 81.5 MB（按均值），实测随机子集 80.9–83.9 MB | COCO json（A §2） |
| ODinW13（开放词汇检测） | GLIP `configs/odinw_13`，数据在 HF `GLIPModel/GLIP/odinw_35/*.zip`（`download.py` 的 `/tree/` 链接返回 404，要用 `/resolve/`） | 逐数据集不同：5 个 Public Domain，5 个 CC BY 4.0，1 个 MIT，1 个 ODbL，1 个 undefined | 13 个数据集合计：train 24,489 / valid 4,938 / test 4,575 张图；GLIP 的 TEST 配置用 val 或 minival | 13 个 zip 共 2,249,062,175 B | 由项目定 | COCO detection json（C §4.4） |
| CountBench（计数） | teaching-clip-to-count 项目页的 `CountBench.json` | **未声明**（GitHub license 为 null） | 540 条，计数 2–10 各 60 条，不分划分 | 只有 URL（json 146,701 B）；HF `nielsr/countbench` 打包版 23,353,242 B（491 张有图） | 前 50 个 URL 的 HEAD 中非 200 占 8/50；540 个全测非 200 占 22.8% | `{image_url, text, number}`（A §3） |
| NYUv2（T 相对深度点对；T 像素深度，Eigen 654；D 深度与法向） | cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html | **主页没有任何许可或条款文字**，只要求引用 | labeled 1,449 张：官方 795/654（本地 splits.mat 与官方逐字节相同）；raw 407,024 帧 | labeled .mat 2,972,037,809 B；raw 428,798,889,146 B | "约 2.4 万帧"：BTS 清单 24,231 对，其 sync.zip 在 Drive 显示 "5.9G"；HF `aradhye/nyu_depth_v2` 6,857,931,462 B，含 36,396 对（BTS 的 24,231 对共 4,240,085,416 B）；FastDepth 47,584 帧，HF `sayakpaul` 35,151,139,339 B | HDF5 变量表（B §1.2），h5 键（B §1.4） |
| NYUv2 表面法向 | 多个来源（B §1.6） | 多数未声明 | Eigen&Fergus normals_gt.tgz（只有 Wayback）；GeoNet 测试 654（HF 镜像）与训练 30,907；Bae 2021 795/654 | 496,214,722 B（Eigen&Fergus，Wayback HEAD）；GeoNet-test 镜像 4,621,182,485 B；Bae 的 zip 在 Drive 显示 "962M" | 由项目定 | B §1.6 |
| BLINK Relative_Depth（T 相对深度） | HF `BLINK-Benchmark/BLINK` | Apache-2.0；图像来自 Depth in the Wild，DIW 的许可未核实 | val 124（答案公开）/ test 124（答案一律为 "hidden"） | 9,203,975 B | 全部 | B §2 |
| CV-Bench（空间问答，留出任务） | HF `nyu-visionx/CV-Bench` | apache-2.0；其中 Omni3D 标注为 CC BY-NC 4.0 | **只有 test**：2,638 行（2D 1,438 / 3D 1,200） | parquet 404,808,364 B | 全部 | C §2.4 |
| EmbSpatial-Bench（空间问答，留出任务） | GitHub mengfeidu/EmbSpatial-Bench，数据在 HF `Phineas476/EmbSpatial-Bench` | 数据仓库为 cc-by-4.0；GitHub 没有 LICENSE；FlagEval 转换版没有 license | **只有 test**：3,640 行；另有单独的 EmbSpatial-SFT | bench json 253,512,493 B；FlagEval parquet 408,685,444 B | 全部 | C §3.4 |
| ADE20K（T 像素语义分割，可选；D 语义分割） | MIT SceneParse150 `ADEChallengeData2016.zip` | 图像："only for non-commercial research and educational purposes"；标注与代码为 BSD-3 | train 20,210 / val 2,000（都有标注）；release_test 3,352 张，只有图像 | 967,382,037 B（HEAD） | train + val 即整个 zip | 标签 0–150，单通道 uint8 PNG，0 表示 "other objects" 且不参与评测（C §1.4） |
| Hypersim（D 深度，可选） | apple-aiml-research/ml-hypersim（由 apple/ml-hypersim 迁移） | CC BY-SA 3.0；三角网格 GT 需要购买 Evermotion 资产 | 官方 v1：59,543 / 7,386 / 7,690 张，场景 365 / 46 / 46 | 457 个场景 zip 共 1,914,288,906,139 B（逐个 HEAD） | 外推：color+depth+normal+tonemap 约 305–432 GB；只取 tonemap+depth+normal 约 69–111 GB | B §3 |

## 2. 任务书要求特别核实的项目

1. **CV-Bench 的图像来源与所用划分**（C §2.3）：
   - ADE20K 部分 633 行、633 张，全部来自 **ADE20K val**；
   - COCO 部分 805 行、805 张，全部来自 **COCO val2017**；
   - Omni3D 部分 1,200 行，Hypersim、SUN RGB-D、nuScenes 各 400 行。其中 SUN RGB-D 的 149 个序列全部在 SUN RGB-D 官方 `alltest` 中；Hypersim 的 242 张全部属于 Hypersim 官方 test；nuScenes 所属划分**未能核实**。
   - **Omni3D 的 SUN RGB-D 部分包含 NYUv2 图像**：路径形如 `SUNRGBD/kv1/NYUdata/NYU####` 的共 89 行、36 张去重图像。把 NYU 编号当作 NYUv2 labeled 集的 1-based 索引与 `splits.mat` 对照，**36/36 张都在 NYUv2 官方 test（654）中**。旁证：SUN RGB-D 的 `alltrain`/`alltest` 里的 NYU 编号集合，分别与 `trainNdxs`/`testNdxs` 完全相同。像素级的同一性未核实。
2. **NYUv2**（B §1）：
   - 官方 795/654：本地 `splits.mat` 与官方 URL 的文件 sha256 相同。`trainNdxs`(795,1)、`testNdxs`(654,1)，uint16，1-based，两者不相交，并集为 {1..1449}。场景层面，train 249 个、test 215 个，没有共同的 scene。
   - 约 2.4 万帧训练集的可得来源：见总表与 B §1.4。DenseDepth 的 S3 返回 403；FastDepth 原站超时，这两个的体积未能核实。
   - 法向 GT 的来源：见 B §1.6。
3. **BLINK 相对深度**（B §2）：val 的答案公开（(A) 64，(B) 60）；test 的 `answer` 在 124/124 行都是 "hidden"。官方唯一的 test 评测渠道是 EvalAI challenge 2287，已于 2026-01-31 结束（is_active=False）。
4. **CountBench**（A §3）：官方只提供 URL，不打包图像。按文件顺序取前 50 条做 HEAD：非 200 占 8/50（16%），200 且 Content-Type 为图像的占 41/50。补测全部 540 条：非 200 占 123/540（22.8%）。
5. **只有测试划分的 benchmark**（C §5）：
   - **CV-Bench**：只有 test。
   - **EmbSpatial-Bench**：benchmark 只有 test；另有单独发布的训练集 EmbSpatial-SFT。
   - **BLINK**：14 个子任务都只有 val 和 test，没有 train；test 的答案隐藏。
   - **CountBench**：不分划分，只有一个文件（HF 镜像把它命名为 `train` 或 `test`）。
   - 有 train 的：RefCOCO 系列、ADE20K、NYUv2、ODinW13。
6. **Hypersim（可选）**（B §3）：许可、总体积、按场景的下载粒度与子集估算，见总表。

## 3. 与去重和泄漏相关的交叉事实（只陈述）

- **COCO val2017 与 RefCOCO 系列**：图像交集为 0，ann_id 交集也为 0（A §2）。
- **CV-Bench 与其他数据集的重合**：
  - 含 ADE20K val 的 633 张、COCO val2017 的 805 张、NYUv2 官方 test 的 36 张（C §2.3）。
  - 本次 §5 探针用过的两张 val2017 图（000000039769、000000000139）不在 CV-Bench 的 COCO 行中（服务器 CC 对 `raw/p6/C/cvbench_rows_meta.jsonl` 做了 grep）。
- **RefCOCO 系列内部的交叉**（A §1.4）：
  - 跨数据集时，一个数据集的 train 图像会出现在另一个数据集的评测划分里：RefCOCOg(umd) train ∩ RefCOCO(unc) val+testA+testB = 1,860 张；RefCOCO(unc) train ∩ RefCOCOg(umd) val+test = 1,891 张。
  - google 划分内部，train/val/test 之间有图像重叠；unc 与 umd 划分没有。
- **BTS 训练清单的场景名**：有 1,383 帧来自 raw 目录 `bookstore_0001d…j`。去掉字母后缀后，`bookstore_0001` 是 NYUv2 官方 test 划分里的 scene（B 组"值得注意的事实"第 4 条）。
- **EmbSpatial-Bench 的场景来源**：ScanNet 的 175 个 scan 全部属于 ScanNet 官方 val；MP3D 的 26 个场景中 17 个属于官方 test，9 个属于官方 val（C §3.3）。
- **同一份 NYUv2 654 张测试集的编号基**：`splits.mat` 与 GeoNet-test 镜像为 1-based；BTS 与 Marigold 的文件名为 0-based（B 组第 7 条）。
- **Eigen crop 的两种写法**：官方 toolbox 是 MATLAB `mask(45:471, 41:601)`，即 1-based 闭区间 427×561；BTS 是 Python `eval_mask[45:471, 41:601]`，即 0-based 半开区间 426×560（B §1.5）。

---

## 4. A 组：RefCOCO 系列、COCO val2017、CountBench（下文 "A §x" 指本节的 x 小节）

- 采集时间：2026-09-30 05:49–06:30 UTC，机器 b200。只做 HEAD、HF API 元数据、datasets-server JSON、zip 的 Range 读取。
- 证据目录：`experiments/P0/0001_env_probe/raw/p6/A/`（下文写"证据:"时用相对这个目录的路径）。脚本在 `experiments/P0/0001_env_probe/probe_scripts/p6_A_*.py`。
- 下载记账见 `downloads.tsv`。计量到的网络字节共 58,272,991，另有约 3 MB 元数据 JSON 未计量。落盘只有 168,736 字节，已全部删除。
- 本文只陈述事实，不做推荐。

---

### 1. RefCOCO / RefCOCO+ / RefCOCOg

#### 1.1 官方来源与许可

| 项 | 事实 | 证据 |
|---|---|---|
| 官方仓库 | https://github.com/lichengunc/refer 。GitHub API 的 license 字段为 `Apache-2.0`，这是仓库级许可，仓库内容为 API 代码 | `license_quotes_refer_refexp.txt` |
| 官方下载地址（README 原文） | `https://bvisionweb1.cs.unc.edu/licheng/referit/data/{refclef,refcoco,refcoco+,refcocog}.zip` | 同上 |
| README 对服务器的说明（原文） | "As the webserver is broken (sry about this), please check this [Issue](https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183) for all datasets downloading." | 同上 |
| 官方主机现状 | `bvisionweb1.cs.unc.edu` 在 2026-09-30 **DNS 无法解析**（curl: Could not resolve host） | `head_key_urls.txt` |
| Issue #14 给出的替代地址 | Wayback 快照：`https://web.archive.org/web/20220413011718/…/refcoco.zip`、`…20220413011656/…/refcoco+.zip`、`…20220413012904/…/refcocog.zip`（评论者 jamestiotio，2022-09-26） | `license_quotes_refer_refexp.txt` |
| Wayback 快照大小（HEAD，`if_` 原始字节形式） | refcoco.zip 53,993,993 B（原站 Last-Modified 2016-03-24）；refcoco+.zip 45,613,210 B（2016-03-24）；refcocog.zip 56,712,951 B（2016-11-20）；三个都返回 `accept-ranges: bytes` | `head_key_urls.txt`、`zip_list_*.txt` |
| RefCOCO / RefCOCO+ 标注的数据许可 | refer 的 README 没有数据许可声明；三个 zip 里只有 `instances.json` 和 `refs(*).p`，没有 license 文件。**数据许可未能核实（未找到明示声明）** | `zip_list_*.txt` |
| RefCOCOg（Google split）原始发布 | https://github.com/mjhucla/Google_Refexp_toolbox 。README 原文："This data is released by Google under the following license: … Creative Commons Attribution 4.0 International License"。其数据链接 `https://storage.googleapis.com/refexp/google_refexp_dataset_release.zip` 在 2026-09-30 HEAD 返回 **403** | `license_quotes_refer_refexp.txt`、`head_key_urls.txt` |
| 图像 | 全部来自 COCO train2014：五个 pickle 里所有 `file_name` 的前缀都是 `COCO_train2014`，许可见第 2 节的 COCO 条款 | `refs_stats.json` → `file_name_prefix_counts` |

zip 成员（通过 `remote_zip.py list` 读中央目录）：

| zip | 成员 | 解压后 B | 压缩后 B |
|---|---|---:|---:|
| refcoco.zip | refcoco/instances.json | 120,213,916 | 37,406,307 |
| | refcoco/refs(google).p | 31,776,065 | 8,577,416 |
| | refcoco/refs(unc).p | 31,742,595 | 8,009,738 |
| refcoco+.zip | refcoco+/instances.json | 120,197,185 | 37,401,196 |
| | refcoco+/refs(unc).p | 31,893,061 | 8,211,650 |
| refcocog.zip | refcocog/instances.json | 124,416,571 | 38,487,306 |
| | refcocog/refs(google).p | 33,853,786 | 9,106,927 |
| | refcocog/refs(umd).p | 33,853,676 | 9,118,034 |

#### 1.2 refs pickle 的真实 schema

读法：5 个 `refs(*).p` 都用 Range 读进内存，未落盘，再用拒绝一切 global 的受限 Unpickler 解析，默认编码即可解析。证据：`refs_stats.json`。

- 顶层结构：`list[dict]`，一条对应一个 ref。所有 ref 的键集合都相同。
- ref 的字段：`sent_ids: list[int]`、`file_name: str`、`ann_id: int`、`ref_id: int`、`image_id: int`、`split: str`、`sentences: list[dict]`、`category_id: int`。
- sentence 的字段：`tokens: list[str]`、`raw: str`、`sent_id: int`、`sent: str`。
- 示例（refcoco(unc) 第 0 条）：`{"ref_id":0,"ann_id":1719310,"image_id":581857,"split":"train","category_id":1,"file_name":"COCO_train2014_000000581857_16.jpg","sent_ids":[0,1,2],"sentences":[{"raw":"THE LADY WITH THE BLUE SHIRT","sent":"the lady with the blue shirt","sent_id":0,"tokens":[…]},…]}`。`file_name` 末尾的 `_16` 不是真实文件名，README 注释写的是 "ignore filename"。
- sha256：refcoco(unc) `d4a8dd3152f13012…`，refcoco(google) `2bb306561f788f0e…`，refcoco+(unc) `7de24c182449758b…`，refcocog(umd) `0331c7533537b67c…`，refcocog(google) `e4d8320dfd15fc21…`。完整值见 `refs_stats.json` → `member_sha256`。

#### 1.3 各 split 计数（官方 pickle，逐条统计）

证据：`refs_stats.json` → `per_split`、`image_overlap_between_splits`。

| 数据集(splitBy) | split | refs | sentences | images | ann_ids | 类别数 |
|---|---|---:|---:|---:|---:|---:|
| RefCOCO (unc) | train | 42,404 | 120,624 | 16,994 | 42,404 | 78 |
| | val | 3,811 | 10,834 | 1,500 | 3,811 | 66 |
| | testA | 1,975 | 5,657 | 750 | 1,975 | 31 |
| | testB | 1,810 | 5,095 | 750 | 1,810 | 64 |
| | **合计** | **50,000** | **142,210** | **19,994** | **50,000** | 78 |
| RefCOCO (google) | train | 40,000 | 113,762 | 19,213 | 40,000 | 78 |
| | val | 5,000 | 14,246 | 4,559 | 5,000 | 72 |
| | test | 5,000 | 14,202 | 4,527 | 5,000 | 70 |
| | **合计** | 50,000 | 142,210 | 19,994 | 50,000 | 78 |
| RefCOCO+ (unc) | train | 42,278 | 120,191 | 16,992 | 42,278 | 78 |
| | val | 3,805 | 10,758 | 1,500 | 3,805 | 66 |
| | testA | 1,975 | 5,726 | 750 | 1,975 | 31 |
| | testB | 1,798 | 4,889 | 750 | 1,798 | 64 |
| | **合计** | **49,856** | **141,564** | **19,992** | **49,856** | 78 |
| RefCOCOg (umd) | train | 42,226 | 80,512 | 21,899 | 42,224 | 78 |
| | val | 2,573 | 4,896 | 1,300 | 2,573 | 72 |
| | test | 5,023 | 9,602 | 2,600 | 5,023 | 76 |
| | **合计** | **49,822** | **95,010** | **25,799** | **49,820** | 78 |
| RefCOCOg (google) | train | 44,822 | 85,474 | 24,698 | 44,820 | 78 |
| | val | 5,000 | 9,536 | 4,650 | 5,000 | 76 |
| | **合计** | 49,822 | 95,010 | 25,799 | 49,820 | 78 |

split 之间的图像重叠：

- **unc 和 umd：** split 之间图像互不重叠，所有两两交集都是 0。
- **google：** split 之间有重叠。RefCOCO(google) 的 train∩val 为 4,042 张，train∩test 为 3,989 张，val∩test 为 852 张。RefCOCOg(google) 的 train∩val 为 3,549 张。

#### 1.4 splitBy 之间、数据集之间的关系

证据：`refs_stats.json` → `_cross`、`refs_crosstab_images.json`。

- **RefCOCO unc 与 google：** 50,000 个 ref_id 完全相同，每个 ref_id 对应的 ann_id、image_id、sent_ids 也完全一致，两者只是划分不同。按 ref 计的交叉表（unc|google）为：train|train 33,962、train|val 4,223、train|test 4,219、val|train 3,022、testA|train 1,580、testB|train 1,436，其余格见证据。
- **RefCOCOg umd 与 google：** ann_id 集合相同（49,820 个），图像集合相同（25,799 张）。但**同一 ref_id 在两个 pickle 里指向不同对象**，只有 27 个 ref_id 的 ann_id、image_id、sent_ids 一致。图像级交叉表（umd|google）为：train|train 20,950、train|val 3,975、val|train 1,251、val|val 228、test|train 2,497、test|val 447。
- **RefCOCO(unc) 与 RefCOCO+(unc)：** val、testA、testB 的图像集合逐 split 完全相同。train 相差 2 张，即 image 115260 和 174967，只出现在 RefCOCO。RefCOCO+ 的 ann_id 是 RefCOCO 的子集，少 144 个。
- **图像并集：** RefCOCO 19,994 张，RefCOCO+ 19,992 张，RefCOCOg 25,799 张。**三者并集 30,956 张。** 交集为 RefCOCO∩RefCOCO+ 19,992、RefCOCO∩RefCOCOg 14,837、RefCOCO+∩RefCOCOg 14,836、三者共有 14,836。image_id 范围是 9 到 581,857。
- **ann_id 并集：** 71,991 个。RefCOCO∩RefCOCOg 为 27,829 个。
- **一个数据集的 train 图像与另一个数据集评测 split 图像的交集：**
  - RefCOCOg(umd) train ∩ RefCOCO(unc) val+testA+testB = 1,860 张，与 RefCOCO+(unc) 评测 split 的交集同为 1,860 张。
  - RefCOCO(unc) train ∩ RefCOCOg(umd) val+test = 1,891 张。
  - RefCOCO(unc) train ∩ RefCOCO+(unc) val+testA+testB = 0。
  - 全表见 `refs_stats.json` → `_cross.train_image_vs_other_dataset_eval_image_overlap`。

#### 1.5 图像：COCO train2014 单图 URL 与字节数

| 项 | 事实 | 证据 |
|---|---|---|
| 单图 URL 模板 | `http://images.cocodataset.org/train2014/COCO_train2014_{id:012d}.jpg` | — |
| 单图 HEAD 示例 | id=9：HTTP 200，`Accept-Ranges: bytes`，`Content-Type: image/jpeg`，`Content-Length: 224297` | `head_key_urls.txt` |
| 对并集 30,956 张全部 HEAD（并发 16，用时 499 s） | 30,956 张全部返回 200，失败 **0**；全部为 `image/jpeg`，全部带 `Accept-Ranges: bytes`，没有重定向 | `refcoco_images/head_train2014_union_summary.json`、`…_compact.tsv` |
| 并集总字节（按 HEAD 的 Content-Length） | **5,252,007,403 B**（约 5.25 GB）。均值 169,660 B，中位数 163,502.5 B，最小 9,694 B，最大 744,541 B | 同上 |
| 按数据集的字节 | RefCOCO 19,994 张，3,460,649,556 B；RefCOCO+ 19,992 张，3,460,190,802 B；RefCOCOg 25,799 张，4,309,867,610 B | `refcoco_images/refs_image_bytes.json` |
| 只取评测 split 的并集 | RefCOCO(unc) val/testA/testB、RefCOCO+(unc) val/testA/testB、RefCOCOg(umd) val/test，合计 **6,549 张，1,105,415,725 B** | 同上 |
| 只取 train split 的并集 | unc/unc/umd 三个 train 合计 28,158 张，4,780,845,148 B；去掉评测图像后剩 24,407 张，4,146,591,678 B | 同上 |
| 按 split 的字节 | 例：RefCOCO(unc) val 258,337,876 B、testA 126,617,267 B、testB 131,247,412 B；RefCOCOg(umd) val 215,762,148 B、test 433,283,702 B；全表见证据 | 同上 |
| train2014.zip | HEAD 大小 13,510,573,713 B（Last-Modified 2018-07-11），带 `Accept-Ranges: bytes`。zip64 EOCD 记录的条目数为 **82,784**，是否含 1 个目录条目未核实 | `head_key_urls.txt`、`zip_eocd_coco.jsonl` |
| 并集占 train2014 的比例 | 按张数为 30,956/82,784 = 37.4%（分母是 zip 条目数）；按字节为 5,252,007,403/13,510,573,713 = 38.9% | 由上两行计算 |

#### 1.6 mask 的来源（ann_id → COCO instances）

| 项 | 事实 | 证据 |
|---|---|---|
| 关联方式 | 每个 ref 带一个 `ann_id`，指向 COCO instances 的 annotation id | §1.2 |
| 来源 A：refer zip 自带的 `instances.json` | 大小见 §1.1 的表（约 120–124 MB，超过 50 MB，没有提取）。只流式读了开头 1,500 B（网络取回 131,431 B）：第一个键是 `"info": {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.", "url": "http://mscoco.org", "version": "1.0", "year": 2014, …}`，随后是 `"images"`。image 记录里的 `coco_url` 是旧式的 `http://mscoco.org/images/<id>`。其余顶层键未能核实（需要解压全文件） | `refcoco_instances_json_head.txt` |
| 来源 B：COCO `annotations_trainval2014.zip` | HEAD 252,872,794 B。成员（解压后/压缩后）：instances_train2014.json **332,556,225 / 107,297,879**；instances_val2014.json 160,682,675 / 51,955,093；person_keypoints_train2014.json 170,733,465 / 50,041,387；person_keypoints_val2014.json 81,637,509 / 23,951,481；captions_train2014.json 66,782,097 / 13,124,596；captions_val2014.json 32,421,077 / 6,501,124。instances_train2014.json 超过 50 MB，没有提取 | `zip_list_annotations_trainval2014.txt` |
| 格式说明 | COCO 官方格式文档：https://cocodataset.org/#format-data ，源文件为 `cocodataset.github.io/dataset/format-data.htm` | — |
| iscrowd / polygon 的直接证据 1 | HF 镜像 jxu124/refcoco 的 testB 全部 1,810 行，逐行解析 `raw_anns`：iscrowd=0 共 1,810 行，全部是 polygon（`list[list[float]]`）。polygon 段数分布：1 段 1,683 行，2 段 85 行，3 段 18 行，4 段 11 行，≥5 段 13 行。`raw_anns.id` 与 `ann_id` 1,810/1,810 一致；该 split 的图像集合与官方 refcoco(unc) testB 完全相同（750 张） | `hf/jxu124_refcoco_testB_rows_check.json` |
| iscrowd / polygon 的直接证据 2 | lmms-lab-encoder/RefCOCOg 的 val（7,573 行）和 test（5,023 行）经 datasets-server `/statistics` 统计，`iscrowd` 的 min 和 max 都是 0。RefCOCO 和 RefCOCO+ 这两个仓库的 statistics 接口报错 | `hf/lmms_refcoco_iscrowd_stats.txt` |
| iscrowd 的间接证据 | 本地 instances_val2017.json 里，446 个 crowd 标注的 id 都在 [900100002299, 908800474293] 内，36,335 个非 crowd 标注的 id 都在 [283, 2232119] 内。RefCOCO 系列 71,991 个 ann_id 的最大值是 2,232,112，**没有一个 ≥ 900100002299**。这里依赖一个假设：2014 与 2017 共用同一套标注 id | `refs_crowd_check_via_val2017.json` |
| 全量直接核实 | **未能核实**。train split 的 iscrowd/polygon 需要 instances_train2014.json（332 MB）或 refer 的 instances.json（120 MB），两者都超过 50 MB 单文件上限。另外 datasets-server 的 `/filter` 端点当天多次 90–180 s 超时，没有采用 | — |

#### 1.7 HF 镜像（元数据来自 HF API 和 datasets-server，均未下载数据）

证据：`hf/hf_<repo>.json`。里面有 files 及大小、README、`/splits`、`/size`、`/first-rows` 的 features 和前 2 行。

| 请求的 id | 实际 id（重定向后） | sha | 最后修改 | card license | 仓库总字节（HF API） | datasets-server 行数 |
|---|---|---|---|---|---:|---|
| lmms-lab/RefCOCO | lmms-lab-encoder/RefCOCO | a5dff0b319 | 2024-03-08 | 无 | 2,278,341,779 | val=8811; test=5000; testA=1975; testB=1810 |
| lmms-lab/RefCOCOplus | lmms-lab-encoder/RefCOCOplus | a283082593 | 2024-03-08 | 无 | 526,627,540 | val=3805; testA=1975; testB=1798 |
| lmms-lab/RefCOCOg | lmms-lab-encoder/RefCOCOg | 93af555459 | 2024-03-08 | 无 | 2,053,390,258 | val=7573; test=5023 |
| jxu124/refcoco | 同 | 9fba8200c5 | 2023-05-20 | 无 | 38,898,660 | train=42404; validation=3811; test=1975; testB=1810 |
| jxu124/refcocoplus | 同 | 12e20b0b60 | 2023-05-20 | 无 | 39,776,336 | train=42278; validation=3805; test=1975; testB=1798 |
| jxu124/refcocog | 同 | 55319436ad | 2023-05-20 | 无 | 45,972,327 | train=42226; validation=2573; test=5023 |
| jxu124/refcoco-benchmark | 同 | 2f2f892835 | 2023-10-30 | 无 | 4,072,694,910 | 11 个 split，一行对应一张图，例如 refcoco_unc_val=1500、refcocog_umd_test=2600 |
| Kangheng/refcoco | 同 | c23f7adcde | 2024-09-18 | 无 | 9,818,016,501 | val=10834; testA=5657; testB=5095 |
| Kangheng/refcocop | 同 | 82c6a1a143 | 2024-09-18 | 无 | 549,396,824 | val=10758; testA=5726; testB=4889 |
| Kangheng/refcocog | 同 | 7bbe9ea47f | 2024-09-18 | 无 | 1,259,399,510 | test=9602; val=4896 |
| PaDT-MLLM/RefCOCO | 同 | 63c2c22808 | 2025-10-10 | apache-2.0 | 579,970,949 | train=321327; validation=26488; test=9602 |

所有 RefCOCO 镜像的 dataset card 都没有 license 字段，唯一例外是 PaDT-MLLM/RefCOCO（apache-2.0）。lmms-lab 三个仓库的 README 写的是 "This is a formatted version of [RefCOCO](https://github.com/lichengunc/refer)"。jxu124 仓库的 README 只有 "More Information needed"。

真实 schema（datasets-server `/first-rows` 的 features）：

- **lmms-lab-encoder/RefCOCO、RefCOCOplus、RefCOCOg**（三者相同）：`question_id:string; image:Image; question:string; answer:List[string]; segmentation:List[float32]; bbox:List[float32]; iscrowd:int8; file_name:string`。
  - 图像打包在 parquet 里。一行对应一个 ref，`question_id` 就是 ann_id（例：1537681）。`answer` 是该 ref 的全部 sentences。`segmentation` 是**展平的单个 polygon**。
- **jxu124/refcoco、refcocoplus**：`sent_ids:Sequence[int64]; file_name:string; ann_id:int64; ref_id:int64; image_id:int64; split:string; sentences:list[{raw:string, sent:string, sent_id:int64, tokens:Sequence[string]}]; category_id:int64; raw_anns:string; raw_image_info:string; raw_sentences:string; image_path:string; bbox:Sequence[float64]; captions:Sequence[string]; global_image_id:string; anns_id:string`。
  - **没有图像**。`image_path` 形如 `coco/train2014/COCO_train2014_000000581563.jpg`。`raw_anns` 是 COCO annotation 的 JSON 字符串，键为 area、bbox、category_id、id、image_id、iscrowd、segmentation。
- **jxu124/refcocog**：字段与上面相同，只是顺序不同。
- **jxu124/refcoco-benchmark**：`ref_list:list[{ann_info:{area, bbox, category_id, id, image_id, iscrowd, segmentation:Sequence[Sequence[float64]]}, ref_info:{ann_id, category_id, file_name, image_id, ref_id, sent_ids, sentences, split}}]; image_info:{coco_url, date_captured, file_name, flickr_url, height, id, license, width}; image:Image`。一行对应一张图，图像打包在 parquet 里。
- **Kangheng/refcoco、refcocop、refcocog**：`question_id:int64; question:string; image:Image; bbox:string; image_size:Sequence[int64]`。
  - 一行对应一个 sentence。例：`bbox` 为 `"[468, 0, 639, 116]"`，是整数像素 xyxy 字符串。
  - Kangheng/refcoco 仓库里另有 125 个 train parquet，共 9,269,715,447 B，但 card 只声明了 val/testA/testB，datasets-server 不索引 train。
- **PaDT-MLLM/RefCOCO**：仓库里放的是 json 文件，如 refcoco_train.json（180,701,848 B）。datasets-server 推断出的 features 为 `id:int64; image:string; conversations:List[{from,value}]; task:string; answer_template:string; objects:List[{patches:List[int64], bbox:List[float64], iscrowd:int64, area:float64, rle:{size, counts:string}, label:string}]`，mask 以压缩 RLE 存储。

数值对照（只做算术比对，未按 id 核实）：

- Kangheng 各 split 行数等于官方对应 split 的 sentences 数。例：refcoco val 10,834，refcocog test 9,602。
- lmms-lab RefCOCO 的 val 8,811 = unc val refs 3,811 + google val refs 5,000；test 5,000 = google test refs。lmms-lab RefCOCOg 的 val 7,573 = umd val 2,573 + google val 5,000；test 5,023 = umd test。
- PaDT 的 train 321,327 = 三个数据集 train 的 sentences 之和（120,624 + 120,191 + 80,512）。validation 26,488 同理（10,834 + 10,758 + 4,896）。
- jxu124/refcoco-benchmark 各 split 行数等于对应 split 的图像数。

jxu124 的 split 命名（`/statistics` 统计 `split` 列）：

- `validation` 列的值为 "val"。
- `test` 列的值为 **"testA"**（refcoco 和 refcocoplus 都是 1,975 行）。
- `testB` 列的值为 "testB"。
- refcocog 的 `test` 列值就是 "test"。

证据：`hf/jxu124_split_field_stats.txt`。另外 jxu124/refcoco testB 的逐行核对显示，图像集合与官方一致，ann_id 都属于官方集合，sentences 为 5,095 条，与官方相同。

---

### 2. COCO val2017

| 项 | 事实 | 证据 |
|---|---|---|
| 官方下载页 | https://cocodataset.org/#download ，源文件 `cocodataset.github.io/dataset/download.htm`。列出 `http://images.cocodataset.org/zips/val2017.zip` 和 `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` 等 | `coco_site_download.txt` |
| val2017.zip | HEAD 815,585,330 B（Last-Modified 2018-07-11），带 `Accept-Ranges: bytes`；EOCD 记录 5,001 个条目 | `head_key_urls.txt`、`zip_eocd_coco.jsonl` |
| annotations_trainval2017.zip | HEAD 252,907,541 B。instances_val2017.json 解压后 19,987,840 B、压缩后 6,529,031 B；instances_train2017.json 469,785,474 B；captions_val2017.json 3,872,473 B；person_keypoints_val2017.json 10,020,657 B | `zip_list_annotations_trainval2017.txt` |
| 许可：Terms of Use 原文 | "The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License." / "The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use. The users of the images accept full responsibility for the use of the dataset…" 链接：https://cocodataset.org/#termsofuse （源文件 https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/termsofuse.htm） | `coco_site_termsofuse.txt` |
| 逐图 Flickr 许可（instances_val2017.json 的 `images[].license`） | 1 BY-NC-SA 2.0：1,431；2 BY-NC 2.0：630；3 BY-NC-ND 2.0：1,414；4 BY 2.0：857；5 BY-SA 2.0：417；6 BY-ND 2.0：246；7 "No known copyright restrictions"：5；8 "United States Government Work"：0 | `coco_val2017/coco_val2017_schema.json` |
| 计数 | 图像 5,000 张，annotations 36,781 条，类别 80 个（id 范围 1–90，不连续），supercategory 12 个。有标注的图像 4,952 张，无标注的 48 张。每图最多 63 条标注 | 同上 |
| segmentation 类型 | iscrowd=0 的 36,335 条全部是 polygon；iscrowd=1 的 446 条全部是 RLE，且为未压缩 RLE（counts 是 list） | 同上 |
| schema | 顶层键为 `info, licenses, images, annotations, categories`。<br>image 字段：`license:int, file_name:str, coco_url:str, height:int, width:int, date_captured:str, flickr_url:str, id:int`。<br>annotation 字段：`segmentation:list[list[float]] 或 {counts,size}, area:float, iscrowd:int, image_id:int, bbox:list[float](xywh), category_id:int, id:int`。<br>category 字段：`supercategory:str, id:int, name:str`。<br>info 为 `{"description":"COCO 2017 Dataset","version":"1.0","year":2017,"contributor":"COCO Consortium","date_created":"2017/09/01"}`。<br>文件 sha256 `e8c7f7908f1d7278…` | 同上 |
| 格式文档 | https://cocodataset.org/#format-data | — |
| 逐图 URL：coco_url | 5,000 张全部 HEAD（并发 16，用时 79 s）：全部 200，**失败 0**，全部为 `image/jpeg`，全部带 `Accept-Ranges: bytes`。总字节 **814,705,164**，均值 162,941.0，中位数 154,637.5，最小 8,886，最大 696,529 | `coco_val2017/head_val2017_all5000_summary.json` |
| 逐图 URL：flickr_url | 用 `random.Random(0)` 抽 200 条：162 条返回 200（全部经过重定向，没有 Content-Length），404 共 30 条，410 共 8 条，**失败率 19%** | `coco_val2017/head_val2017_flickr_seed0_200_summary.json` |
| 500 张子集的大小 | 均值 × 500 = **81,470,516 B（约 81.5 MB）**。几个具体子集的实测：seed 0/1/2 随机 500 张分别为 82,238,798 / 81,408,959 / 83,877,239 B；按 id 排序取前 500 张为 80,882,315 B；在有标注的 4,952 张里用 seed 0 抽 500 张为 82,206,681 B | `coco_val2017/val2017_500_subset_size.json` |
| 与 RefCOCO 系列的关系 | val2017 的 5,000 张图与 RefCOCO 系列 30,956 张图交集为 **0**；ann_id 交集也为 0 | `refs_crowd_check_via_val2017.json` |

HF 镜像（均为元数据）：

| repo | card license | 总字节 | 行数 | 内容/schema |
|---|---|---:|---|---|
| detection-datasets/coco | 无 | 19,966,849,859 | train=117,266; val=**4,952** | `image_id, image:Image, width, height, objects:{bbox_id, category:ClassLabel(80), bbox, area}`，**没有 segmentation**。val 的两个 parquet 共 807,426,880 B |
| rafaelpadilla/coco2017 | 无 | 30,594（仓库里只有加载脚本） | train=118,287; val=5,000（datasets-server 转换） | `image:Image, image_id, objects:{id, area, bbox, label:ClassLabel(92), iscrowd:bool}`，没有 segmentation |
| phiyodr/coco2017 | 无 | 30,175,139 | train=118,287; validation=5,000 | 只有 captions 元数据（`coco_url, flickr_url, captions…`），没有图像 |
| pcuenq/coco-2017-mirror | other | 20,405,357,537 | — | 原样镜像官方 zip：val2017.zip 815,585,330 B（与官方 HEAD 相同），annotations_trainval2017.zip 252,907,541 B，train2017.zip 19,336,861,798 B |
| LibreYOLO/coco-val2017-mini500 | cc-by-4.0 | 83,040,696 | validation=500 | imagefolder 格式，另有 `annotations/instances_val2017_mini500.json`（2,135,816 B）。所选 500 张是哪些图未核实 |

---

### 3. CountBench

| 项 | 事实 | 证据 |
|---|---|---|
| 论文 | Paiss et al., "Teaching CLIP to Count to Ten", arXiv:2302.12066 | `countbench_project_page.txt` |
| 官方发布 | 项目页仓库 https://github.com/teaching-clip-to-count/teaching-clip-to-count.github.io ，其中的 `CountBench.json`（146,701 B，sha256 `2b1586d1b945243429d54afd9f719c787ec56bdb0e66dfa8a1f151ad58c164b7`）。项目页原文："CountBench contains a total of 540 images containing between two and ten instances of a particular object, where their corresponding captions reflect this number. The Images urls and their corresponding captions can be found here." | `downloads.tsv`、`countbench_project_page.txt` |
| 许可 | 官方仓库 GitHub license 为 **None**，项目页没有许可声明，因此**未声明许可**。页面说数据 "curated … from the publicly available LAION-400M image-text dataset"，LAION-400M 本身的许可本次未核实 | `license_quotes_refer_refexp.txt`、`countbench_project_page.txt` |
| 官方 schema | `list[dict]`，540 条，每条 `{image_url:str, text:str, number:int}`。number 取 2–10，每个值恰好 60 条。唯一 URL 538 个，有 2 个 URL 各出现 2 次（bing 缩略图、shutterstock 各一个） | `countbench/countbench_urls_all540.txt` |
| 图像形态（官方） | **只给 URL**，不打包图像 | 同上 |
| HEAD 抽样：CountBench.json 中的前 50 条 | 返回 200 的 42 条，其中 41 条 `image/*`，1 条 `text/html`：wirerealm 的图重定向到站点首页。返回 403 的 2 条，404 的 2 条。连接失败 4 条：证书过期 2 条，DNS 失败 2 条。**非 200 占 8/50 = 16%**；200 且为图像的占 41/50 = 82% | `countbench/head_first50_summary.json`、`.tsv` |
| 补充：540 条全部 HEAD | 返回 200 的 417 条，其中 407 条为图像类型。非 200 的 123 条（22.8%）：404 共 48，403 共 21，证书过期 16，DNS 失败 22，其余 400/405/410/429/503 等。200 且为图像的占 407/540 = 75.4% | `countbench/head_all540_summary.json` |

HF 镜像：

| repo | card license | 字节 | 行数 | schema / 图像 |
|---|---|---:|---|---|
| nielsr/countbench | 无（README 只写 "introduced in the paper … Teaching CLIP to Count to Ten"） | 23,353,242（1 个 parquet，23,350,530） | train=540 | `image_url:string; text:string; number:int64; image:Image`。**图像打包在 parquet 里**：491 行有图，**49 行 image 为 null**。540 行的 (image_url, text, number) 与官方 json 逐条同序一致 |
| vikhyatk/CountBenchQA | 无 | 23,390,324 | test=491 | `image:Image; text:string; question:string; number:int32`。README 原文："This version only includes 491 images from the original CountBench dataset, since some of the original URLs can no longer be accessed." |
| khalidalt/countbench | 无 | 23,403,324 | train=540 | `image_url, text, number, image:{bytes,path}, ar_text`，是阿拉伯语翻译版 |
| moondream/CountBenchQA-VLMEvalKit | apache-2.0 | 29,531,204（1 个 TSV） | train=488 | `image:string`（base64）、`question, answer:int64, index` |

nielsr 的证据：`countbench/nielsr_countbench_rows_check.json`，逐行来自 `/rows`，图像以签名 URL 形式返回，未下载。

---

### 未能核实（汇总）

- **RefCOCO / RefCOCO+ 标注的数据许可：** refer 的 README 和 zip 里都没有声明。仓库级 Apache-2.0 是否覆盖数据，未核实。
- **RefCOCO 系列 train split 的 iscrowd/polygon：** 需要超过 50 MB 的 instances 文件，只有间接证据（id 区间）和 testB 全量、RefCOCOg val/test 的直接证据。
- **refer `instances.json` 的完整顶层键：** 只读了开头 1.5 KB。
- **train2014.zip 的 82,784 个条目是否含目录条目；COCO 官方文档所称的 82,783 张：** 本次没有直接核实。
- **lmms-lab RefCOCO val=8,811 的组成：** 只做了算术吻合，没有按 id 核实。
- **LibreYOLO mini500 选取的具体是哪些图。**
- **LAION-400M 的许可。**
- **datasets-server `/filter` 查询：** 当天超时，未得到结果。

### 值得注意的事实（只陈述不解读）

1. refer 官方数据主机 `bvisionweb1.cs.unc.edu` 在 2026-09-30 DNS 无法解析，README 自己写着 "the webserver is broken"。能取到的是 2022-04-13 的 Wayback 快照（带 Range 支持）。Google Refexp 的原始 zip 返回 403。
2. RefCOCO 的 google 与 unc 两种划分是**同一批 50,000 个 ref**。google 划分的 split 之间有图像重叠：train∩val 4,042 张，train∩test 3,989 张，val∩test 852 张；unc 和 umd 划分没有。
3. RefCOCOg 的 umd 与 google 两个 pickle **ref_id 编号不对应**：同一 ref_id 只有 27 个指向同一对象，但 ann_id 集合和图像集合完全相同。
4. 一个数据集的 train 图像会出现在另一个数据集的评测 split 里：RefCOCOg(umd) train ∩ RefCOCO(unc) val+testA+testB = 1,860 张；RefCOCO(unc) train ∩ RefCOCOg(umd) val+test = 1,891 张。
5. RefCOCO+ 的图像和 ann_id 都是 RefCOCO 的子集，分别少 2 张图和 144 个 ann。评测 split 的图像集合完全相同。
6. RefCOCOg(umd) train 有 42,226 个 refs，但只有 42,224 个不同的 ann_id。
7. 三个数据集的图像并集为 30,956 张、5,252,007,403 B，占 train2014.zip 字节数的 38.9%。30,956 个单图 URL 的 HEAD 全部成功。
8. COCO val2017 与 RefCOCO 系列图像交集为 0。
9. jxu124 镜像把 testA 叫作 `test`，列内 `split` 值为 "testA"。jxu124 的 `raw_image_info.coco_url` 和 refer 的 `instances.json` 都用旧式 `http://mscoco.org/images/<id>`，这个地址 HEAD 30 s 超时。
10. `lmms-lab/RefCOCO*` 现已重定向到 `lmms-lab-encoder/*`，都没有 train split。它的 `segmentation` 是展平的单个 polygon（`List[float32]`）。
11. COCO val2017 逐图许可中，NC 类（id 1/2/3）共 3,475/5,000 张（69.5%），ND 类（id 3/6）共 1,660 张，纯 BY（id 4）只有 857 张。
12. COCO val2017 的 `flickr_url` 在 seed 0 抽的 200 条样本里 19% 返回 404/410；`coco_url` 100% 可用。
13. detection-datasets/coco 的 val 只有 4,952 行，去掉了 48 张无标注图，也没有 segmentation。COCO 类别 id 为 1–90，不连续，共 80 类。
14. CountBench 官方只发布 URL，没有许可声明。540 个 URL 中 22.8% 在 HEAD 时不是 200，另有 10 个返回 200 但 Content-Type 不是 image/*（其中 5 个是 application/octet-stream，5 个是 text/html）。HF 上打包版只有 491 张（nielsr 版有 49 行 null）。
15. nielsr/countbench 第 0 行的图像已打包在 parquet 里，但它的原始 URL 现在重定向到一个 HTML 首页。


---

## 5. B 组：NYU Depth V2、BLINK Relative Depth、Hypersim（下文 "B §x" 指本节的 x 小节）

- 采集时间：2026-09-30（UTC），采集者：服务器 CC 子代理 p6_B。只陈述事实，不做推荐。
- 下文所说的"证据"均为 `experiments/P0/0001_env_probe/raw/p6/B/` 下的相对路径（`$PROJ=/home/ubuntu/workspace/wzg/gjl/vision-internalization`）。脚本放在 `experiments/P0/0001_env_probe/probe_scripts/p6_B_*.py`。
- 字节数写原值，括号里给出 GB（1e9）或 GiB（2^30），并注明单位。"HEAD"指 HTTP HEAD 的 Content-Length；"HF API"指 `HfApi().dataset_info(files_metadata=True)` 汇总的 siblings 大小。
- 没有做任何大体量下载。range read 只读进内存，不落盘；所有网络读取都记在 `downloads.tsv`。

---

### 1. NYU Depth V2

#### 1.1 官方来源与许可

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

#### 1.2 Labeled `.mat` 的变量（MATLAB v7.3 / HDF5；range read 元数据共 3.34 MB，只进内存）

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

#### 1.3 官方 795/654 划分（本地 `splits.mat`，只读，用 scipy.io.loadmat）

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

#### 1.4 常用的大训练集（Eigen / BTS "sync" / DenseDepth / FastDepth 及 HF 镜像）

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

#### 1.5 Eigen 654 测试集与 crop 的定义出处（只给出处，不引用论文数值）

- 测试集：Eigen, Puhrsch, Fergus, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NIPS 2014（arXiv:1406.2283；HEAD 6,579,016 B，**PDF 未下载，论文正文未核实**）。作者页的 Wayback 快照见 `wayback/eigen_depth.txt`。
- 实测：BTS 的 654 条测试清单，文件名编号 0..1448（0-based），加 1 后**与 `splits.mat` 的 testNdxs 完全相同**。BTS 生成测试 GT 的脚本 `utils/extract_official_train_test_set_from_mat.py` 读的是 `h5_file['rawDepths']`，不是补洞后的 `depths`；它按 `sceneTypes` 分目录，并把 RGB 的 `[7:474, 7:632]` 以外置黑。证据：`gh/cleinc_bts_master_utils_extract_official_train_test_set_from_mat.py`。
- crop 在代码里的定义：BTS `pytorch/bts_eval.py` 的参数 `--eigen_crop`（help 原文 "crops according to Eigen NIPS14"），NYU 分支写 `eval_mask[45:471, 41:601] = 1`，Python 0-based 半开区间。官方 toolbox 写 `mask(45:471, 41:601)`，MATLAB 1-based 闭区间，`sz=[427 561]`。证据：`gh/cleinc_bts_master_pytorch_bts_eval.py`、`nyu_toolbox_excerpts.txt`。
- Eigen & Fergus ICCV 2015 的页面（Wayback 2022-10-10）提供 NYUv2 测试集的 predictions_depth/normals `.mat` 文件。见 `wayback/eigen_dnl.txt`。

#### 1.6 NYUv2 表面法向 GT 来源

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

### 2. BLINK：Relative Depth 子任务

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

### 3. Hypersim（可选）

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

### 未能核实汇总

- DenseDepth `nyu_data.zip` 与 `nyu_test.zip` 的真实字节数：S3 返回 403，tinyurl 指向的 Drive 文件夹返回 404。
- FastDepth/sparse-to-dense 原始 tar.gz（datasets.lids.mit.edu）：连接超时。TFDS 页面的数字是间接来源。
- BTS sync.zip 的精确字节数：Drive 只显示 "5.9G"。aradhye 镜像是否与它逐字节相同：未验证。
- Eigen 作者页（~deigen/depth、~deigen/dnl）现在 403，只能看 Wayback 快照。Eigen NIPS14 论文正文（crop 的文字定义）没有读，因为 PDF 有 6.6 MB，为节省预算没有下载。
- Ladický 2014 的 `nyu_normals_gt.zip`：原站失效，Wayback 没有存档。
- Eigen `normals_gt.tgz` 的成员总数。GeoNet 30,907 张训练集在 SharePoint 上的大小（只在 Bae 2021 README 里看到这个张数）。
- DIW（BLINK 图像来源）的许可。FastDepth 47,584 帧所属 scene 与官方 test scene 是否重叠：需要逐个遍历 tar 头，没有做。

### 值得注意的事实（只陈述不解读）

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


---

## 6. C 组：ADE20K、CV-Bench、EmbSpatial-Bench、ODinW13、各基准的划分（下文 "C §x" 指本节的 Cx 小节）

范围：ADE20K、CV-Bench、EmbSpatial-Bench、ODinW13，以及"哪些基准只有 test/eval split"。
采集时间：2026-09-30 05:49–06:11 UTC，机器 b200。本文只记录事实，不给建议。

**证据目录：** `experiments/P0/0001_env_probe/raw/p6/C/`，下文简称 `C/`。表中每个数字后面标了来源文件或 URL。
**方法：**
- HF Hub API：`dataset_info(files_metadata=True)` 和 `list_repo_tree`。
- datasets-server JSON API：`/splits`、`/size`、`/info`、`/first-rows`、`/rows`、`/statistics`。
- HTTP HEAD。
- 远程 zip 中央目录：用 HTTP Range 读取（`probe_scripts/p6_C_zip.py`，由 `probe_scripts/remote_zip.py` 改写，增加了网络字节计数）。
- 少量小文件下载，逐条记在 `C/downloads.tsv`；用完已全部删除。

证据脚本共 11 个，均为 `experiments/P0/0001_env_probe/probe_scripts/p6_C_*.py`：hfsearch、hfmeta、html2txt、zip、cvbench_rows、cvbench_tab、sunsplit、dsrows、odinw_members、odinw_fetch、tables。

---

### C1. ADE20K

#### C1.1 候选来源

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

#### C1.2 许可 / 条款（原文引用）

- **SceneParse150 页面** http://sceneparsing.csail.mit.edu/ 页脚：写 "License BSD 3-Clause License"，链接到 https://github.com/CSAILVision/sceneparsing/blob/master/LICENSE（`C/ade_sceneparsing_site.txt`）。该 LICENSE 首行为 "BSD 3-Clause License / Copyright (c) 2019, MIT CSAIL Computer Vision"。
- **ADE20K Terms of Use** https://ade20k.csail.mit.edu/terms/（`C/ade_terms_page.txt`）：
  - 图像部分："MIT, CSAIL does not own the copyright of the images. … Researcher shall use the Database only for non-commercial research and educational purposes. … If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions"
  - 标注部分："Software and Annotations — This website, image annotations and the software provided belongs to MIT CSAIL and is licensed under a Creative Commons BSD-3 License Agreement"
- HF `zhoubolei/scene_parse_150` 卡片："The MIT Scene Parsing Benchmark dataset is licensed under a BSD 3-Clause License"。

#### C1.3 Split 与计数

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

#### C1.4 类别数与标签编码（官方文档）

- sceneparsing README（`C/ade_sceneparsing_README.md`）原文："There are in total 150 semantic categories included in the benchmark for evaluation"；"Note: annotations masks contain labels ranging from 0 to 150, where 0 refers to "other objects". We do not consider those pixels in our evaluation."；"As mentioned before, we ignore pixels labeled with 0's."
- objectInfo150.txt（`C/ade_objectInfo150.txt`）共 150 行，列为 `Idx Ratio Train Val Name`，Idx 从 1 到 150。第 1 类是 wall，第 150 类是 flag。
- 实测两张标注 PNG（`C/ade_annotation_png_probe.txt`）：
  - 格式：mode `L`，uint8，单通道。
  - ADE_val_00000001.png：值域 0–18，0 占 1.03%。
  - ADE_train_00000001.png：值域 0–150，0 占 9.84%。
- 完整 ADE20K 的 `*_seg.png` 用另一套编码。官网原文："The channels R and G encode the objects class masks. The channel B encodes the instance object masks"，而且类别表有 3,688 类，不是 150 类（`C/ade_homepage.txt`、`C/ade_ADE20K_github_README.md`）。

#### C1.5 我们需要的子集大小

- train（20,210 张图 + 20,210 张标注）和 val（2,000 + 2,000）都在同一个 zip 里：**967,382,037 B（HEAD）**。
- 分开算（压缩后）：train 图像约 788.0 MB、标注约 80.2 MB；val 图像约 79.7 MB、标注约 8.3 MB。

---

### C2. CV-Bench（nyu-visionx/CV-Bench）

#### C2.1 来源与许可

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

#### C2.2 Split、计数与任务

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

#### C2.3 图像来源与所用 split（逐行字段统计）

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

#### C2.4 真实 schema（datasets-server `/rows` 返回的 features，`C/cvbench_rows_run.txt`）

```
idx int32 | type string | task string | image Image | question string | choices List<string> | answer string |
prompt string | filename string | source string | source_dataset string | source_filename string |
target_class string | target_size int32 | bbox List<List<float32>>
```
- answer 形如 `"(C)"`。
- bbox 只在 Omni3D 行有值，例如 `[[373.91,432.78,670.32,583.83],[177.79,274.94,351.70,343.55]]`。
- target_class 和 target_size 按卡片说明只对 COCO 行有值。

#### C2.5 大小

- 需要的就是整个 test，parquet 合计 **404,808,364 B**。default config 与 2D+3D 是同一批数据（`/size`）。
- 只要文本和元数据的话：test_2d.jsonl + test_3d.jsonl 共 1,890,033 B。

---

### C3. EmbSpatial-Bench

#### C3.1 来源与许可

| 来源 | 性质 | 内容 | 许可 |
|---|---|---|---|
| GitHub https://github.com/mengfeidu/EmbSpatial-Bench | **官方**代码与说明（`C/embspatial_github_README.md`） | 数据链接指向 HF `Phineas476/EmbSpatial-Bench`，原文："We share the datset in: EmbSpatial-Bench (containing our benchmark data and sft data) for model evaluation and training." | GitHub API 返回 `license: null`，仓库没有 LICENSE 文件 |
| HF `Phineas476/EmbSpatial-Bench` | **官方数据发布**（作者 README 指向此处） | `embspatial_bench.json`：253,512,493 B（图像以 base64 内嵌）<br>`embspatial_sft.json`：1,201,627,311 B | 卡片 `license: cc-by-4.0`（`C/hf/Phineas476__EmbSpatial-Bench.json`） |
| HF `FlagEval/EmbSpatial-Bench` | 第三方转成的 parquet | `data/test-00000-of-00001.parquet`：408,685,444 B | 卡片**没有 license 字段**，原文："This dataset is organized and adapted from Phineas476/EmbSpatial-Bench" |
| 论文 | arXiv:2406.05756 | | — |

- datasets-server 对 Phineas476 仓库给出一个自动命名的 `default/train` split，但 size/info 计算失败（`failed: config-size`）。这个 "train" 名称是自动生成的，不对应论文里的任何 split。

#### C3.2 Split 与计数

- FlagEval parquet：**只有 `test`，3,640 行**。来源是 datasets-server `/size`（`C/hf/FlagEval__EmbSpatial-Bench.json`）和 `/statistics`（`C/embspatial_flageval_statistics.json`，`partial: False`）。
- `/statistics` 结果：
  - `data_source`：mp3d 1,201 / ai2thor 1,239 / scannet 1,200
  - `relation`：left 616 / right 620 / above 596 / under 602 / close 612 / far 594
  - `scene_id` 去重 294 个：ScanNet 175 个 scan（来自 90 个不同的 scene 编号）、MP3D 26、AI2-THOR 93
  - `answer` 取值 0–3；`answer_options` 恒为 4 个
- 行顺序：offset 0–1200 为 mp3d，1201–2439 为 ai2thor，2440–3639 为 scannet。依据 `/rows` 在边界 offset 上的抽样（`C/embspatial_flageval_rows_off*.json`）。
- 论文 Table 1 原文："Matterport3D 1,201 928 133 26 | AI2-THOR 1,239 683 95 93 | ScanNet 1,200 570 35 175 | Overall 3,640 2,181 294 277"，列依次为 #QA Pairs、#Image、#Object、#Scene。
- **训练集：** 有，是单独的 EmbSpatial-SFT（`embspatial_sft.json`，1.2 GB）。论文原文："EmbSpatial-SFT is solely built on the training split of MP3D"；"We construct 25K training samples for spatial relation identification"。

#### C3.3 图像来源（原文引用）与场景 split 核对

- 论文："The images used for evaluation are directly collected from embodied 3D scenes, namely MP3D (Chang et al., 2017), AI2-THOR (Kolve et al., 2017) and ScanNet (Dai et al., 2017)"；"Specifically, we utilize the test scenes from MP3D and validation scenes from ScanNet and A[I2-THOR]"。arXiv HTML 在这里原文被截断成 "and A."。另有："In AI2-THOR, we select 7 types of household tasks from ALFRED"。
- 用官方 split 列表对照（`C/embspatial_scene_split_check.txt`）：
  - **ScanNet**：175/175 个 scan 都在 `scannetv2_val.txt`（312 个）中，在 `scannetv2_train.txt` 中为 0，覆盖 1,200/1,200 行。列表来自 ScanNet/ScanNet GitHub 的 `Tasks/Benchmark/`。
  - **MP3D**：26 个场景中，**17 个**在 `scenes_test.txt`（18 个）中，**9 个**在 `scenes_val.txt`（11 个）中，在 train（61 个）中为 0。按行算，1,131 行来自 test 场景，70 行来自 val 场景。列表来自 niessner/Matterport GitHub 的 `tasks/benchmark/`。
  - **AI2-THOR**：共 93 个 FloorPlan，编号覆盖 1–30、201–230、301–330、401–430 中的大部分，完整列表见上面的 check 输出。ALFRED 的 split 定义**未核实**。

#### C3.4 真实 schema（FlagEval，datasets-server `/first-rows` 的 features，`C/embspatial_flageval_first_rows.json`）

```
data_source string | scene_id string | question_id string | question string | relation string | image Image |
answer_options Sequence<string> | answer int32 | objects Sequence{bbox: Sequence<int32, length 4>, name: string} |
image_base64 string
```
- FlagEval 卡片说 bbox 是 "(x_min, y_min, x_max, y_max)"。实测抽样见 C3.5 和末节。
- 图像尺寸：mp3d 640×480，ai2thor 300×300，scannet 1296×968。依据抽样行和 `/statistics` 的 image 列（min 300，max 1296）。
- `image_base64` 列长度 6,576–285,428 字符，均值 69,133。

#### C3.5 bbox 格式实测（抽样 10 行，`C/embspatial_flageval_rows_off*.json`、`C/embspatial_flageval_first_rows.json`）

- mp3d 行：`[343,406,102,74]`、`[361,1,279,53]`（图宽 640、高 480）。按 xywh 读，y+h=480、x+w=640，恰好落在边缘。官方 GitHub README 的 mp3d 示例 `[123,256,130,73]` 按 xyxy 读会出现 y2<y1。
- scannet 行：`[596,508,444,319]`，按 xyxy 读 x2<x1；按 xywh 读 `[186,538,357,429]` 得 y+h=967（图高 968）。
- ai2thor 行：`[68,245,210,269]` 按 xywh 读 y+h=514，超过图高 300；按 xyxy 读则落在图内。`[277,1,299,47]`、`[253,1,299,29]` 的 x2=299，即图宽减 1。

#### C3.6 大小

- 需要的是整个 test：FlagEval parquet 408,685,444 B，或 Phineas476 的 `embspatial_bench.json` 253,512,493 B。
- SFT 1,201,627,311 B。

---

### C4. ODinW13（GLIP 使用的 13 个数据集）

#### C4.1 数据集清单与官方来源

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

#### C4.2 各数据集的 split 计数与大小

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

#### C4.3 许可（Roboflow `README.dataset.txt` 原文，完整文本存于 `C/odinw13_ann_stats.json` 的 readmes 字段）

- **Public Domain：** AerialMaritimeDrone、CottontailRabbits、NorthAmericaMushrooms、Packages、pistols，共 5 个。标注 json 的 `licenses` 字段指向 CC0 1.0 URL。
- **CC BY 4.0：** Aquarium、EgoHands、PascalVOC、ShellfishOpenImages、VehiclesOpenImages，共 5 个。
  - EgoHands 原文："It is released under a Creative Commons by Attribution 4.0 license. You may use it for academic or commercial purposes but must cite the original paper"
- **MIT：** Raccoon。原文："originally collected by Dat Tran, released with MIT license"
- **ODbL v1.0：** pothole。
- **undefined：** thermalDogsAndPeople。README 为 "License: undefined"，json 里 `licenses: [{"name": "Unknown"}]`，contributor 为空，url 为 `https://public.roboflow.ai/object-detection/undefined`。
- AerialMaritimeDrone 的 README 里同时出现 "License: Public Domain" 和 "This dataset was originally collected by the Roboflow team, released with MIT license"。

#### C4.4 标注格式

- 格式为 **COCO detection json**。GLIP README 原文："Currently, the code supports evaluation on COCO-formatted dataset"；"Remove the background class from the annotation file … remove the entry with "id:0" from "categories""。`annotations_without_background.json` 就是按这个做法去掉背景类后的文件。
- 实测字段：
  - 顶层：`categories, images, annotations, info, licenses`
  - images：`id, file_name, width, height, license, date_captured`
  - annotations：`id, image_id, category_id, bbox, area, segmentation([]), iscrowd`
  - bbox 为 COCO 的 `[x, y, w, h]`，可以是浮点，例如 `[870, 342, 30.5, 25.5]`
- GLIP 的 13 个 config 都设 `TRAIN: ("train",)`。

---

### C5. 各基准可用的 split（HF / 官方元数据）

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

### 值得注意的事实（只陈述不解读）

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

### 下载记账

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

## 探针用图（D20：这些图从所有评测子集中排除）

| 图像 | 来源 | 用在 |
|---|---|---|
| `000000039769.jpg` | COCO val2017 | RUN-0001、0002、0004、0005 |
| `000000000139.jpg` | COCO val2017 | RUN-0001、0002、0005 |

## 训练数据的选定来源（DEV-0022；D11-1、D11-2、D11-7）

- 核实时间：2026-10-02，开发机。只读了元数据：HEAD、HF API、GitHub API、各 zip 的中央目录（Range）。另有 `--limit 3` 的小样本落地，放在 scratch 目录，不入库。
- 钉住的值都写在落地代码的 `SPEC` 里（`src/data/land/{refcoco_train,nyu_depth_train,nyu_normals,ade20k}.py`），测试核对。
- "尾部"指 zip 从中央目录起点到文件末尾的字节，含 zip64 记录与 EOCD。
  - 尾部的 sha256 相同，说明每个成员的名字、偏移、大小与 CRC32 都相同。
  - 落地时先核对尾部，再逐个成员核对 CRC32。用本地副本（`--local-zip`）时也照此核对。

| 源 | 选定来源与 revision | 尾部（中央目录） | 落地内容与实测体积 |
|---|---|---|---|
| `refcoco_train` | 图像：`http://images.cocodataset.org/zips/train2014.zip`。13,510,573,713 B，Last-Modified 2018-07-11，ETag `"289af3433b84b5b8622626b10146c1bc-1611"`，82,784 个条目，全部 stored。refs 与 mask：E1 钉住的 refer 存档（A §1.1 的 Wayback 快照），用 Range 只取 `refs(unc\|umd).p` 与 `instances.json` | train2014.zip：起点 13,500,707,642，9,866,071 B，sha256 `7b2e2131…549e`。refer 三个 zip 的尾部各 205–377 B，sha256 见 SPEC | 24,407 张图，4,146,591,678 B。中央目录的成员字节与 A §1.5 逐张 HEAD 的结果相同。另有三套 train 标注（估 ≤ 250 MB）。开发机抽了 3 个成员，与单图 URL 逐字节相同 |
| `nyu_depth_train` | RGB 与深度：HF `aradhye/nyu_depth_v2@4d7f422d47e78ce7b59d4c635294564cca7fadba`，`nyu_depth_v2.zip`，6,857,931,462 B，LFS sha256 `0eb48e96…f37d`。清单：`cleinc/bts@5e3406b35d1497b2e55d2dd600524d1f4efacaed` 的 `train_test_inputs/nyudepthv2_train_files_with_gt.txt`，1,855,580 B，sha256 `66e2beb7…e0e9`。该文件自首个 commit d1ed7caf 起未改过，与仓库里的 P0 副本相同 | 起点 6,848,271,766，9,659,696 B，sha256 `2e8b36d1…836b`。成员表与 P0 的 listing 逐条相同 | 22,848 对，解压后 3,961,165,376 B。深度 PNG 为 16 位灰度、640×480，整数值 / 1000 = 米，0 为无效（BTS `bts_dataloader.py`）。focal 518.8579 |
| `nyu_normals` | Bae, Budvytis, Cipolla 2021 的 `nyu_dataset.zip`（GeoNet 的 GT 与划分）。Drive 文件 id `1Rl-x6GwIXWdRW6LbnSayjIAfqCqygvYQ`，直链 `drive.usercontent.google.com/download?id=…&export=download&confirm=t`，支持 Range。1,008,950,467 B，Last-Modified 2022-05-02 | 起点 1,008,654,083，296,384 B，sha256 `0202a05c…3291` | train 795 + test 654 对（img + norm），1,008,804,374 B。PNG 为 8 位 RGBA，α 恒为 255。n = rgb/255×2−1，(0,0,0) 为无效。有效区在第 44–470 行、第 40–600 列，解码后长度为 0.987–1.010 |
| `ade20k` | `https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`。967,382,037 B，Last-Modified 2022-07-22，ETag `"39a91415-5e46c686a7a37"`，44,429 个条目 | 起点 961,539,833，5,842,204 B，sha256 `d43af205…ac04` | train 20,210 + val 2,000 对，另有 objectInfo150.txt 与 sceneCategories.txt。即 zip 的全部文件，967,861,125 B。相邻成员合并成约 29 个 Range 请求 |

**D11-1。** 用官方 pickle 的逐条统计（P0 的 `refs_image_ids.json`）计算：

- 三者 train 划分图像的并集为 28,158 张，评测划分图像的并集为 6,549 张。
- 剔除 3,751 张，保留 24,407 张。
- 交叉：refcocog train ∩ refcoco 评测 = 1,860；refcoco train ∩ refcocog 评测 = 1,891。
- 保留 id 的指纹钉在 SPEC 里。落地时用官方 pickle 重算并比对。

**D11-2。** 规则：基础场景名属于 215 个官方 test 场景的帧全部去掉。

- 结果：BTS 清单 24,231 对，去掉 1,383 对，剩 22,848 对。
- 去掉的帧只来自 bookstore_0001d、e、f、g、h、i、j 这 7 个目录，分别为 242、158、348、189、119、114、213 帧。

**法向来源的选择。** 选 Bae 2021。

- 选它的理由：
  - 中央目录可以用 Range 读到。
  - 下标加 1 后，与 splits.mat 的 trainNdxs、testNdxs 逐个相同。
  - 训练与测试同在一个包里，编码也相同，满足 D11-2"训练与测试同源、覆盖 654"。
- 其他候选为什么不选：
  - Eigen & Fergus 的 `normals_gt.tgz`：只有 Wayback 存档，又是 tar.gz，成员数核实不了。
  - Ladický 2014：链接已失效，没有存档。
  - GeoNet-test 镜像与 Marigold 的评测包：只有 654 张测试图。
  - `tanganke/nyuv2`：是 288×384 的缩小版。
- 许可：未声明。Bae 的仓库是 MIT（指代码）；GeoNet 没有 LICENSE。

**去重（`scripts/data/dedupe_report.py`）。**

- 阈值：感知哈希的汉明距离 ≤ 8，即 `PHASH_MAX_HAMMING`。报告另列 0–12 各档的计数。
- 开发机的标定数据：

| 情况 | 汉明距离 |
|---|---|
| 同一张 NYU test 图的两种加工：Bae 包的 PNG，对 BTS official_splits 的 JPEG（外圈置黑，平均像素差 16–23） | 2、4 |
| 12 张真实图做合成变换：JPEG q50 | 0 |
| 缩到一半 | ≤ 2 |
| 亮度 ×1.2 | ≤ 6 |
| 外圈 7 px 置黑 | ≤ 6 |
| 四边各裁 3% | 0–14 |
| 四边各裁 8% | 8–24 |
| 水平翻转 | 30–34（感知哈希查不出） |
| 12 张互不相同的真实图，两两之间 | 最小 24 |

- 正式的标定写进去重报告，用评测集之间的已知重合：
  - CV-Bench 的 COCO 图对 E1 的 COCO 子集，83 张；
  - CV-Bench 的 NYU 图对 E1 的 NYU test，36 张；
  - 法向包的 RGB 对 E1 的 NYU 654 张与开发子集 50 张。
- 国内网络的备选：
  - `--local-zip`：本地 zip，同样核对尾部 sha256 与 CRC32；
  - `--refer-zip-dir`：refer 的三个 zip；
  - `--hf-base https://hf-mirror.com`；
  - BTS 清单：默认用仓库里的副本，不联网。
