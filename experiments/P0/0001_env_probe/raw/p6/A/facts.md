# RUN-0001 §6 数据来源事实 · A 组（RefCOCO 系列 / COCO val2017 / CountBench）

- 采集时间：2026-09-30 05:49–06:30 UTC，机器 b200。只做 HEAD、HF API 元数据、datasets-server JSON、zip 的 Range 读取。
- 证据目录：`$PROJ/logs/0001/p6/A/`（下文写"证据:"时用相对这个目录的路径）。脚本在 `$PROJ/scripts/p6_A_*.py`。
- 下载记账见 `downloads.tsv`。计量到的网络字节共 58,272,991，另有约 3 MB 元数据 JSON 未计量。落盘只有 168,736 字节，已全部删除。
- 本文只陈述事实，不做推荐。

---

## 1. RefCOCO / RefCOCO+ / RefCOCOg

### 1.1 官方来源与许可

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

### 1.2 refs pickle 的真实 schema

读法：5 个 `refs(*).p` 都用 Range 读进内存，未落盘，再用拒绝一切 global 的受限 Unpickler 解析，默认编码即可解析。证据：`refs_stats.json`。

- 顶层结构：`list[dict]`，一条对应一个 ref。所有 ref 的键集合都相同。
- ref 的字段：`sent_ids: list[int]`、`file_name: str`、`ann_id: int`、`ref_id: int`、`image_id: int`、`split: str`、`sentences: list[dict]`、`category_id: int`。
- sentence 的字段：`tokens: list[str]`、`raw: str`、`sent_id: int`、`sent: str`。
- 示例（refcoco(unc) 第 0 条）：`{"ref_id":0,"ann_id":1719310,"image_id":581857,"split":"train","category_id":1,"file_name":"COCO_train2014_000000581857_16.jpg","sent_ids":[0,1,2],"sentences":[{"raw":"THE LADY WITH THE BLUE SHIRT","sent":"the lady with the blue shirt","sent_id":0,"tokens":[…]},…]}`。`file_name` 末尾的 `_16` 不是真实文件名，README 注释写的是 "ignore filename"。
- sha256：refcoco(unc) `d4a8dd3152f13012…`，refcoco(google) `2bb306561f788f0e…`，refcoco+(unc) `7de24c182449758b…`，refcocog(umd) `0331c7533537b67c…`，refcocog(google) `e4d8320dfd15fc21…`。完整值见 `refs_stats.json` → `member_sha256`。

### 1.3 各 split 计数（官方 pickle，逐条统计）

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

### 1.4 splitBy 之间、数据集之间的关系

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

### 1.5 图像：COCO train2014 单图 URL 与字节数

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

### 1.6 mask 的来源（ann_id → COCO instances）

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

### 1.7 HF 镜像（元数据来自 HF API 和 datasets-server，均未下载数据）

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

## 2. COCO val2017

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

## 3. CountBench

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

## 未能核实（汇总）

- **RefCOCO / RefCOCO+ 标注的数据许可：** refer 的 README 和 zip 里都没有声明。仓库级 Apache-2.0 是否覆盖数据，未核实。
- **RefCOCO 系列 train split 的 iscrowd/polygon：** 需要超过 50 MB 的 instances 文件，只有间接证据（id 区间）和 testB 全量、RefCOCOg val/test 的直接证据。
- **refer `instances.json` 的完整顶层键：** 只读了开头 1.5 KB。
- **train2014.zip 的 82,784 个条目是否含目录条目；COCO 官方文档所称的 82,783 张：** 本次没有直接核实。
- **lmms-lab RefCOCO val=8,811 的组成：** 只做了算术吻合，没有按 id 核实。
- **LibreYOLO mini500 选取的具体是哪些图。**
- **LAION-400M 的许可。**
- **datasets-server `/filter` 查询：** 当天超时，未得到结果。

## 值得注意的事实（只陈述不解读）

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
