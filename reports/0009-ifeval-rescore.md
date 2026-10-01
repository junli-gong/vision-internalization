# RUN-0009 — punkt_tab 安装与 8 个基座的 IFEval 补评分；权重下载清单

服务器 CC，b200，2026-10-01 04:43–04:46 UTC（值守）。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- punkt_tab 下载到 `$PROJ/nltk_data`（nltk 3.10.3），english 下 4 个文件的 sha256 与钉住值全部相同。
- 8 个基座的 IFEval 补评分全部退出码 0、`ok=True`，每个约 1.8 s。MMLU-Pro 与 GSM8K 的分数与 RUN-0007 原样相同。
- 权重清单覆盖 8 个模型、共 112 个文件，revision 全部与 HF API 读到的 sha 相同；用本地的 Qwen3.5-0.8B 快照核对，13/13 一致。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `9291a99`；本任务书所在的 commit `0dbf3bf` 在 HEAD 历史中。
- 没有修改 `src/`、`scripts/`、`tests/`、`configs/`；没有安装 Python 包。新写的只有 `experiments/P0/0009_ifeval_rescore/probe_scripts/` 下的清单与核对脚本。
- 没有写全局的 `~/nltk_data`：下载前后 `~/nltk_data` 都不存在。`NLTK_DATA` 只在本任务的 shell 里 export。

## 2. 执行记录

| 步骤 | 命令 | 墙钟 | 结果 |
|---|---|---|---|
| 下载 punkt_tab | `python -m nltk.downloader -d $PROJ/nltk_data punkt_tab` | 约 1 s | 解压到 `$PROJ/nltk_data/tokenizers/punkt_tab/`（含所有语言，15 MB；punkt_tab.zip 4,319,076 B） |
| 哈希核对 | 比对 `src/health/protocol.py` 的 `PUNKT_TAB_ENGLISH_SHA256` | <1 s | 4/4 相同 |
| IFEval 补评分 ×8 | `export NLTK_DATA=$PROJ/nltk_data; python scripts/health_check.py --rescore experiments/P0/0007_health_base/health_<键>.json --out experiments/P0/0009_ifeval_rescore/health_<键>.json --only-ifeval` | 各 1.76–1.80 s | 全部 rc=0 |
| 权重清单 | `p12_weights_manifest.py weights_manifest.json`（只读 HF API） | 数秒 | 8 个模型 |
| 清单抽查 | `p12_verify_weights.py qwen3.5-0.8b`（本地快照，不下载） | 1.0 s | 13/13 一致，rc=0 |

`nltk_download.log`（原样）：
```
nltk version: 3.10.3
download start (UTC): 2026-10-01T04:43:52Z
<frozen runpy>:128: RuntimeWarning: 'nltk.downloader' found in sys.modules after import of package 'nltk', but prior to execution of 'nltk.downloader'; this may result in unpredictable behaviour
[nltk_data] Downloading package punkt_tab to
[nltk_data]     /home/ubuntu/workspace/wzg/gjl/vision-
[nltk_data]     internalization/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt_tab.zip.
download end (UTC): 2026-10-01T04:43:53Z
```

## 3. 结果表（原样）

### 3.1 punkt_tab 核对（`punkt_tab_check.txt` 原样）

```
nltk 3.10.3 | dir $PROJ/nltk_data/tokenizers/punkt_tab/english
files in dir: ['abbrev_types.txt', 'collocations.tab', 'ortho_context.tab', 'sent_starters.txt']
abbrev_types.txt: local 92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779 | pinned 92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779 | EQUAL
collocations.tab: local 8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf | pinned 8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf | EQUAL
ortho_context.tab: local 4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c | pinned 4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c | EQUAL
sent_starters.txt: local f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac | pinned f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac | EQUAL
files not in pin: []
punkt_tab.zip sha256 e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106 bytes 4319076
ALL_EQUAL
```

### 3.2 IFEval 四个口径，与 RUN-0007 的 MMLU-Pro、GSM8K 合表

| 模型 | IFEval prompt-level strict（标题） | IFEval instruction-level strict | IFEval prompt-level loose | IFEval instruction-level loose | MMLU-Pro 准确率（RUN-0007，32 token） | GSM8K flexible（RUN-0007） | GSM8K strict（RUN-0007） |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 64.0（32/50） | 74.0（57/77） | 66.0（33/50） | 76.6（59/77） | 14.0（14/100） | 58.0（29/50） | 56.0（28/50） |
| qwen3.5-2b | 74.0（37/50） | 83.1（64/77） | 80.0（40/50） | 87.0（67/77） | 27.0（27/100） | 76.0（38/50） | 72.0（36/50） |
| gemma-4-e2b-it | 78.0（39/50） | 84.4（65/77） | 86.0（43/50） | 90.9（70/77） | 25.0（25/100） | 80.0（40/50） | 78.0（39/50） |
| internvl3.5-2b-hf | 72.0（36/50） | 79.2（61/77） | 72.0（36/50） | 81.8（63/77） | 34.0（34/100） | 80.0（40/50） | 76.0（38/50） |
| qwen3.5-4b | 88.0（44/50） | 92.2（71/77） | 92.0（46/50） | 94.8（73/77） | 46.0（46/100） | 82.0（41/50） | 82.0（41/50） |
| gemma-4-e4b-it | 82.0（41/50） | 87.0（67/77） | 88.0（44/50） | 92.2（71/77） | 34.0（34/100） | 88.0（44/50） | 86.0（43/50） |
| internvl3.5-4b-hf | 82.0（41/50） | 84.4（65/77） | 84.0（42/50） | 89.6（69/77） | 46.0（46/100） | 92.0（46/50） | 80.0（40/50） |
| qwen3.5-9b | 92.0（46/50） | 94.8（73/77） | 98.0（49/50） | 98.7（76/77） | 53.0（53/100） | 86.0（43/50） | 86.0（43/50） |

- MMLU-Pro 一列是 RUN-0007 的 32 token 口径。D28-7 已把标题口径改为 256 token，那需要重跑模型，不在本任务内。
- IFEval 的 instruction-level 共 77 条指令，来自 50 题。

重评的元数据（各 JSON 的 `scoring` 字段）：

| 模型 | ok | MMLU-Pro/GSM8K 与 RUN-0007 相同 | punkt_tab 与钉住值相同 | 源文件 sha256（前 16 位） | 评分时间（UTC） |
|---|---|---|---|---|---|
| qwen3.5-0.8b | True | True | True | `4cef6476798fe73a` | 2026-10-01T04:44:11+00:00 |
| qwen3.5-2b | True | True | True | `dae4c1eec7fdc657` | 2026-10-01T04:44:13+00:00 |
| gemma-4-e2b-it | True | True | True | `bc18f2a436cce7b0` | 2026-10-01T04:44:14+00:00 |
| internvl3.5-2b-hf | True | True | True | `92544a42a974781a` | 2026-10-01T04:44:16+00:00 |
| qwen3.5-4b | True | True | True | `2146bd1ebf011365` | 2026-10-01T04:44:18+00:00 |
| gemma-4-e4b-it | True | True | True | `38a392b8035c2a14` | 2026-10-01T04:44:20+00:00 |
| internvl3.5-4b-hf | True | True | True | `dcd57cc6b0f79c8f` | 2026-10-01T04:44:22+00:00 |
| qwen3.5-9b | True | True | True | `e16d3a8722087071` | 2026-10-01T04:44:23+00:00 |

- 评分代码：commit `9291a9980e0572f8be7389eb3c92f7cfe2cfa1c7`；包版本 nltk 3.10.3、langdetect 1.0.9、immutabledict 4.3.1；langdetect 种子 0。
- `scoring.code.dirty=true`：`diff_shortstat` 为空，`n_untracked=1`。评分时本运行的输出目录 `experiments/P0/0009_ifeval_rescore/` 还没提交，所以被算作未跟踪文件；已跟踪的文件没有改动。

### 3.3 权重下载清单（`weights_manifest.json`）

| 模型键 | 仓库 | revision | api_sha 与 revision 相同 | 文件数（LFS / 小文件） | 总字节 | safetensors 字节 |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | Qwen/Qwen3.5-0.8B | `2fc06364715b` | True | 13（2 / 11） | 1,769,980,465 | 1,746,942,600 |
| qwen3.5-2b | Qwen/Qwen3.5-2B | `15852e8c1636` | True | 13（2 / 11） | 4,571,274,023 | 4,548,221,488 |
| qwen3.5-4b | Qwen/Qwen3.5-4B | `851bf6e806ef` | True | 14（3 / 11） | 9,342,907,469 | 9,319,828,096 |
| qwen3.5-9b | Qwen/Qwen3.5-9B | `c20223623576` | True | 16（5 / 11） | 19,329,393,661 | 19,306,310,880 |
| gemma-4-e2b-it | google/gemma-4-E2B-it | `3e22461f65e8` | True | 9（2 / 7） | 10,278,849,571 | 10,246,621,918 |
| gemma-4-e4b-it | google/gemma-4-E4B-it | `ee0ef6023621` | True | 9（2 / 7） | 16,024,823,729 | 15,992,595,884 |
| internvl3.5-2b-hf | OpenGVLab/InternVL3_5-2B-HF | `3f301ffcf3dc` | True | 18（4 / 14） | 4,714,794,051 | 4,696,790,328 |
| internvl3.5-4b-hf | OpenGVLab/InternVL3_5-4B-HF | `6bd448740211` | True | 20（5 / 15） | 9,483,169,313 | 9,465,085,464 |

清单抽查（`manifest_local_check.txt` 原样）：
```
local check model.safetensors-00001-of-00001.safetensors manifest 04b1c301231dd422 local 04b1c301231dd422 EQUAL
local check config.json git blob 715f0448b9d38103 715f0448b9d38103
qwen3.5-0.8b @ 2fc06364715b967f1860aea9cf38778875588b17: 13/13 files verified (/dev/shm/visint/hf-hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17)

real	0m1.011s
user	0m0.868s
sys	0m0.139s
```

恢复步骤写在 `experiments/P0/0009_ifeval_rescore/README.md` 的"重启后按清单恢复权重"一节。

## 4. 与任务书假设不符的发现

1. `python -m nltk.downloader` 打印了一条 `RuntimeWarning: 'nltk.downloader' found in sys.modules after import of package 'nltk', but prior to execution of 'nltk.downloader'; this may result in unpredictable behaviour`。下载与解压正常完成，哈希也一致。
2. `nltk.downloader punkt_tab` 会下载全部语言（15 MB），没有只取 english 的选项；评分只用到 english。
3. InternVL 的仓库里，`examples/image2.jpg`、`examples/red-panda.mp4` 也是 LFS 文件，已收进清单（`snapshot_download` 会一并下载）。

## 5. 自己造成的错误或浪费

- 提交 `21d53d3` 的 commit message 里，权重清单的文件数手写成 "124 files"，实际是 **112**（13+13+14+16+9+9+18+20）。本 report 与 `weights_manifest.json` 里的数由脚本算出，是对的。已推送的 commit message 不改写，在此更正。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0009_ifeval_rescore/`：
- `health_<键>.json` 与 `rescore_<键>.log` 各 8 份
- `weights_manifest.json`、`punkt_tab_check.txt`、`nltk_download.log`、`manifest_local_check.txt`
- `disk_ledger.tsv`、`README.md`、`probe_scripts/`

另有本 report。

**仓库外**：`$PROJ/nltk_data/`（15 MB，不入库）。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 04:43:52 | 0009 start | 42.728 | 1.826 | 0.527 |
| 04:43:53 | after punkt_tab download | 42.712 | 1.826 | 0.543 |
| 04:44:24 | after rescore x8 | 42.709 | 1.826 | 0.546 |
| 04:46:09 | before 0009 commit | 42.709 | 1.826 | 0.546 |

- `$PROJ` 共 0.546 GB，在 D28-8 的 15 GB 份额之内。根盘最低 42.709 GB。
- 内存盘没有变动，只有 Qwen3.5-0.8B 与测试下载的小文件。
