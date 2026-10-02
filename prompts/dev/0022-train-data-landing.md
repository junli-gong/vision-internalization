[DEV-TASK 0022] E2/E3 训练数据落地：RefCOCO 训练图、NYU 深度训练集（BTS）、NYU 法向、ADE20K，加跨机器的去重报告（D11、硬约束 7）

## 背景

- E2 的探针与 E3 的训练都要训练数据，现在一样都没落地。
  - E1 的落地源（`src/data/land/`、`scripts/data/land_e1.py`）只覆盖评测集与开发子集。
- 体积约 10–11 GB。B200 的根盘份额是 15 GB，已用约 4.5 GB，放不下（D28-8）。
  - 这些数据落在 A800 上：700 GB 磁盘，由合作者手动运行脚本包（D35、D47）。
  - 所以落地脚本必须**机器无关、可离线续传、自带核对**。
- D11 已定的规则：
  1. **RefCOCO 训练图**：RefCOCO(unc)、RefCOCO+(unc)、RefCOCOg(umd) 三者 train 划分的图像并集，减去三者任一评测划分里出现过的图像（按 COCO image_id）。
  2. **NYU 深度训练集**：用 BTS 清单（24,231 对），按场景过滤。基础场景名属于官方 test 场景的帧全部去掉，至少包括 bookstore_0001d–j 的 1,383 帧。过滤前后的帧数写进去重报告。
  3. **NYU 法向**：GT 的来源在本任务里选定，要求训练与测试同源、覆盖 654 测试集。选定之前不做法向训练。
  7. **去重**：落地后按 image_id 与感知哈希，在所有训练集与所有评测集之间去重，出去重报告。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §4（硬约束 7：数据泄漏）、§7.3、§7.5、§12（数据泄漏）
- `DECISIONS.md` 的 D11、D20（探针用图）、D28-8/9（份额、非商业）、D30（E1 数据落地）、D34、D47
- `docs/data_sources.md` 的 A §1（RefCOCO）、B §1（NYU：BTS sync.zip、HF `aradhye/nyu_depth_v2`、Eigen 法向）、C §1（ADE20K）
- 已合入的 `src/data/land/`（common、net、manifest、refcoco、nyuv2、coco 等）、`scripts/data/land_e1.py`、`src/data/` 的加载器与去重

## 任务

### 1. 新的落地源（只做加法，E1 的源与行为不变）

新入口 `scripts/data/land_train.py <源> --root <数据根> [--tmpdir] [--budget-gb] [--min-free-gb] [--workers] [--limit N] [--dry-run]`，沿用 `land_e1.py` 的约定：
- 断点续传与幂等；
- `MANIFEST.json`：来源与 revision、逐文件 sha256 与字节数、许可原文、git commit、估算、df；
- 体积预算与磁盘红线；
- zip 用 Range 只取需要的成员；
- 失败重试。

| 源 | 内容 | 来源（以 `docs/data_sources.md` 为准，选定的写进 SPEC 并钉住） |
|---|---|---|
| `refcoco_train` | D11-1 的 RefCOCO 训练图（COCO train2014 的 jpg，约 5.3 GB）与三套 refs 的 train 划分标注 | COCO 官方 `train2014.zip`，Range 只取需要的成员；refs 复用 E1 已钉住的存档 |
| `nyu_depth_train` | D11-2 的 BTS 训练对（RGB jpg + 深度 png，过滤前约 4.2 GB），加 focal | 优先用 HF `aradhye/nyu_depth_v2`（固定 revision），Range 只取 BTS 清单里的成员。逐对核对：文件名与 BTS 清单一一对应、深度 png 是 16 位、尺寸 640×480。BTS 清单本身从 cleinc/bts 的固定 commit 取 |
| `nyu_normals` | NYU 法向 GT，训练（labeled 795）与测试（654）同源 | 在 data_sources 列的候选里选一个能核实的来源（例如 Eigen 的 `normals_gt.tgz`，现在只有 Wayback）。都核实不了就**不落地**，在总结里写清各候选的状态，由管理员定 |
| `ade20k` | ADEChallengeData2016 的 train 20,210 + val 2,000（图像加标注，约 0.97 GB） | MIT SceneParse150 官方 zip |

- **许可**：照 data_sources 记的原文写进 MANIFEST。ADE20K 图像仅限非商业研究（D28-9）。
- **加载器**：每个源配一个训练集加载器，接口参照 `src/data/` 现有的 Sample。字段：图像路径、GT（深度以米为单位的 float、法向单位向量、ADE 标签 0–150、RefCOCO 的框与多边形）、来源 id、感知哈希。
- **`--limit N`**：开发机小样本，每个源只落 N 条。加上 fixture，测试离线可跑。

### 2. 去重报告（硬约束 7）

新增 `scripts/data/dedupe_report.py --train-root <A800 数据根> --eval-manifest <B200 上 E1 各源的 MANIFEST 或哈希清单> ... --out <json/md>`：
- **跨机器设计**：
  - 训练数据在 A800，评测数据在 B200。各自导出"image_id 加感知哈希"的清单（小文件），再在任意一台机器上比较。
  - 导出清单的命令也写好：`--export-hashes <数据根> --out <清单>`。
- **比较内容**：
  - 按 image_id：COCO train2014 对 COCO val2017；RefCOCO 训练图对 RefCOCO 评测图；
  - 按感知哈希：所有训练集对所有评测集。评测集包括 E1 的全部评测源、开发子集、探针用图（D20），以及 CV-Bench 含的 ADE20K val 633 张、COCO val2017 805 张、NYU test 36 张（D11-7）。
  - 阈值写明，并在已知重合的例子上标定。
- **输出**：
  - 每对集合的重合条数与例子；
  - D11-1、D11-2 过滤前后的条数；
  - 被剔除的训练样本清单。训练加载器读这份清单，自动排除这些样本。

### 3. 测试

- 每个源在 fixture 上：落地、续传、核对、加载器字段。
- D11-1 的减法、D11-2 的场景过滤：在构造的小例子上与手算一致。
- 去重：构造的重合能被查出，阈值边界正确。
- 不联网就能跑全部测试。

## 允许改动的文件

- 新建：`src/data/land/` 下的新源文件、`scripts/data/land_train.py`、`scripts/data/dedupe_report.py`、`src/data/` 下的新加载器与去重模块、对应的 `tests/test_data_train_*.py` 与 fixture
- `src/data/land/common.py`、`net.py`：只为复用加通用函数，现有行为不变
- `docs/data_sources.md`：在末尾加一节"训练数据的选定来源"，写明钉住的 URL、revision、实测体积

**不许碰**：E1 的落地源与 `land_e1.py` 的行为、`src/e1/`、`src/e5/`、`src/adapters/`、`src/synth/`、`configs/`。

## 环境与验收

- conda 环境 `visint`，不装包。
- 为核实来源与体积，允许联网读元数据：HEAD、zip 中央目录（Range 读几十 KB）、HF API。**不下载数据本体**；`--limit` 的小样本例外，最多几 MB，不入库。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 每个源 `--dry-run` 给出估算
  - `--limit 3` 在开发机上实际落地一遍（落到 scratch 目录，不入库）
- 总结里写明：
  - 每个源选定的来源、钉住的 revision、实测或估算体积；
  - 法向来源的选择及理由；
  - D11-1、D11-2 过滤的条数（能算的先算）；
  - 去重的阈值与标定；
  - A800 上的运行步骤：给出可以直接放进脚本包的命令序列，预计用时、磁盘、网络要求。国内网络可能连不上 huggingface.co 或 web.archive.org，备选办法一并写明。
