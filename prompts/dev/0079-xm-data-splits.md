[DEV-TASK 0079] X-M 的数据：COCO 2017 的读取、三个评测划分与密封、去重清单、LVIS 与 ReasonSeg、训练目标（框列表 + 每实例 B 样条）、长度统计

## 背景

- 负责人新开 X-M 线（D165，要点原文 `docs/lead_answers/2026-11-03-xm-directive.md`，先读全文）：用 X-2 的同一套做法，让一个 LLM 同时做
  - **类别提示的目标检测**：给一个类别，框出图中所有实例；
  - **实例分割**：为每个实例描 16 个控制点的 B 样条，GrabCut 三元图贴边。
- 本任务只做数据层。推理格式、评测与指标由 DEV-0080 并行做，训练数据流与训练配置由 DEV-0081 随后做。三者共用下面"接口"一节定的数据结构，**不要改它的字段名**。
- 原始数据由服务器落地（RUN-0102），开发机上没有：
  - `$VI/data/coco2017/{images/train2017,images/val2017,annotations}`；
  - `$VI/data/lvis/lvis_v1_val.json`；
  - `$VI/data/reasonseg/val/`。
  开发机上用 `tests/fixtures` 里造的小数据测，格式照 COCO 官方 json。

## 开工前（省盘）

开发机磁盘紧（约 12 GB，另一个开发 agent 同时在跑）。开工前在**你自己的 worktree 里**执行 `rm -rf experiments/`：本任务用不到已有结果，可省约 6 GB。不要 `git add` 这些删除，也不要动主仓库。只跑你自己的测试。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/lead_answers/2026-11-03-xm-directive.md`（第二节"数据与划分"最要紧）、`DECISIONS.md` 的 D165、D28-9
- X-2 的数据与样条：`src/explore/x2_data.py`（真值 mask 拟合 B 样条、0–1000 网格的 JSON）、`src/explore/guidance.py`（拟合与光栅化）
- 划分与密封的写法：`src/data/explore_splits.py`、`configs/explore_splits.json`（devcheck3 的密封守卫 G-EX）、`src/llmseg/runner.py`（hash_rank 子集）
- RefCOCO 系列的加载：`src/data/` 里 refcoco 相关的加载器，以及 W2 的 `configs/w2_splits.json`、`configs/explore_splits.json`、`configs/w2_llmseg_subsets.json`

## 接口（三个任务共用，写在 `src/xm/data.py`）

```python
@dataclass(frozen=True)
class XMInstance:
    ann_id: int
    box_xyxy: tuple[float, float, float, float]   # 原图像素，COCO bbox 换算
    rle: dict                                      # COCO RLE（size、counts），由多边形或 RLE 统一转
    area: int

@dataclass(frozen=True)
class XMPair:          # 评测与训练的单位：图像 × 该图中出现的类别
    id: str            # "coco2017/<val2017|train2017>/<image_id>/<category_id>"；LVIS 与 ReasonSeg 照此命名
    source: str        # "coco2017" | "lvis_v1" | "reasonseg"
    split: str         # "xm_tune" | "xm_decision" | "xm_final" | "lvis_sub" | "reasonseg_val" | "train"
    image: str         # 相对数据根的路径
    width: int; height: int
    category_id: int; category: str                # 类别名就是提示里给模型的词
    instances: tuple[XMInstance, ...]              # 非 crowd 实例；可为空吗？——不：只为"图中出现的类别"建对
    crowd_rles: tuple[dict, ...]                   # 同类的 crowd 区域，评测时用来忽略落在其中的预测（COCO 惯例）
```

另给 `load_pairs(split, data_root) -> list[XMPair]` 与 `pair_by_id`。

## 任务

1. **COCO 2017 读取**：从官方 instances json 建 `XMPair`。
   - crowd 标注不进 `instances`，进 `crowd_rles`。
   - 一个类别在图中只有 crowd、没有非 crowd 实例时，这一对不建，记数。
2. **三个评测划分**（指令第二节），都从 COCO val2017 抽，按图像，三份互不重叠：
   - 调参 `xm_tune` 300 张；判定 `xm_decision` 500 张；最终报告 `xm_final` 1,000 张。
   - 抽法：对 image_id 用 hash_rank（同 `src/llmseg/runner.py` 的写法，salt `xm_splits_v1`）排序，依次取前 300、接着 500、再接着 1,000。只从至少有一个非 crowd 实例的图里抽。
   - 写 `configs/xm_splits.json`：每份的 image_id 清单、句对数（图 × 类别）、sha256、规则说明。
   - **密封**：
     - `xm_decision` 照 devcheck3 的做法加守卫，门名 `G-XM`：读取它必须给出门名，否则拒绝；
     - `xm_final` 另加一道：必须给出 `G-XM` 判定通过的 gate 文件，类似 DEV-0072 的 `--test-gate`。这一道的接口在这里定好，检查逻辑留给 DEV-0080 的运行器调用。
3. **LVIS 子集**（D165 的管理员补充）：从 LVIS v1 val 里**只取 COCO val2017 的图**、且不在上面三个划分里的图，hash_rank（salt `xm_lvis_v1`）取 1,000 张。
   - 用 LVIS 的类别与标注建 `XMPair`。LVIS 的 `not_exhaustive_category_ids` 与 `neg_category_ids` 照官方惯例记进对的元数据，评测时 DEV-0080 用。
4. **ReasonSeg val**：读官方格式（LISA 的 ReasonSeg json 与图）。
   - 每条是一个推理型的指代表达式与一个 mask，单位是"图像 × 表达式"，建成同样的 `XMPair`，`category` 字段放表达式原文，`instances` 只有 1 个；
   - 标出 `is_sentence`（长句与短语两类）。
5. **去重清单**（指令第二节，最要紧）：生成 `configs/xm_train_exclusions.json`，含：
   - RefCOCO、RefCOCO+、RefCOCOg 的**所有评测划分**的 image_id：val、testA、testB、test，并核对 devcheck2、devcheck3、explore_val、devtune800、W2 test 子集的图都在里面；
   - ReasonSeg val 的图：它们不来自 COCO，按文件名与图像 sha256 两种方式核对是否与 train2017 有重合，记录结果；
   - `xm_decision`、`xm_final`、`xm_tune`、LVIS 子集的图：都在 val2017，与 train2017 不重叠，照样核对并记录。
   - 每一类的条数、清单 sha256，以及与 COCO train2017 的交集大小。`load_pairs("train")` 自动剔除。
6. **训练目标**（COCO train2017，剔除之后）：每个 `XMPair` 生成两个文本目标，写成分片 jsonl 与 manifest，供 DEV-0081 的训练器读。
   - **框列表**：该类全部实例的框，0–1000 网格的整数 `[x1, y1, x2, y2]`，按阅读顺序排。
     - 阅读顺序先按框中心的 y 分行：两个框的中心 y 差小于两者高度的较小者的一半就算同一行。行从上到下，行内按中心 x 从左到右。写成函数，有测试。
     - 最多 30 个；超过的只保留阅读顺序的前 30 个，记数。
   - **每实例 B 样条**：对每个框，用 X-2 的同一拟合函数从真值 mask 拟合 16 个控制点，0–1000 网格，控制点限制在各自框内；记拟合 IoU。
   - 格式与 X-2 的样条目标一致，便于复用解析。
7. **统计**（给负责人的可行性报告用）：`scripts/xm/data_stats.py` 在 train 与三个评测划分上统计：
   - 对数、每对实例数的分布（中位数、p90、p99、最大值）；
   - 两个目标的 token 数分布，用 Qwen3.5 的分词器；
   - 截到 30 个的比例。
   - 输出 json 与 md。服务器在 HPC 上跑。
8. **测试**（`HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/ -q -k "xm"`）：
   - fixture 上建对：crowd 处理、空类不建；
   - 划分：确定、互不重叠、清单 sha256 稳定；
   - 密封：不给门名读 `xm_decision` 被拒；
   - LVIS 子集只取 val2017 的图、不与三个划分重叠；
   - 去重：fixture 里放一张"RefCOCO 评测图"进 train，被剔除；
   - 阅读顺序；最多 30 个；样条控制点在框内；
   - `--out` 末级目录名不带运行编号时拒绝（CLAUDE.md"服务器驱动的写法"）。
9. 在总结里给出服务器命令：
   - 在 HPC 上用真数据生成 `configs/xm_splits.json` 与 `configs/xm_train_exclusions.json`（这两份生成后由管理员审过再提交）；
   - 生成训练目标、跑统计；
   - 给出用时估计。

## 不做

- 不写推理、评测、指标（DEV-0080），不写训练器（DEV-0081）。
- 不改 X-2 的任何代码、配置与结果。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；接口有无偏离；
- 服务器命令与用时；
- 测试结果原文；
- 待裁的地方，尤其是对指令第二节的任何不确定。

尽量 3 小时内完成。
