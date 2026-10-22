# W2（候选加挑选）开发规划与接口约定（管理员，2026-10-22；依据 D113，原文 `docs/lead_answers/2026-10-22-w2-directive.md`）

W2 的原则：**LLM 只判断是哪个物体、在候选里挑一个；逐像素的拟合全部交给工具，LLM 不动手修改。**
本文件是 DEV-0058（候选与渲染）、DEV-0059（回路、划分、对照、判定）的共同接口。两个任务并行开发，按这里的签名对接；要改约定，写进完成总结，由管理员定。W1 的代码与环境完整保留，W2 在其上开发（D113-3），不改 W1 的行为。

## 0. 硬约束（D113 第二节）

- **工具边界与 W1 相同**：
  - 只用不含学习的算法：numpy、scipy、OpenCV、scikit-image；
  - 不用任何学习型视觉模型或预训练权重，不用 OpenCV 的 DNN 模块。
- **只有两轮**：
  - 提出：LLM 用 E1 的原生格式给框，可选再给几个点；
  - 挑选：LLM 只回答一个编号或"都不对"；
  - 没有修改轮。每轮思考至多 2,048 个 token，到上限强制收尾。
- **按输出类型路由，不按任务路由**：
  - 要 mask 的任务走"提出 → 候选 → 挑选"；
  - 要框或数字的任务（定位、计数）按 E1 协议单轮作答，不用工具。
- **"都不对"或解析失败时交 1 号候选**，1 号候选就是 W1-F 的输出。
- **计算记账**：每条样本记 token 数（提示、生成、补入、视觉）、工具调用（候选生成）次数与墙钟。
- **模型**：Qwen3.5 2B、4B、9B，判定用 9B。不用闭源 API。

## 1. 候选生成器（DEV-0058，`src/w2/candidates.py`）

```python
@dataclass(frozen=True)
class Candidate:
    name: str            # 生成器名，例如 "gc_default"、"gc_iter10"、"gc_box_plus10"、"slic_s200_t50"、"watershed_pts"、"box"
    mask: np.ndarray     # (H, W) bool，原图分辨率
    ok: bool             # 生成器是否正常产出；失败时 mask 为框本身（见下）且 ok = False
    info: dict           # 参数、耗时、失败原因

@dataclass(frozen=True)
class CandidateSpec:     # configs/w2.json 的 candidates 一节；生成器按清单顺序产出，清单长度就是 K
    generators: tuple[dict, ...]   # 每项 {"type": ..., 参数…}；第一项必须是 {"type": "grabcut_default"}

def generate_candidates(image_rgb: np.ndarray, box_abs: Sequence[float], *, fg_points_abs=None, bg_points_abs=None,
                        spec: CandidateSpec) -> list[Candidate]: ...
```

- **1 号候选**：`grabcut_default`，逐位等于 W1-F（`src.w1.workbench.grabcut_mask(box_abs=…)`，迭代次数与随机数种子取工作台的常量）。测试钉住这一点。
- **生成器池**：至少实现下面几类，参数可配。
  - GrabCut 的不同迭代次数；
  - 框外扩或内缩 p%；
  - 用内缩的框作 mask 初始化；
  - 有点时，以点为确定前景、背景种子的 GrabCut；
  - 两种尺度的 SLIC：框内合并，超像素有不少于 t% 的面积在框内就并入；
  - 分水岭：有点时以点为标记；没点时，框内缩的中心区域为前景标记，框外为背景标记；
  - 框本身；
  - 最大连通域等形态学清理，作为后处理变体。
- **K 固定**：
  - 某个生成器失败（参数不合法、某类没有像素、OpenCV 报错）时，候选的 mask 用框本身，`ok = False`，原因记进 `info`。编号与 K 不变。
  - 完全相同的候选不去重，照常编号；完成总结里报告重复率。
- **确定性**：同样的输入，mask 逐位相同。
- **速度**：报告 640×480 图上每种生成器的耗时。W2-c 要在 train 上大批生成，见 §5。

## 2. 挑选图的渲染（DEV-0058，`src/w2/render.py`）

```python
def render_selection(image: PIL.Image.Image, candidates: Sequence[Candidate], *, layout: str, style: Mapping[str, Any]) -> list[PIL.Image.Image]: ...
```

- `layout = "grid"`：一张拼图。每格是原图（等比缩放）叠上一个候选：半透明填充加轮廓。编号写在格子左上角，字号足够大。可选地在第 0 格放不叠加的原图作参照，由 `style` 控制。
- `layout = "separate"`：K 张图，每张是原图叠一个候选，编号写在角上；可选地另加一张原图。
- 像素预算：拼图的长边与 E1 的图像策略一致，每张单图不超过 640×480，在 Qwen3.5 的 `PixelBudget` 下的视觉 token 数写进完成总结。
- 颜色、透明度、线宽、字号放在 `style` 里，默认值在 `configs/w2.json`；对色盲友好；渲染逐位确定。
- 挑选提示里的编号从 1 开始，与 `generate_candidates` 的顺序一致。

## 3. 回路、对照与判定（DEV-0059，`src/w2/`、`scripts/run_w2.py`、`scripts/w2_gate.py`、`configs/w2.json`、`configs/w2_splits.json`）

- **提出**：
  - 默认沿用 E1 的 refexp_box 协议：提示、关思考、贪心、批大小都照 E1 冻结的那一份。所以"提出"一步的框就是 W1-0、也就是"自己的框"的框，可以直接复用同一个运行的输出，不重算。
  - 可选的"框加点"模式另写一个提示，用 Qwen 的原生格式（`bbox_2d`、`point_2d`），作为调参选项。
- **挑选**：
  - 用适配器的 `chat_turn_batch`，不给工具，单轮。user 内容是挑选图（`render_selection` 的输出），加挑选提示：说明有 K 个编号的候选，问哪个最好地覆盖 `{expression}`，只回答编号或 `none`。
  - 思考开关与预算在配置里，至多 2,048。
  - 解析：只认一个 1..K 的整数，或 none，大小写与标点宽松；其余算解析失败。"都不对"与解析失败都交 1 号候选，分别记 `fallback = "none"` 与 `fallback = "parse_failure"`。
- **设置**（`--setting`）：

  | 设置 | 做什么 |
  |---|---|
  | `w2a` | 主方法 |
  | `random_pick` | 同一组候选里按 `sha256(salt|id)` 定的伪随机编号挑 |
  | `oracle_pick` | 用真值挑 IoU 最高的候选，并列取编号小的 |
  | `w1f` | 1 号候选 |
  | `ownbox` | 框光栅化 |
  | `baseline_e1` | 定位、计数按 E1 单轮作答；`w2a` 遇到定位、计数也走这一条 |

  - 后四种不需要挑选一步。`random_pick`、`oracle_pick`、`w1f` 与 `w2a` 共用同一份候选，由同一份提出的框生成，候选缓存一次、多处使用。
  - 候选的 mask 不存全图：存 COCO RLE，可选，用于复查。
- **记录**：每条样本记
  - 提出的框（含原文）、候选的名字与 `ok`、各候选与真值的 IoU（评分时算）；
  - 挑选的原文与思考、挑中的编号、`fallback`；
  - 最终 mask 的 RLE、计分。
  - 计算记账同 W1：每次调用一条 `Compute`。
- **划分**（`configs/w2_splits.json`，由 `src/w2/data.py` 生成与核对）：
  - `devtune800` = W1 的 devtune 300 加 devcheck 500，原样复用，可以随便用。
  - **`devcheck2`**：新的 500 句，只用于判定。
    - 从 RefCOCO（UNC）val 的全体句子（`refcoco_unc_val`，10,634 句）里，**按图像**排除 W1 用过的 800 句所在的图，再按 E1 的去重规则排除开发子集与 D20 探针图。
    - 然后按 `sha256(salt|purpose|sent_id)` 升序取前 500 句。salt `w2_dev_v1`，purpose `w2_devcheck`。
    - 清单与 sha256 写进配置。
  - test：同 W1，即 E1 的 poly 子集，每个划分 1,000 句。判定通过之后才跑。
  - **冻结保护**：未冻结的配置拒跑 `devcheck2` 与 test，同 D108。
- **判定**（`scripts/w2_gate.py`，规则写死，同 `w1_gate.py` 的风格）：
  - 输入：一个主配置的运行目录与 `w1f` 的运行目录。
  - 规则：两者 cIoU 之差，配对 bootstrap **97.5%** CI 的下界 > 0 才算通过。统计量是比值之和（同 W1），重抽样次数与种子在脚本里写死。
  - 判定点核对：9B、`devcheck2`、500 句、非 tiny、不加 `--limit`、配置已冻结。
  - 输出 `gate.json`、`gate.md`。两个主配置各跑一次，"至少一个通过"由管理员在门报告里合起来写。

## 4. W1 的错误分析（DEV-0057，只用 CPU）

见 `prompts/dev/0057-w1-error-analysis.md`。

- W1 的工作台逐位确定，所以可以按记录里的调用序列重放每条样本，得到每一步的 mask，与真值算 IoU。
- 由此给失败样本分类。

## 5. W2-c（训练挑选，D113 第二节第 4 条；DEV-0060 数据、DEV-0061 训练与评测）

- **模型**：9B，判定用 9B。LoRA 与 E3 的 C3 相同：r 32、alpha 64、全部线性层、视觉塔冻结。检查点格式与 C3 相同，所以 E3 的 T 文本评测与健康检查可以原样用。
- **两类文本监督**（都是本家族原生的文本输出，不加任何新 token）：
  1. 提出：E1 refexp_box 的提示加原图 → GT 框。目标用 `src/e3/c3.box_target`（适配器的 `format_boxes`，label 写指代句），与 C3 的框监督逐字相同。
  2. 挑选：挑选图（冻结的 `render` 布局与样式）加冻结的挑选提示 → 目标就是编号的十进制文本，例如 `"3"`，关思考。标签是真值 IoU 最高的候选编号，并列取编号小的。
- **W2-c 的评测协议**：提出与挑选都用训练过的模型，都关思考。候选生成器、渲染、K 都用 W2-a 冻结的那一份。"都不对"或解析失败交 1 号。
- **数据文件**：DEV-0060 产出，`<out>/w2c_data/<shard>.jsonl`，每行一条句子：

```json
{"id": "refcoco_unc/train/123", "source": "refcoco", "split": "train", "image": "<相对数据根的路径>", "image_size": [W, H],
 "expression": "...", "gt_box": [x0, y0, x1, y1], "gt_rle": {...},
 "proposal": {"box": [x0, y0, x1, y1] | null, "failure": null | "<失败码>", "raw_text": "..."},
 "candidates": [{"index": 1, "name": "gc_default", "ok": true, "rle": {...}, "iou": 0.71}, ...] | null,
 "label": 3 | null,
 "spec_sha256": "<候选清单的 sha256>", "w2_config_sha256": "<冻结的 configs/w2.json 的 sha256>"}
```

  - `candidates` 已按 W2 的规则处理：1 号失败时为空 mask，其余失败的用框。
  - `proposal.failure` 不为空时，`candidates` 与 `label` 都为 null。这条只用于"提出"的监督，不用于"挑选"。
- **去重**（硬约束 7）：按 COCO 图像 id，排除 W2 用到的全部评测图：devtune800、devcheck2、5 个 test 子集；并沿用 E3 训练池的剔除清单。核对结果写进数据目录的 `manifest.json`。
- **规模**：由服务器上候选生成的耗时定，每句约 4.5 s 的 CPU 时间。默认取 16,000 句，按 `hash_rank(salt='w2c_train_v1')` 抽样。
