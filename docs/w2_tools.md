# W2 候选生成器与挑选图（DEV-0058；D113；接口约定见 `docs/w2_plan.md` §1、§2）

本文件写 W2-a 第二步（候选生成）与第三步（挑选图）用到的全部工具：算法、参数、失败时的处理、渲染样式、尺寸与视觉 token 数、耗时。
准备放进论文附录。代码是 `src/w2/candidates.py` 与 `src/w2/render.py`；下面写成 "`常量名` = 数值" 的固定参数，
`tests/test_w2_candidates.py`、`tests/test_w2_render.py` 会与代码逐个核对。

## 1. 总则

- **原则（D113）**：LLM 只判断是哪个物体、在候选里挑一个；逐像素的拟合全部交给工具。
  - 工具只用不含学习的算法：numpy、scipy、OpenCV、scikit-image；不用任何学习型视觉模型或预训练权重，不用 OpenCV 的 DNN 模块。
  - 底层实现与 W1 工作台（`docs/w1_tools.md`）共用：GrabCut、彩色梯度、点的圆盘、形态学、连通域都直接调用 `src/w1/workbench.py` 的同一批函数。
- **输入**：原图（RGB，uint8）、LLM 提出的框（原图绝对像素 xyxy，与 W1-F 拿到的框相同），可选的前景点与背景点（绝对像素 (x, y)）。
  - 框必须有限且 x1 ≥ x0、y1 ≥ y0（E1 的解析器保证），否则是调用方的错，直接报错。
  - 框可以部分落在图外：光栅化时图外的部分忽略。
  - 落在图外的点丢掉（`info["dropped_points"]` 记个数）；落在框外、图内的点照用。
- **输出**：K 个候选 `Candidate(name, mask, ok, info)`，按清单顺序编号 1..K。
  - mask 是原图分辨率的 bool 数组。
  - `info` 记：编号、类型、补全后的参数、后处理、是否失败与原因、耗时（`time_s`）、面积，以及各生成器自己的读数（裁剪区域、超像素个数、标记来源等）。
- **1 号候选就是 W1-F**：`grabcut_default` 与 `src/w1/baselines.w1f_episode` 走同一条调用
  `src.w1.workbench.grabcut_mask(image_rgb, box_abs=[float(v) for v in box], iterations=GRABCUT_ITERATIONS)`，逐位相同（测试钉住）。
  "都不对"与解析失败时交的就是它。
- **K 固定，失败时用框代替**：
  - 失败指生成器跑不起来：GrabCut 的初始化里某一类没有像素、分水岭的标记为空、SLIC 的框里没有像素或裁剪区域太小、`grabcut_points` 没有点、OpenCV 报错。
  - 失败时候选的 mask 用框本身（与"自己的框"同一个光栅化，见 `box`），`ok = False`，原因写进 `info["error"]`。编号与 K 不变。
  - 生成器正常跑完、结果是空 mask，不算失败，照实给出（`info["area"]` = 0）。这样只要 GrabCut 没有报错，1 号候选就与 W1-F 逐位相同。
  - 与 W1-F 唯一的不同：GrabCut 报错时（例如框盖满全图），W1-F 记 `tool_error`、按空 mask 计分，1 号候选则是框本身、`ok = False`。
- **不去重**：完全相同的候选照常编号。`duplicate_rate` 统计 K 个候选里与编号更小的某个候选逐位相同的比例。
- **共用计算**：同一次调用里，类型与参数相同、只差 `post` 或 `name` 的生成器只算一次基础结果（`info["reused"]` = true，耗时只计后处理）。
  分水岭的彩色梯度每次调用也只算一次。
- **确定性**：同样的输入逐位得到同样的 mask（测试钉住，中途打乱 OpenCV 的全局随机数也一样）。
  - GrabCut 每次调用前由 `grabcut_mask` 把 OpenCV 的随机数种子设成 `GRABCUT_RNG_SEED` = 0（它用 k-means 初始化 GMM）；
  - SLIC（scikit-image，网格初始化）、分水岭、连通域（`scipy.ndimage.label`，行优先扫描）、形态学都是确定的算法。
  - 耗时 `time_s` 不在确定性之内。

## 2. 清单的写法（`CandidateSpec`）

`configs/w2.json` 的 `candidates` 一节是 `{"generators": [...]}`，每项是一个对象：

| 键 | 说明 |
|---|---|
| `type` | 生成器类型，见第 3 节。 |
| 各类型的参数 | 见第 3 节各表；不写的取缺省值。未知的参数、超出范围的值都报错。 |
| `name` | 候选名，可选。不写时按类型与参数自动生成（与缺省值相同的参数不进名字），例如 `gc_box_plus10`、`slic_s30_t50`、`ws_pts`。 |
| `post` | 后处理，可选，按顺序执行，见第 3.7 节。名字后面接 `+lcc`、`+fill`、`+open3` 之类。 |

- 第一项必须恰为 `{"type": "grabcut_default"}`（1 号候选 = W1-F），不能带别的键。
- 候选名必须两两不同：同一个生成器写两遍没有意义；确需如此请给不同的 `name`。
- `CandidateSpec.digest()` 是补全缺省值之后的清单的 sha256：写法不同、含义相同的清单得到同一个值，用于候选缓存与运行身份。

## 3. 生成器

记号：框 (x0, y0, x1, y1)，宽 bw = x1 − x0，高 bh = y1 − y0。"框每边外扩 p%" 指左右两边各移动 p% · bw、上下两边各移动 p% · bh，
负数为内缩；不夹到图内（光栅化时图外的部分自然忽略）。`box_pct` 的范围是 [`BOX_PCT_MIN` = -45, `BOX_PCT_MAX` = 100]。
框的光栅化一律用 `src/executors/raster.box_to_mask`：像素中心落在 (lo, hi] 内记 1。

### 3.1 `grabcut_default`

W1-F 本身。没有参数。

- **算法**：GrabCut（Rother, Kolmogorov, Blake 2004；`cv2.grabCut`，`GC_INIT_WITH_MASK`）。框内记"可能前景"，框外记"确定背景"，
  迭代 `GRABCUT_ITERATIONS` = 5 次；与 W1 工作台的 `grabcut` 工具、W1-F 是同一个函数。
- **失败**：框里没有像素（退化的框）或框外没有像素（框盖满全图）；OpenCV 报错。

### 3.2 `grabcut`

参数不同、初始化不同的 GrabCut。

| 参数 | 类型 | 范围 | 缺省 | 说明 |
|---|---|---|---|---|
| `iterations` | integer | 1–20 | 5 | GrabCut 的迭代次数。 |
| `box_pct` | number | −45–100 | 0 | 框每边外扩的百分比（负数内缩）。 |
| `init` | `rect` / `mask` | — | `rect` | 初始化方式，见下。 |

- **`init = "rect"`**：调整后的框内记"可能前景"，框外记"确定背景"（同 W1-F）。框外扩时，被原框切掉的物体部分还能长回来。
- **`init = "mask"`**：调整后的框光栅化之后作为 GrabCut 的初始 mask：mask 内记"可能前景"；mask 外、但在它的外接框每边外扩
  `GRABCUT_MASK_MARGIN` = 0.2（按框宽、框高）以内的记"可能背景"；更远处记"确定背景"（`grabcut_mask` 的 `init_mask` 路径）。
  框的边界因此是"软"的：算法可以把前景长出框外一圈，也可以缩回框内。
- **名字**：`gc` 或 `gc_soft`（`init = "mask"`），接 `_box_plus10` / `_box_minus10`、`_iter10`；参数全是缺省值时是 `gc_rect`（与 `gc_default` 结果相同）。
- **失败**：同 `grabcut_default`。

### 3.3 `grabcut_points`

以 LLM 给的点为确定前景、确定背景种子的 GrabCut。只在"框加点"的提出模式下有意义。

| 参数 | 类型 | 范围 | 缺省 | 说明 |
|---|---|---|---|---|
| `iterations` | integer | 1–20 | 5 | 同 `grabcut`。 |
| `box_pct` | number | −45–100 | 0 | 同 `grabcut`。 |
| `init` | `rect` / `mask` | — | `rect` | 同 `grabcut`。 |
| `seed_radius_pct` | number | 0–50 | 4 | 种子圆盘的半径：框短边的这个百分比，至少 `SEED_RADIUS_MIN_PX` = 2 px。 |

- **算法**：先按 `init` 与 `box_pct` 初始化（同 `grabcut`），再把每个前景点周围的圆盘记"确定前景"、每个背景点周围的圆盘记"确定背景"
  （背景种子最后写，与前景种子重叠处算背景），然后照常迭代。圆盘与 W1 工作台的 `draw_points` 同一个函数（`point_disks`：像素中心到点的距离
  ≤ 半径，外加点所在的那个像素）。落在框外的前景点照用：这时 GrabCut 可以把框外的那部分也算作前景。
- **名字**：`gc_pts`（`gc_pts_soft`），后缀同 `grabcut`，半径不是缺省值时接 `_r8` 之类。
- **失败**：没有任何落在图内的点（`info["error"]` 写"没有落在图内的点"）；以及 `grabcut` 的各种失败。

### 3.4 `slic`

框内合并的 SLIC 超像素。

| 参数 | 类型 | 范围 | 缺省 | 说明 |
|---|---|---|---|---|
| `n_segments` | integer | 4–2000 | 50 | 裁剪区域里的超像素个数（约数；scikit-image 的 `n_segments`）。 |
| `threshold_pct` | number | 1–100 | 50 | 超像素有不少于这个百分比的像素落在框内，就并入。 |
| `compactness` | number | 0.01–100 | 10 | SLIC 的紧致度（CIELAB 颜色距离与空间距离的权衡）。 |
| `context_pct` | number | 0–200 | 25 | 裁剪区域 = 框每边外扩这个百分比，夹到图内。 |

- **算法**：
  1. 从原图裁出裁剪区域（左上取 floor、右下取 ceil）；
  2. 在裁剪区域上跑 SLIC（Achanta et al. 2012；`skimage.segmentation.slic`，CIELAB，`sigma` = `SLIC_SIGMA` = 1，`start_label=1`，强制连通）；
  3. 超像素 s 并入，当且仅当 |s ∩ 框| / |s| ≥ `threshold_pct` / 100；
  4. 输出是并入的超像素之并（裁剪区域之外全是 0）。
- 在裁剪区域上跑、而不是整张图：超像素的尺度随物体大小变化。`n_segments` = 30 时每个超像素约占框面积的 7.5%，100 时约 2.3%
  （缺省的裁剪区域是框面积的 2.25 倍）。
- **名字**：`slic_s{n_segments}_t{threshold_pct}`，紧致度、裁剪不是缺省值时接 `_c20`、`_ctx50`。
- **失败**：框里没有像素；裁剪区域窄于 2 px。

### 3.5 `watershed`

以标记为种子的分水岭。

| 参数 | 类型 | 范围 | 缺省 | 说明 |
|---|---|---|---|---|
| `use_points` | boolean | — | true | 给了点就用点作标记。 |
| `fg_shrink_pct` | number | 0–49 | 30 | 没有前景点时，前景标记 = 框每边内缩这个百分比后的中心区域。 |
| `bg_margin_pct` | number | 0–100 | 0 | 背景标记 = 框每边外扩这个百分比之外的全部像素。 |
| `seed_radius_pct` | number | 0–50 | 4 | 点标记圆盘的半径，同 `grabcut_points`。 |

- **算法**：在整张图的彩色梯度上做标记分水岭（Meyer 1994；`skimage.segmentation.watershed`）。
  - 彩色梯度：逐通道 Sobel（图像先归一化到 [0, 1]），三通道平方和开方（工作台的 `color_gradient`）。
  - 前景标记：`use_points` 且有前景点时，是前景点周围的圆盘；否则是框每边内缩 `fg_shrink_pct`% 的中心区域
    （缺省 30%：框中心 40% × 40% 的区域）。`info["fg_marker"]` 记 `points` 或 `center`。
  - 背景标记：框每边外扩 `bg_margin_pct`% 之外的全部像素；`use_points` 时再加上背景点周围的圆盘。前景、背景标记重叠处算前景（例如前景点落在框外）。
  - 输出是从前景标记长出来的区域。
- **名字**：`ws_pts`（`use_points` = true）或 `ws_center`，非缺省参数接 `_shrink20`、`_margin10`、`_r8`。
  在只给框的提出模式下，`ws_pts` 与 `ws_center` 结果相同。
- **失败**：前景标记为空（框太小，内缩之后没有像素）；背景标记为空（框盖满全图）。

### 3.6 `box`

框本身，没有参数。光栅化同 `box_to_mask`；这与"自己的框"对照（`scripts/summarize_e1_posthoc.own_box_record`：四个角当多边形，经 pycocotools
光栅化）逐位相同（测试用随机框钉住），所以挑中 `box` 就等于交"自己的框"。从不失败；退化的框给空 mask。

### 3.7 后处理 `post`

任何一项都可以带 `post`：按顺序执行的列表。

| 写法 | 名字后缀 | 算法 |
|---|---|---|
| `largest_cc` | `+lcc` | 只留面积最大的 8 连通域（同面积取扫描顺序在前的；工作台 `components` 的同一个排序）。空 mask 原样。 |
| `fill_holes` | `+fill` | `scipy.ndimage.binary_fill_holes`。 |
| `open:r`、`close:r`、`dilate:r`、`erode:r` | `+open3` 等 | 半径 r 的圆盘结构元（1 ≤ r ≤ `MORPH_RADIUS_MAX` = 50），工作台 `morph` 的同一个实现。 |

基础结果失败时，后处理不执行，候选照失败处理（框本身，`ok = False`）。

## 4. 默认清单 `DEFAULT_SPEC`（调参的起点）

K = 8。DEV-0059 原样抄进 `configs/w2.json`（每项的参数都写全）：

```json
{"generators": [
  {"type": "grabcut_default"},
  {"type": "grabcut", "iterations": 5, "box_pct": 10.0, "init": "rect"},
  {"type": "grabcut", "iterations": 5, "box_pct": -10.0, "init": "mask"},
  {"type": "grabcut_default", "post": ["largest_cc", "fill_holes"]},
  {"type": "slic", "n_segments": 30, "threshold_pct": 50.0, "compactness": 10.0, "context_pct": 25.0},
  {"type": "slic", "n_segments": 100, "threshold_pct": 50.0, "compactness": 10.0, "context_pct": 25.0},
  {"type": "watershed", "use_points": true, "fg_shrink_pct": 30.0, "bg_margin_pct": 0.0, "seed_radius_pct": 4.0},
  {"type": "box"}
]}
```

| 编号 | 名字 | 为什么在清单里 |
|---:|---|---|
| 1 | `gc_default` | W1-F 本身（约定）。"都不对"与解析失败时交它，所以 W2-a 的下限就是 W1-F。 |
| 2 | `gc_box_plus10` | 框偏紧时：W1-F 把框外记成确定背景，被框切掉的部分再也长不回来；每边外扩 10% 让它能长回来。 |
| 3 | `gc_soft_box_minus10` | 框偏松或边界拿不准时：从内缩 10% 的核心出发，框外一圈只是"可能背景"，前景可以缩进框内、也可以长出框外约 6%（软边界）。与 2 号的失败方式互补。 |
| 4 | `gc_default+lcc+fill` | GrabCut 常在框里留下同色的零碎小块和孔；指代表达式指的是一个物体，只留最大连通域再补孔。复用 1 号的计算，几乎不花时间。 |
| 5 | `slic_s30_t50` | 不靠颜色模型：前景、背景颜色接近时 GrabCut 会失败，框内合并的超像素沿局部边缘贴住框的边界。粗尺度。 |
| 6 | `slic_s100_t50` | 同上，细尺度（D113 的例子"两种尺度的 SLIC"）。 |
| 7 | `ws_pts` | 靠边缘而不是颜色模型：从框中心长到最强的边缘为止，常给出比 GrabCut 更紧的物体轮廓。在"框加点"模式下自动改用 LLM 的点作标记。 |
| 8 | `box` | 框本身 = "自己的框"对照。物体本身接近矩形、或者贴边算法都跑偏时，这是最好的一个。 |

- 多样性优先：三类算法（颜色模型的 GrabCut、超像素、边缘的分水岭）加框本身，GrabCut 的三个变体分别针对框偏紧、偏松、碎块三种失败。
- 开发机上的粗略读数（没有真值，只看候选彼此有多不同；第 6 节的 4 张真实照片 × 3 个框 = 12 对）：
  - 8 个候选两两之间的 IoU 平均 0.56；逐位相同的重复率平均 2.1%（最多 1/8）；
  - 最接近的一对是 `slic_s100_t50` 与 `box`（IoU 0.83），其次 `gc_default` 与 `gc_default+lcc+fill`（0.88，同一个结果的清理）；
  - 12 对里有 1 对 GrabCut 正常跑完却给出空 mask（W1-F 在这种样本上是 0 分）。
  - 这些只说明候选彼此不同，不说明哪个好；好坏要在 devtune800 上用真值看"最优候选"。
- 没放进默认清单的：
  - `grabcut` 的 `iterations` = 10：与 1 号很接近（同一组 12 对上，去掉空 mask 的那对，与 1 号的 IoU 约 0.96），多样性低，又慢；
  - `grabcut_points`：默认的提出模式只给框。改用"框加点"模式时，建议用 `{"type": "grabcut_points"}` 换掉 6 号（`slic_s100_t50`）。
- 网格布局加上原图参照，K = 8 正好排满 3 × 3。
- 这是起点，不是结论：K 与清单在开发-调参集（devtune800）上按"最优候选"（候选池的上限）与"随机挑选"两条线来定，然后冻结（D113 第二节第 3 条）。

## 5. 挑选图的渲染（`src/w2/render.py`）

`render_selection(image, candidates, layout=, style=)` 返回交给 LLM 的图（PIL，RGB）。编号从 1 开始，与 `generate_candidates` 的顺序一致。

### 5.1 布局

- **`grid`**：一张拼图。
  - 格子按行优先排：[参照,] 1, 2, …, K。参照是不叠加的原图（`reference` = true 时放在第 0 格）。
  - 列数缺省 ceil(√格数)，行数 ceil(格数 / 列数)：K = 8 加参照是 3 × 3；空着的格子填间隔色。
  - 格子尺寸：等比缩放原图，使拼图的长边（含格子之间的间隔）不超过 `grid_long_side` 且尽量接近。
    640×480 的原图、3 × 3：格子 210×158，拼图 638×482。
- **`separate`**：[参照,] 然后 K 张图，每张是原图叠一个候选。
  - 每张不超过 `separate_max_wh`：横图（宽 ≥ 高）640×480，竖图 480×640；等比缩小，不放大。COCO 的图长边都不超过 640，通常原样。
- **建议的默认布局** `DEFAULT_LAYOUT` = `separate`，理由见 5.3。最后由 `configs/w2.json`（DEV-0059）定，在调参集上可以换。

### 5.2 叠加与编号

- 先把原图与 mask 缩到格子（单图）的大小：缩小用 `cv2.INTER_AREA`，放大用 `cv2.INTER_LINEAR`；mask 按缩放后的面积占比 ≥ 0.5 取 1。
  然后在这个分辨率上画填充、轮廓与编号，所以线宽与字号就是最终图上的像素。
- **填充**：mask 内的像素逐通道取 (原值 · (1000 − a) + 颜色 · a + 500) // 1000，a = round(`fill_alpha` · 1000)：整数运算，逐位确定。
  填充够淡，物体本身的纹理与边缘透过填充仍然看得见。
- **轮廓**：mask 内侧一圈（`src/executors/overlay.mask_contour`：mask 减去 3×3 腐蚀 `contour_px` 次）涂成实色。
- **描边**：mask 外侧一圈（3×3 膨胀 `halo_px` 次减去 mask）涂成深色，浅色背景上也看得清轮廓。
- **编号**：左上角的黑底白字小方块，OpenCV 的 Hershey 矢量字体（`FONT_HERSHEY_SIMPLEX`，抗锯齿），不依赖 FreeType，跨机器一致（同 W1 工作台）。
  字高（数字的像素高度）是 `grid_label_px` 或 `separate_label_px`，但不超过格子短边的 1/4；线宽 round(字高 / 9)，至少 1。
- **颜色**：Okabe–Ito 色盲友好调色板（Okabe & Ito 2008）：蓝色填充、黄色轮廓。蓝–黄对比在红绿色盲（protan、deutan）下保留。

默认样式 `DEFAULT_STYLE`（DEV-0059 抄进 `configs/w2.json` 的 `render.style`；`style` 可以只给一部分键，其余取默认，未知的键报错）：

| 键 | 默认值 | 说明 |
|---|---|---|
| `fill_rgb` | `[0, 114, 178]` | 填充色（Okabe–Ito 蓝）。 |
| `fill_alpha` | `0.4` | 填充的不透明度（0–1）。 |
| `contour_rgb` | `[240, 228, 66]` | 内侧轮廓色（Okabe–Ito 黄）。 |
| `contour_px` | `2` | 内侧轮廓宽度（px）；0 = 不画。 |
| `halo_rgb` | `[0, 0, 0]` | 外侧描边色。 |
| `halo_px` | `1` | 外侧描边宽度（px）；0 = 不画。 |
| `label_text_rgb` | `[255, 255, 255]` | 编号的字色。 |
| `label_bg_rgb` | `[0, 0, 0]` | 编号的底色。 |
| `grid_label_px` | `22` | 网格每格的编号字高（px）。 |
| `separate_label_px` | `32` | 单图的编号字高（px）。 |
| `label_pad_px` | `4` | 编号四周的留白（px）。 |
| `reference` | `true` | 是否放一张不叠加的原图作参照。 |
| `reference_label` | `''` | 参照上写的字；空 = 不写。 |
| `grid_long_side` | `640` | 拼图的长边上限（px）。 |
| `grid_cols` | `0` | 列数；0 = 自动。 |
| `gutter_px` | `4` | 格子之间的间隔（px）。 |
| `gutter_rgb` | `[255, 255, 255]` | 间隔与空格子的颜色。 |
| `separate_max_wh` | `[640, 480]` | 单图的上限 [宽, 高]（竖图对调）。 |

### 5.3 尺寸与视觉 token 数（Qwen3.5，`PixelBudget(300)`）

用适配器自己的图像处理现算（`render.visual_tokens`：`build_inputs` → 图像处理器 → 每张图在 input_ids 里的视觉 token 段长度；
开发机上用 Qwen3.5 的 tiny 适配器，即 0.8B 的图像处理器；它与 9B 的相同：patch 16、merge 2，每个视觉 token 对应 32×32 px，
见 `docs/model_notes.md` 的 Qwen3.5 对照表），
K = 8，默认样式（含参照）：

| 原图 | grid：尺寸 → token | separate：每张尺寸 → token | separate 合计（9 张） |
|---|---|---|---:|
| 640×480 | 638×482 → 300 | 640×480 → 300 | 2,700 |
| 640×427 | 638×428 → 330 | 640×427 → 330 | 2,970 |
| 480×640 | 482×638 → 300 | 480×640 → 300 | 2,700 |
| 500×375 | 638×482 → 300 | 500×375 → 300 | 2,700 |
| 640×640 | 638×638 → 289 | 480×480 → 324 | 2,916 |

- `PixelBudget(300)` 的 `min_pixels = max_pixels`：处理器把**任何**一张图缩放到约 300 个 token（smart resize，保持长宽比）。
  所以拼图不论画多大，整张都只有约 300 个 token，3 × 3 的每格约 33 个（约 6.6 × 5 个 token，即每格在视觉编码器里只有约 210×160 px）；
  separate 的每个候选各有约 300 个，与 E1 看原图的分辨率相同，代价是 9 倍的视觉 token。
- 由此建议 `separate` 作默认：候选之间的差别常常只在物体边界附近的一窄条上，3 × 3 拼图里每格的边长只有原图的 1/3，很可能分不出来。
  `grid` 是便宜的对照（每条样本省约 2,400 个视觉 token）。两者在开发-调参集上比，按计算记账报告。
- 作参考，`PixelBudget(1024)` 下 640×480 的原图是 1,036 个 token。

### 5.4 确定性

同样的输入逐位得到同样的图（测试钉住）：缩放、混合、腐蚀膨胀、矢量字体都是确定的。不同的 OpenCV 版本之间，缩放与抗锯齿可能差几个灰度级
（开发机 OpenCV 5.0.0，服务器以自己的环境为准）；同一个运行里的所有图都在同一个环境里生成。

## 6. 耗时（CPU）

测量：`python -m src.w2.candidates --bench <图目录>`（`BENCH_SPEC`：每种类型至少一项，外加几个常见变体；
每张图配三个框 `BENCH_BOXES`：居中的中框（图面积的 25%）、小框（约 9%）、大框（约 72%），相对原图宽高给出；`grabcut_points` 与
`ws_pts` 用框中心一个前景点、框左上角外侧一个背景点）。单进程、单次、不并行；每个候选的耗时是 `info["time_s"]`，共用的基础计算只计一次。
服务器上用同一条命令在 COCO 的图上复测，W2-c 按那个数排。

**机器**：开发机 Apple M4（10 核：4 性能核 + 6 能效核），macOS 15.7.3，Python 3.12.14，OpenCV 5.0.0（`cv2.getNumThreads()` = 10），
scikit-image 0.24.0，numpy 2.5.3；2026-10-22 实测。

**图**：
- 夹具图：`tests/fixtures/e2/root/nyu_depth_train/sync/*/rgb_*.jpg`，10 张 640×480（合成的平滑色块，不是真实照片），共 30 个图–框对；
- 真实照片：scikit-image 自带的 astronaut、chelsea、coffee、rocket 四张，缩放到 640×480，共 12 个图–框对（作对照：夹具图不是真实照片）。

单个候选的耗时（秒；中位数 / 均值 / 最大）：

| 候选 | 夹具图（n = 30） | 真实照片（n = 12） |
|---|---:|---:|
| `gc_default` | 0.719 / 1.552 / 5.02 | 0.772 / 1.129 / 3.50 |
| `gc_box_plus10` | 1.369 / 1.287 / 2.72 | 1.221 / 1.452 / 4.02 |
| `gc_soft_box_minus10` | 0.901 / 1.615 / 6.76 | 1.010 / 1.389 / 3.59 |
| `gc_iter10` | 1.090 / 2.594 / 8.75 | 1.263 / 2.005 / 6.65 |
| `gc_iter1` | 0.438 / 0.534 / 1.69 | 0.297 / 0.368 / 1.08 |
| `gc_pts` | 0.885 / 1.546 / 5.00 | 0.769 / 1.153 / 3.61 |
| `gc_default+lcc+fill`（复用 1 号，只计后处理） | 0.006 / 0.006 / 0.01 | 0.006 / 0.006 / 0.01 |
| `slic_s30_t50` | 0.067 / 0.070 / 0.14 | 0.061 / 0.069 / 0.13 |
| `slic_s100_t50` | 0.075 / 0.081 / 0.17 | 0.074 / 0.082 / 0.16 |
| `ws_center`（含彩色梯度） | 0.072 / 0.073 / 0.09 | 0.068 / 0.067 / 0.08 |
| `ws_pts`（含彩色梯度） | 0.060 / 0.061 / 0.11 | 0.056 / 0.054 / 0.07 |
| `box` | < 0.001 | < 0.001 |
| **`DEFAULT_SPEC` 整份（8 个候选，墙钟）** | **3.38 / 4.71 / 14.41** | **3.40 / 4.23 / 10.62** |

- 时间几乎全花在 GrabCut 上（`DEFAULT_SPEC` 的三次 GrabCut 占九成左右）。GrabCut 的耗时随框的大小与图的内容变化很大：同一张图上，大框（图面积的 72%）可以比中框慢 4 倍以上，也可能更快；小框最快。
  超像素、分水岭、后处理与框本身合起来每张图约 0.2 s。
- **W2-c 的规模估算**：候选取决于每句话的框，所以按句子算。`DEFAULT_SPEC` 每句约 3.4 s（中位数）、4–5 s（均值），单进程。
  RefCOCO（UNC）train 约 12 万句，合计约 41–54 万 CPU 秒，即约 115–150 CPU 小时；按句子分给 32 个进程并行约 3.5–4.7 小时，
  64 个进程约 2–2.5 小时。
  服务器 CPU 的单核通常比 M4 慢，应先在服务器上用 `--bench` 复测再排。
- 要更快：GrabCut 的变体是成本的大头。在调参集上若发现某个 GrabCut 变体很少成为最优候选，先换掉它；`iterations` = 1 的变体只要约 0.3–0.4 s。
