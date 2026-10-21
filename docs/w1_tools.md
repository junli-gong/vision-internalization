# W1 工作台工具清单（DEV-0050；D103 §二.2；接口约定见 `docs/w1_plan.md` §2）

本文件列出 W1 工作台的全部工具：名字、参数、返回、所属的组、算法与固定参数。准备放进论文附录。
代码是 `src/w1/workbench.py`（工具）与 `src/w1/sandbox.py`（`run_python` 的沙箱）。
文中"给模型的原文"就是 `Workbench.specs()` 交给模型的说明，逐字照录。下面写成 "`常量名` = 数值" 的固定参数，
`tests/test_w1_workbench.py` 会与代码逐个核对。

## 1. 总则

- **原则（D103）**：语义判断由 LLM 做，底层拟合交给不含学习的算法。
  - 只用 numpy、scipy、OpenCV、scikit-image 里不含学习的函数，不用任何学习型视觉模型或预训练权重，也不用 OpenCV 的 DNN 模块。
  - 只用 OpenCV 4.10 与 5.x 都有的 API。
- **与任务无关**：所有任务用同一套工具；工具说明只写工具做什么，不写任何任务的解法（测试钉住说明里不出现任务词）。
- **坐标**：
  - 所有坐标都是整张原图的 0–1000 相对坐标，x 在前，与 Qwen 的原生框格式一致。
  - 换算成绝对像素：x_abs = x / 1000 · W，y_abs = y / 1000 · H，一律走 `src/executors/geometry.norm1000_to_abs`。
  - 放大视图里看到的位置也要换回原图坐标，`zoom` 与 `show` 的文字给出换算式。
  - 开了 `view_frames`（默认关）时，带坐标的工具还可以按某张返回的视图自己的 0–1000 给坐标，由工作台换回原图坐标，见第 9 节。
  - 长度（半径）以原图像素计。原图尺寸写在每条 mask 统计里（"of the W x H image"）。
- **mask**：
  - 原图分辨率的二值数组，按名字存放。名字由字母、数字、`_`、`-` 组成，字母开头，至多 32 个字符；同时至多 `MAX_MASKS` = 32 个。
  - 画图类工具都有 `mode`：
    - `new`：新建或覆盖；
    - `union`、`subtract`、`intersect`：与同名的已有 mask 取并、差、交。
  - 写 mask 的工具都返回统计：面积（像素与占比）、外接框（0–1000）、8 连通域个数。
- **光栅化**（与 GT 同一套规则）：
  - 框：`src/executors/raster.box_to_mask`，像素中心落在 (lo, hi] 内记 1；
  - 多边形与样条：`polygons_to_mask`，即 pycocotools，与 COCO/RefCOCO 的 GT 相同。
- **返回**：每次调用返回 `ToolResult(text, images, error)`。
  - 图都是 RGB 的 PIL 图：叠加图与原图同尺寸；放大视图长边 `ZOOM_LONG_SIDE` = 640 px。
  - **哪些工具返回图**（管理员裁定，DEV-0050 审查）：把 mask 叠回原图属于"核对"，只有 check 组的 `show` 做。
    - 返回图的只有：eye 组（`zoom`、`tile_view`，原图的视图），以及 snap 组里两个"看图本身"的视图 `edges`（边缘图）与
      `superpixels`（带编号的分割图，选编号要用）。这四张图都与 mask 无关。
    - hand 组、snap 组其余工具、`run_python` 与 submit 组只返回文字；写了 mask 的，文字给出它的面积（像素与占比）和 0–1000 外接框。
    - 这样 A2（去掉 `check`）之后，模型没有任何办法把结果叠回原图再看。测试 `test_only_check_overlays_a_mask_on_the_original` 钉住。
    - 640×480 在 Qwen3.5 的 `PixelBudget(300)` 下恰为 300 个视觉 token。
  - 出错（参数不对、越界、未知的 mask 名、算法失败）不抛异常：`error` 写原因，`text` 是 "Error: " 加原因。
  - 参数解析：
    - 模板里的参数值都是字符串，按 JSON Schema 解析：数与数组按 JSON 读；只要一个数的数组也接受裸数。
    - 多余的参数与缺少的必填参数都报错。
- **日志**（`Workbench.log()`）：每次调用记一条。
  - 工具名、组、参数原文、耗时、是否出错、写了哪些 mask 及其面积；
  - 返回的图：id、尺寸、sha256；
  - `run_python` 的代码原文与沙箱摘要；提交的答案。
- **最终答案**：
  - `submit_*` 写入 `Workbench.submission`：mask（原图分辨率 bool）、框（绝对像素 xyxy）或计数（int）。
  - 再次提交会覆盖。
- **确定性**：同样的调用序列，文字、图、mask 逐位相同（测试钉住，中途打乱 OpenCV 的全局随机数也一样）。
  - GrabCut 每次调用前把 OpenCV 的随机数种子设成 `GRABCUT_RNG_SEED` = 0；
  - 连通域用 `scipy.ndimage.label`（行优先扫描）。
- **组与消融**：组是 `eye`、`hand`、`snap`、`check`、`code`、`submit`。
  - A1 去掉 `eye`；A2 去掉 `check`；A3 去掉 `snap` 与 `code`。
  - 被关掉的组不出现在 `specs()` 里；调用时返回与未知工具相同的错误（"unknown tool"）。

## 2. 总表

| 组 | 工具 |
|---|---|
| `eye`（眼） | `zoom`、`read_pixels`、`tile_view` |
| `hand`（手） | `draw_box`、`draw_points`、`draw_polygon`、`draw_spline`、`combine_masks`、`delete_mask` |
| `snap`（贴边） | `edges`、`grabcut`、`watershed`、`superpixels`、`select_superpixels`、`refine`、`morph`、`components` |
| `check`（核对） | `show` |
| `code`（编排） | `run_python` |
| `submit`（提交） | `submit_mask`、`submit_box`、`submit_count` |

共 22 个工具。全部工具说明按 JSON 计约 1.4 万字符（约 3.5–4 千 token），每一轮都在系统提示里。

## 3. eye（眼）

### `zoom`

给模型的原文：Return an enlarged view of a rectangular region of the original image (long side 640 px); the text gives the exact region covered.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `x1` | number 0–1000 | 是 | — | Left edge, 0-1000 (full image). |
| `y1` | number 0–1000 | 是 | — | Top edge, 0-1000 (full image). |
| `x2` | number 0–1000 | 是 | — | Right edge, 0-1000 (full image). |
| `y2` | number 0–1000 | 是 | — | Bottom edge, 0-1000 (full image). |

- **返回**：一张放大视图，加一段文字。
  - 文字写明视图实际覆盖的原图范围（0–1000 与像素行列）和放大倍数；
  - 还给出换算式：视图里比例位置 (u, v) 对应原图坐标 x = x1' + u·(x2' − x1')，y 同理。
- **算法**：
  - 区域换成像素后向外取整（左上取 floor、右下取 ceil，至少 1 px），从原图裁出；
  - 长边缩放到 `ZOOM_LONG_SIDE` = 640：放大用 `cv2.INTER_LINEAR`，缩小用 `cv2.INTER_AREA`。
  - 视图上不画任何标注。

### `read_pixels`

给模型的原文：Read the RGB values and luminance of the original image at given points, optionally averaged over a square window.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `points` | array of [x, y]，1–64 个 | 是 | — | Points [[x, y], ...] in 0-1000 full-image coordinates. |
| `radius` | integer 0–20 | 否 | 0 | Half-size of the square window in pixels (0 = the single pixel). |

- **返回**：文字，每点一行。
  - 内容：所在像素（列、行）、RGB、亮度；
  - `radius` > 0 时给 (2r+1)² 窗口（截在图内）的均值与标准差。
- **算法**：点所在的像素是 floor(x_abs)、floor(y_abs)，夹到图内。
  - 亮度 Y = 0.299 R + 0.587 G + 0.114 B（`LUMA_WEIGHTS`，与 OpenCV 的 RGB→GRAY 相同）。

### `tile_view`

给模型的原文：Return the original image divided into a grid of numbered cells; the text lists each cell's coordinate range.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `rows` | integer 1–8 | 是 | — | Number of rows. |
| `cols` | integer 1–8 | 是 | — | Number of columns. |

- **返回**：原图加白色网格线（黑边），每格左上角标格号（1 起，行优先），与原图同尺寸；文字列出每格的 0–1000 范围。
- **算法**：
  - 第 k 条格线在 floor(k·H/rows) 行、floor(k·W/cols) 列；
  - 格号用 OpenCV 的 Hershey 矢量字体，不依赖 FreeType，跨机器一致。

## 4. hand（手）

### `draw_box`

给模型的原文：Rasterize an axis-aligned box into a named mask.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of the mask to write (created if it does not exist). |
| `x1` | number 0–1000 | 是 | — | Left edge, 0-1000 (full image). |
| `y1` | number 0–1000 | 是 | — | Top edge, 0-1000 (full image). |
| `x2` | number 0–1000 | 是 | — | Right edge, 0-1000 (full image). |
| `y2` | number 0–1000 | 是 | — | Bottom edge, 0-1000 (full image). |
| `mode` | string new/union/subtract/intersect | 否 | new | new: create or overwrite the mask; union / subtract / intersect: combine with the existing mask of that name. |

- **返回**：文字，mask 统计。
- **算法**：`box_to_mask`，像素中心在 (x1, x2] × (y1, y2] 内。要求 x1 < x2、y1 < y2。

### `draw_points`

给模型的原文：Paint discs at points into a named mask: positive points (label 1) add discs, negative points (label 0) erase discs.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of the mask to write (created if it does not exist). |
| `points` | array of [x, y]，1–64 个 | 是 | — | Points [[x, y], ...] in 0-1000 full-image coordinates. |
| `labels` | array of integer，0–64 个，元素取 0 或 1 | 否 | — | One label per point: 1 = positive, 0 = negative (default: all 1). |
| `radius` | number 0–200 | 否 | 3 | Disc radius in pixels of the original image. |
| `mode` | string new/union/subtract/intersect | 否 | new | （同 `draw_box`） |

- **返回**：文字，正负点个数与 mask 统计。
- **算法**：
  - 圆盘 = 像素中心到点的距离 ≤ radius 的像素，外加点所在的那个像素（`radius` = 0 时只有它）。
    默认半径 `POINT_RADIUS_DEFAULT` = 3。
  - 结果 = 按 `mode` 把正点圆盘之并合进已有 mask，再擦掉负点圆盘之并。
    要从已有 mask 上擦，用 `mode` = `union`（只给负点）。
  - 正负点 mask 可作 `grabcut` 的 `fg` / `bg` 种子或 `watershed` 的标记。

### `draw_polygon`

给模型的原文：Rasterize a polygon (vertices in order) into a named mask.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of the mask to write (created if it does not exist). |
| `points` | array of [x, y]，3–64 个 | 是 | — | Vertices [[x, y], ...] in 0-1000 full-image coordinates, at least 3. |
| `mode` | string new/union/subtract/intersect | 否 | new | （同 `draw_box`） |

- **返回**：文字，mask 统计。
- **算法**：顶点换成绝对像素后交给 `polygons_to_mask`（pycocotools），与 RefCOCO GT 同一套规则。

### `draw_spline`

给模型的原文：Rasterize a closed smooth curve through the given points (centripetal Catmull-Rom spline) into a named mask.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of the mask to write (created if it does not exist). |
| `points` | array of [x, y]，3–64 个 | 是 | — | Points on the curve [[x, y], ...] in order, 0-1000 full-image coordinates, at least 3. |
| `mode` | string new/union/subtract/intersect | 否 | new | （同 `draw_box`） |

- **返回**：文字，mask 统计。
- **算法**：闭合的向心 Catmull–Rom 样条（Barry–Goldman 金字塔公式）。
  - 参数：alpha = `SPLINE_ALPHA` = 0.5；
  - 曲线经过每个控制点，向心参数化不会在控制点附近打结；
  - 每段采样 `SPLINE_SAMPLES_PER_SEGMENT` = 16 个点，连成多边形后交给 `polygons_to_mask`；
  - 相邻重合的控制点先去掉，至少要 3 个不同的点。

### `combine_masks`

给模型的原文：Combine mask 'other' into mask 'name' (new = copy, union, subtract or intersect).

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of the mask to write (created if it does not exist). |
| `other` | string | 是 | — | Name of an existing mask. |
| `mode` | string new/union/subtract/intersect | 否 | union | （同 `draw_box`；`new` 即复制） |

- **返回**：文字，mask 统计。
- **算法**：逐像素布尔运算。

### `delete_mask`

给模型的原文：Delete a named mask.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of an existing mask. |

- **返回**：文字，剩下的 mask 名。
- **用途**：结果不对时删掉重来。所有工具都从原图出发，原图永远不变。

## 5. snap（贴边）

### `edges`

给模型的原文：Detect edges in the original image (Canny); returns the edge map over a darkened copy of the image, optionally saved as a mask.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `low` | number 0–2000 | 否 | 50 | Lower hysteresis threshold. |
| `high` | number 0–2000 | 否 | 150 | Upper hysteresis threshold. |
| `out` | string | 否 | — | Optional name of a mask to save the edge pixels to. |

- **返回**：一张图（白色边缘画在亮度压到 40% 的灰度原图上，与原图同尺寸）与边缘像素数；给了 `out` 就把边缘存成 mask。
- **算法**：
  - 先做 5×5 高斯模糊（`CANNY_BLUR_KSIZE` = 5，`CANNY_BLUR_SIGMA` = 1.4），再 `cv2.Canny`（L2 梯度，孔径 3）。
  - 输入是三通道彩色图：每个像素取梯度最大的通道，亮度相同、颜色不同的边界也能检出。
  - 默认阈值 `CANNY_LOW` = 50、`CANNY_HIGH` = 150（高 = 3 × 低）。要求 `low` < `high`。

### `grabcut`

给模型的原文：Segment with GrabCut (color models plus graph cuts), initialized from a box (outside = background) or from an existing mask (inside = probable foreground; outside it but within its bounding box enlarged by 20% per side = probable background; elsewhere = background); optional seed masks fix pixels as foreground or background.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `out` | string | 是 | — | Name of the mask to write. |
| `box` | array of number，4 个，元素 0–1000 | 否 | — | Initial box [x1, y1, x2, y2] in 0-1000 full-image coordinates. |
| `mask` | string | 否 | — | Name of an existing mask to initialize from (instead of a box). |
| `fg` | string | 否 | — | Optional mask of pixels that must be foreground. |
| `bg` | string | 否 | — | Optional mask of pixels that must be background. |
| `iterations` | integer 1–20 | 否 | 5 | Number of iterations. |

- **返回**：文字，mask 统计。
- **算法**：GrabCut（Rother, Kolmogorov, Blake 2004），即 `cv2.grabCut`，模式 `GC_INIT_WITH_MASK`。
  - 能量：前景、背景各 5 个高斯分量的颜色 GMM，加上对比度敏感的 Potts 平滑项，用图割求解。
  - 初始化，`box` 与 `mask` 恰好给一个：
    - `box`：框内（`box_to_mask` 规则）记"可能前景"，框外记"确定背景"，即 `GC_INIT_WITH_RECT` 的语义；
    - `mask`：mask 内记"可能前景"；mask 外、但在它的外接框每边外扩 `GRABCUT_MASK_MARGIN` = 0.2（按框宽、框高）以内的记"可能背景"；
      更远处记"确定背景"。不设这个范围时前景可以长到整张图。比例暂定，在开发-调参集上定。
  - `fg` / `bg` 覆盖为"确定前景 / 确定背景"。
  - 迭代 `GRABCUT_ITERATIONS` = 5 次（默认）；每次调用前 `cv2.setRNGSeed(0)`，结果确定。
  - 前景或背景太小时 OpenCV 报错，工具把它转成 `error` 返回。
- **复用**：W1-F 固定流程直接调用同一个函数 `src.w1.workbench.grabcut_mask(image_rgb, box_abs=...)`。

### `watershed`

给模型的原文：Marker-based watershed on the color gradient of the original image: grows foreground seeds against background seeds and writes the foreground region.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `out` | string | 是 | — | Name of the mask to write. |
| `fg` | string | 是 | — | Mask of foreground seed pixels. |
| `bg` | string | 否 | — | Optional mask of background seed pixels (default: the image border). |

- **返回**：文字，mask 统计。
- **算法**：标记分水岭 `skimage.segmentation.watershed`（4 邻域，无分水线）。
  - 高程：彩色梯度幅值 = 三个通道各自 Sobel（`skimage.filters.sobel`，图像先归一到 [0, 1]）的平方和开方。
  - 标记：前景种子记 2，背景种子记 1；输出标号 2 的区域。
  - 不给 `bg` 时，用图像最外一圈像素（去掉前景种子）作背景种子。
  - 前景种子为空，或前景与背景种子重叠，都报错。

### `superpixels`

给模型的原文：Partition the original image into superpixels (SLIC); returns the image with superpixel boundaries and numeric ids.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `n_segments` | integer 2–600 | 否 | 100 | Approximate number of superpixels. |
| `compactness` | number 0.01–100 | 否 | 10 | Higher = more regular shapes, lower = follows color more closely. |

- **返回**：一张图与超像素个数 K。
  - 图：黄色边界，编号写在每个超像素离边界最远的点上，与原图同尺寸；
  - 编号是 1..K。
- **算法**：SLIC（Achanta et al. 2012），即 `skimage.segmentation.slic`。
  - 在 CIELAB 色彩空间，先做 sigma = `SLIC_SIGMA` = 1 的高斯平滑；
  - 强制连通，标号重排为连续的 1..K；
  - 默认 `SLIC_N_SEGMENTS` = 100，`SLIC_COMPACTNESS` = 10。
  - 标号图留在工作台里，下一次调用 `superpixels` 时替换。

### `select_superpixels`

给模型的原文：Write the union of chosen superpixels (from the latest superpixels call) into a named mask, chosen by id and/or by points inside them.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of the mask to write (created if it does not exist). |
| `ids` | array of integer，至多 600 个，元素 ≥ 1 | 否 | — | Superpixel ids. |
| `points` | array of [x, y]，1–64 个 | 否 | — | Points [[x, y], ...] in 0-1000 full-image coordinates; the superpixel containing each point is selected. |
| `mode` | string new/union/subtract/intersect | 否 | new | （同 `draw_box`） |

- **返回**：文字，选中的编号与 mask 统计。
- **算法**：选中的超像素 = 给出的编号，加上每个点所在像素的编号；取并集后按 `mode` 写入。
  - `ids` 与 `points` 至少给一个；
  - 还没调用过 `superpixels` 时报错。

### `refine`

给模型的原文：Snap a mask's boundary to image edges: guided filter of the mask with the original color image as guide, then threshold.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of an existing mask. |
| `out` | string | 否 | — | Name of the mask to write the result to (default: overwrite the input mask). |
| `radius` | integer 1–32 | 否 | 8 | Filter radius in pixels. |
| `eps` | number 1e-6–1 | 否 | 0.001 | Regularization; smaller follows edges more closely. |

- **返回**：文字，改变的像素数与 mask 统计。
- **算法**：彩色引导滤波（He, Sun, Tang 2010/2013），即 `src/executors/upsample.guided_filter`。
  - 引导图：原图 RGB 除以 255；被滤波的是 mask 的 0/1 浮点图；滤波后 ≥ `REFINE_THRESHOLD` = 0.5 记前景。
  - 默认半径与 eps 沿用 `src/constants.py` 的 `GUIDED_FILTER_RADIUS`、`GUIDED_FILTER_EPS`，即 E3 引导上采样的同一组值。

### `morph`

给模型的原文：Morphological operation on a mask with a disc: dilate, erode, open, close, or fill_holes.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of an existing mask. |
| `op` | string dilate/erode/open/close/fill_holes | 是 | — | Operation. |
| `radius` | integer 1–50 | 否 | 2 | Disc radius in pixels (ignored by fill_holes). |
| `out` | string | 否 | — | Name of the mask to write the result to (default: overwrite the input mask). |

- **返回**：文字，mask 统计。
- **算法**：
  - 结构元是半径 r 的圆盘：到中心距离 ≤ r 的格点，边长 2r+1；默认 `MORPH_RADIUS_DEFAULT` = 2。
  - `dilate`、`erode` 用 `cv2.dilate`、`cv2.erode`；`open`、`close` 用 `cv2.morphologyEx`。
    边界取 OpenCV 默认：图外不参与，贴着图边的 mask 不会因为图边而被腐蚀。
  - `fill_holes` 用 `scipy.ndimage.binary_fill_holes`。

### `components`

给模型的原文：List the 8-connected components of a mask (largest first) with area, box and center; optionally keep only some of them.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of an existing mask. |
| `keep` | array of integer，至多 64 个，元素 ≥ 1 | 否 | — | Ranks (1 = largest) of the components to keep. |
| `out` | string | 否 | — | Name of the mask to write the result to (default: overwrite the input mask). |

- **返回**：文字。
  - 列出连通域，至多 `COMPONENTS_LIST_MAX` = 20 个：名次、面积、外接框、质心（0–1000）；
  - 给了 `keep` 就把这些名次的连通域写进 `out`（默认覆盖输入）。
- **算法**：`scipy.ndimage.label`，8 连通。按面积从大到小排名，同面积按扫描顺序。

## 6. check（核对）

### `show`

给模型的原文：Return the original image with a named mask overlaid (fill and outline), optionally zoomed to a region, with the mask's area, box and number of components.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of an existing mask. |
| `region` | array of number，4 个，元素 0–1000 | 否 | — | Optional region [x1, y1, x2, y2] in 0-1000 full-image coordinates. |

- **返回**：叠加图与 mask 统计。
  - 不带 `region`：与原图同尺寸；
  - 带 `region`：按 `zoom` 的同一套规则裁剪放大，文字写明覆盖范围。
- **算法**：`src/executors/overlay.render_overlay`，叠加在**原图**上，不叠在任何中间结果上。
  - 填充：alpha = 1/2，第一种调色板颜色；轮廓：2 px 实色。
  - 参数取自 `src/constants.py`，与 E3 的叠加反馈同一套。
- 这是工作台里唯一把 mask 叠回原图的工具（见第 1 节"哪些工具返回图"）。

## 7. code（编排）

### `run_python`

给模型的原文：Run Python code in a sandbox with numpy, scipy, cv2, skimage, math, itertools, collections, functools, json, re and statistics (no files, network or learned models). Variables: image (H x W x 3 uint8 RGB array of the original image), masks (dict: name -> H x W bool array), H, W. Assign an H x W boolean array to result_mask to save it as a mask. Printed output is returned (first 4000 characters). Limits: 20 s, 4 GB.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `code` | string | 是 | — | Python source code. |
| `out` | string | 否 | code_result | Name of the mask to save result_mask to. |

- **返回**：
  - stdout 文字，截到 `STDOUT_LIMIT_CHARS` = 4000 字符；
  - 代码给 `result_mask` 赋了 H×W 的布尔（或 0/1）数组，就存成 mask `out`；
  - 被拒、违规、出错、超时、超内存都返回 `error`，写明是哪一种。
- **沙箱**（`src/w1/sandbox.py`）：研究诚信的防线，不是对抗性安全边界。
  - **用途**：防止模型借代码调用学习型模型、读权重、联网或读写文件；不防刻意构造的逃逸。每段代码原文都进日志。
  - **进程**：
    - 子进程 `python -I -B`，环境变量为空，不带 HF_HOME、PATH 等；
    - 工作目录是一次性临时目录；
    - 子进程在导入 numpy 之前把 BLAS/OpenMP 线程数设为 1，结果确定。
  - **限额**：
    - 墙钟 `DEFAULT_TIMEOUT_S` = 20 s，到时由父进程结束子进程；
    - 内存 `DEFAULT_MEMORY_BYTES` = 4 GB：Linux 用 `RLIMIT_AS`，另有看门狗线程每 10 ms 读一次内存占用，超了就结束进程；
    - macOS 不支持 RLIMIT_AS，只有看门狗，读的是含压缩页的 phys_footprint。
  - **数据**：输入图像与 mask、输出 mask 都经 npz 传递，`allow_pickle=False`，不用 pickle。
  - **import 白名单**：
    - 允许 numpy、scipy（子模块都允许）、cv2、skimage、math、itertools、collections、functools、json、re、statistics；
    - 其余一律拒绝，包括 torch、transformers、os、sys、subprocess、socket、pathlib、importlib、io、ctypes、pickle。
  - **屏蔽**（"严格"一类，谁调用都算违规）：
    - 学习型模型与预训练权重：`cv2.dnn` 及一切 `dnn_*`、`cv2.dnn_superres`、`cv2.face`、`cv2.text`、`cv2.ml`；
      `cv2.CascadeClassifier`；整个 `cv2.HOGDescriptor`，因为默认行人检测器是它的静态方法；`cv2.FaceDetectorYN`、`cv2.FaceRecognizerSF`；
      全部视频跟踪器 `cv2.Tracker*`；contrib 里带模型的模块（ximgproc、xfeatures2d、saliency、wechat_qrcode、mcc、barcode、xphoto 等）；
      `skimage.feature.Cascade`、`learn_gmm`、`fisher_vector`；`skimage.future`（可训练分割器）。
    - 样例数据与下载：`skimage.data`、`skimage.data_dir`、`cv2.data`（haarcascades 路径）、`cv2.samples`、`cv2.utils`、`scipy.datasets`、`scipy.misc`。
  - **屏蔽**（"仅用户代码"一类，白名单库内部照常可用）：
    - 文件与设备读写：`np.load`、`np.save*`、`np.loadtxt`、`np.fromfile`、`np.memmap` 等，`scipy.io`、`scipy.sparse.load_npz`，
      `skimage.io`，`cv2.imread*`、`cv2.imwrite*`、`cv2.FileStorage`、`cv2.VideoCapture` 等；
    - 加载本地代码：`numpy.ctypeslib`、`numpy.f2py`、`numpy.testing`、`scipy.LowLevelCallable`。
    - 白名单库内部照常可用，是因为 skimage.morphology 导入时要用 np.load 读自带的圆盘分解表（整数表，不是权重）。
  - **四层拦截**：
    1. 启动前的 AST 检查。禁用名字：open、exec、eval、compile、`__import__`、getattr、setattr、globals、vars 等，以及一切双下划线名字。
       禁用属性：以下划线开头的属性、标准库模块名（如 `cv2.os`）、帧对象属性、明确的学习型与文件读写 API 名。
    2. 受限的内建函数，加上带白名单的 `__import__`。白名单只管用户代码里的 import 语句。
       用户代码调用的 C 函数在内部发起的导入属于库内部，私有子模块照常导入；例如 `image.max()` 第一次调用时 numpy 会导入 `numpy._core._methods`（RUN-0059）。
    3. `sys.meta_path` 拦截器：学习型框架与下载器（torch、transformers、sklearn、onnxruntime、pooch 等）谁都不能加载。
       库代码可选地导入它们时只得到 ModuleNotFoundError，回退照常工作，记进 `blocked_imports`，不算违规；用户代码导入它们才算违规。
    4. 属性下毒：被屏蔽的属性换成一调用、一取属性就报违规的占位对象，别名也一并换掉。
  - **违规处理**：任何一次违规都判这次运行为 violation，返回的 mask 作废。`except` 接不住，光秃的 `except:` 接住了也照样记下。
  - **不算学习、照常允许的算法**：k-means（`cv2.kmeans`、`scipy.cluster`）、HOG 特征（`skimage.feature.hog`）、局部二值模式等。
    它们没有预训练权重，只在当前这张图上计算。

## 8. submit（提交）

### `submit_mask`

给模型的原文：Submit a named mask as the final answer.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `name` | string | 是 | — | Name of an existing mask. |

- **返回**：文字。`Workbench.submission` = mask 的副本（原图分辨率 bool）。

### `submit_box`

给模型的原文：Submit a box as the final answer.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `x1` | number 0–1000 | 是 | — | Left edge, 0-1000 (full image). |
| `y1` | number 0–1000 | 是 | — | Top edge, 0-1000 (full image). |
| `x2` | number 0–1000 | 是 | — | Right edge, 0-1000 (full image). |
| `y2` | number 0–1000 | 是 | — | Bottom edge, 0-1000 (full image). |

- **返回**：文字。`Workbench.submission.box` = 绝对像素 xyxy（浮点，`norm1000_to_abs`）。

### `submit_count`

给模型的原文：Submit a non-negative integer as the final answer.

| 参数 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `n` | integer 0–10000 | 是 | — | The count. |

- **返回**：文字。`Workbench.submission.count` = n。

## 9. 视图坐标（`view_frames`，默认关；DEV-0054）

- **用途**（RUN-0060 冒烟的诊断）：
  - 模型先放大一块区域再给框时，常把放大窗口的边界或整百的数当成框，或者把放大图里看到的位置按换算式换回原图时算错。
  - Qwen 的原生坐标是"所看那张图的 0–1000"。开了这个开关，模型可以直接按放大图自己的 0–1000 给坐标，由工作台换回原图。
  - 这是工作台的通用功能，与任务无关：所有带坐标的工具都适用。
- **开关**：`Workbench(image, *, disabled_groups=…, sandbox=…, view_frames=False)`。回路经 `configs/w1.json` 的 `workbench.view_frames` 透传（DEV-0053）。
  - **关（默认）**：`specs()`、所有返回的文字、图、mask、日志都与没有这个开关时逐位相同；传 `view` 参数按"多余的参数"报错。
    `specs()` 的 `tools_digest` = `dcc91402712dd16de304c385f64197d73a6a119b1dcaeb252fa3368b041f8a11`，与改动前相同（测试钉住）。
  - **开**：`tools_digest` = `0bdcaf73354329f7dad411d774d4eec2c32295fd2db7a81968ab3a9ddcf6e038`（测试钉住）。与关时只有两处差别：
    - 下面 10 个工具的参数表最后多一个 `view`；
    - 这 10 个工具的坐标参数，说明里"原图坐标"的提法改成同时涵盖视图的说法（管理员审查 DEV-0054 时定），原文见本节末。
      例如 "Left edge, 0-1000 (full image)." 改为 "Left edge, 0-1000 (full image, or the view given by 'view')."。
    - 工具说明本身、其余参数的说明不变。
- **视图编号**：
  - 视图 0 是原图本身，即第一条 user 消息里的图。
  - 每次成功的调用返回的每一张图都登记成一个新视图，编号从 1 起递增，不复用。登记的是这张图覆盖的原图区域，用原图 0–1000 的 `[x1, y1, x2, y2]` 表示：
    - `zoom`、带 `region` 的 `show`：像素窗口向外取整之后实际覆盖的区域，即文字里 "The view covers ..." 的那个范围（存的是不舍入的浮点值）；
    - `tile_view`、`edges`、`superpixels`、不带 `region` 的 `show`：与原图同尺寸，区域是 `[0, 0, 1000, 1000]`。
  - 返回文字的**最前面**写 "This is view k."（工具消息里文字在图之前）。
  - 出错的调用不返回图，也不登记视图、不占编号。
  - 至多 `MAX_VIEWS` = 64 个视图，编号 0–63，含视图 0。
    - 登记满之后，返回图的工具在执行之前就报错，不留下写了一半的状态（例如 `edges` 的 `out`）；其余工具照常可用。
    - 回路里计数的工具调用至多 8 次，正常碰不到这个上限。
- **`view` 参数**：
  - 带坐标参数的 10 个工具多一个可选的整数参数 `view`（0–63，默认 0），放在参数表最后：

    | 工具 | 随 `view` 换算的参数 |
    |---|---|
    | `zoom` | `x1`、`y1`、`x2`、`y2` |
    | `read_pixels` | `points` |
    | `draw_box` | `x1`、`y1`、`x2`、`y2` |
    | `draw_points` | `points` |
    | `draw_polygon` | `points` |
    | `draw_spline` | `points` |
    | `grabcut` | `box` |
    | `select_superpixels` | `points` |
    | `show` | `region` |
    | `submit_box` | `x1`、`y1`、`x2`、`y2` |

  - 其余工具没有坐标参数，也就没有 `view`。其中 `watershed` 的种子是 mask 名，不是点或框。
    测试核对：参数表里每个取值为 0–1000 的数（或这种数的数组）都在上表里，没有漏的。
  - **换算**：给了 `view` = k，这次调用里的所有坐标都按视图 k 自己的 0–1000 读（x 在前），先换成原图 0–1000，再照常走 `norm1000_to_abs` 与工具原来的算法：
    - x_full = vx1 + x · ((vx2 − vx1) / 1000)，y_full = vy1 + y · ((vy2 − vy1) / 1000)。[vx1, vy1, vx2, vy2] 是视图 k 登记的区域。
    - 这与 x_full = vx1 + x / 1000 · (vx2 − vx1) 是同一个式子，只是先算比例：覆盖整图的视图比例恰为 1.0，换算逐位是恒等（测试钉住）。
    - 结果夹到 [vx1, vx2] 与 [vy1, vy2]，只去掉浮点误差。
    - 函数是 `src.w1.workbench.view_point_to_full(region, x, y)`。工具本身的算法与固定参数一处不改。
    - `zoom` 裁出整像素窗口后用 `cv2.resize` 缩放，resize 按像素边缘对齐整个窗口，所以视图里的比例位置正对应覆盖区域里的同一比例位置。
  - **校验**：
    - 坐标仍按 0–1000 校验，在视图坐标里校验；
    - `view` 不是已有的编号时报错，返回错误文字，不抛异常。
  - 长度类参数（`read_pixels` 与 `draw_points` 的 `radius`）不受影响，仍以原图像素计。
- **返回文字**：
  - 坐标一律是原图 0–1000，与关时相同。例如 `read_pixels` 回显的点、`grabcut` 与 `submit_box` 回显的框，都是换算之后的原图坐标。
  - 给了非 0 的 `view` 时，文字前面加一句 "Coordinates given in view k, converted to full-image 0-1000 coordinates."；返回图的工具放在 "This is view m." 之后。
  - 出错时报错末尾加 " (coordinates were converted from view k to full-image 0-1000 coordinates)"，因为报错里的坐标已经换算过。
  - `zoom` 与 `show` 原来文字里的换算式保留。
- **日志**（`Workbench.log()`，只在开时多两个键，关时与改动前相同）：
  - `view`：这次调用的坐标按哪个视图读，没给 `view` 时为 null；
  - `views`：这次调用登记的视图，`[{"view": k, "region": [x1, y1, x2, y2]}]`，出错时为空。
  - `Workbench.views` 是到目前为止登记的全部区域。
- **确定性**：同样的调用序列，文字、图、mask、日志逐位相同（测试钉住，中途打乱 OpenCV 的全局随机数也一样）。
- **给模型的原文**（逐字照录）：
  - `view` 参数：`{"type": "integer", "minimum": 0, "maximum": 63, "default": 0, "description": …}`，说明是：
    Optional id of an earlier view (0 = the original image); if given, all coordinates in this call are 0-1000 relative to that view.
  - 坐标参数的说明（只在开时；关时是第 3–8 节表里的原文）。改写规则只有两条，在 `src/w1/workbench.py` 的 `_VIEW_DESC_REWRITES`：
    "0-1000 (full image)" → "0-1000 (full image, or the view given by 'view')"；
    "0-1000 full-image coordinates" → "0-1000 coordinates of the full image or of the view given by 'view'"。
    哪个坐标参数的说明里没有这两种提法，构造工作台时就报错，不会静默漏改。改写后的全部说明：

    | 参数 | 所在工具 | 开时的说明 |
    |---|---|---|
    | `x1` | `zoom`、`draw_box`、`submit_box` | Left edge, 0-1000 (full image, or the view given by 'view'). |
    | `y1` | 同上 | Top edge, 0-1000 (full image, or the view given by 'view'). |
    | `x2` | 同上 | Right edge, 0-1000 (full image, or the view given by 'view'). |
    | `y2` | 同上 | Bottom edge, 0-1000 (full image, or the view given by 'view'). |
    | `points` | `read_pixels`、`draw_points` | Points [[x, y], ...] in 0-1000 coordinates of the full image or of the view given by 'view'. |
    | `points` | `draw_polygon` | Vertices [[x, y], ...] in 0-1000 coordinates of the full image or of the view given by 'view', at least 3. |
    | `points` | `draw_spline` | Points on the curve [[x, y], ...] in order, 0-1000 coordinates of the full image or of the view given by 'view', at least 3. |
    | `box` | `grabcut` | Initial box [x1, y1, x2, y2] in 0-1000 coordinates of the full image or of the view given by 'view'. |
    | `points` | `select_superpixels` | Points [[x, y], ...] in 0-1000 coordinates of the full image or of the view given by 'view'; the superpixel containing each point is selected. |
    | `region` | `show` | Optional region [x1, y1, x2, y2] in 0-1000 coordinates of the full image or of the view given by 'view'. |

  - 返回图的调用，文字最前面：This is view k.
  - 给了非 0 的 `view`：Coordinates given in view k, converted to full-image 0-1000 coordinates.
  - 报错：
    - unknown view k (existing views: 0..n; 0 = the original image)（只有视图 0 时写 "existing views: 0"）
    - too many views (limit 64, ids 0..63); this tool returns an image and no view id is left
    - 换算之后的其他报错，末尾加：(coordinates were converted from view k to full-image 0-1000 coordinates)

## 10. 没做的与已知的问题

- **参数固定的 CRF：没做。** 任务书要求不装任何 CRF 库。
  - 负责人说的"参数固定的 CRF"通常指 DeepLab 式的全连接 CRF（Krähenbühl & Koltun 2011）。
    它的双边核消息传递要靠 permutohedral lattice 高维滤波；numpy、scipy、OpenCV 主模块与 scikit-image 里都没有实现。
    现成实现都是第三方库（pydensecrf、SimpleCRF）。
  - 自己写一个近似（例如用引导滤波代替双边核）就不再是标准算法，正确性要另外验证，论文里也不能称为"CRF"。
  - 网格（4/8 邻域）CRF 加对比度敏感的 Potts 项、用图割求解，正是 GrabCut 最小化的能量。
    `grabcut` 用 `mask` 初始化，等价于"以当前 mask 为一元项的网格 CRF 细化"，这部分功能已经覆盖。
  - 如果负责人要标准的全连接 CRF，需要另行批准装 pydensecrf，作为新的贴边工具。
- **原任务书 hand 组的 `render(name)` 已删去**（管理员裁定，DEV-0050 审查）：它与 `show` 给出同一张叠加图，A2（去掉 `check`）时
  模型仍能用它核对，消融不干净。现在叠回原图只在 check 组。
