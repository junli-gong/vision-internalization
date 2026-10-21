[DEV-TASK 0054] W1 工作台的"视图坐标"：每张返回的图有编号，带坐标的工具可以按某张视图自己的 0–1000 坐标给点和框（可选开关，默认关；D103 §二.2；RUN-0060 冒烟的诊断）

## 背景

- RUN-0060（9B，开发-调参集 32 句）：
  - 模型写进 `grabcut`、`draw_box` 的框，与它自己的原生框（E1 定位，这 32 句里 Acc@0.5 约 0.9）相比，IoU 中位数只有 0.68。
  - 这 27 个框里有 24 个是在放大之后给的。
- 典型的错：
  - 先放大一块区域，再把框写成放大窗口的边界，或整百的数；
  - 模型在放大图上看到的位置，要按文字里的换算式换回原图坐标，算错或懒得算。
- Qwen 的原生坐标是"所看那张图的 0–1000"。如果模型能直接按放大图自己的 0–1000 给坐标，由工作台换回原图，就不用心算换算式了。
- 这是工作台的通用功能，与任务无关：所有带坐标的工具都适用。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w1_plan.md` §0、§2；`docs/w1_tools.md`（全文）
- `src/w1/workbench.py`、`tests/test_w1_workbench.py`
- DECISIONS.md 的 D103、D106

## 任务

### 1. `Workbench(image, *, disabled_groups=…, sandbox=…, view_frames: bool = False)`

**`view_frames=False`（默认）：行为完全不变。**
- `specs()` 逐字节不变：`tools_digest` 与改动前相同，写进测试钉住。
- 所有工具返回的文字、图、mask 逐位不变。现有测试原样通过。

**`view_frames=True`：**

1. **视图编号**：
   - 视图 0 是原图本身，即第一条 user 消息里的图。
   - 每次工具调用返回的每一张图都登记成一个新视图，编号从 1 起递增。登记内容是这张图覆盖的原图区域，用原图的 0–1000 坐标 `[x1, y1, x2, y2]` 表示。
   - 覆盖整张原图的图（`tile_view`、`edges`、`superpixels`、不带 region 的 `show`）也登记，区域是 `[0, 0, 1000, 1000]`。
   - 返回文字里写明 "This is view k."，放在文字最前面或描述区域的句子里，你定，写进文档。
   - 至多 `MAX_VIEWS` = 64 个视图，超过就报错，不复用编号。
2. **`view` 参数**：
   - 所有带坐标参数的工具（`zoom`、`read_pixels`、`draw_box`、`draw_points`、`draw_polygon`、`draw_spline`、`grabcut`、`watershed` 的点或框、`select_superpixels` 的点、`show` 的 region，以及你在代码里找到的其他所有坐标参数）都加一个可选的整数参数 `view`，默认 0。
   - 给了 `view` 时，这次调用里的所有坐标都按那张视图自己的 0–1000 读（x 在前），先换成原图 0–1000 再执行：`x_full = vx1 + x / 1000 · (vx2 − vx1)`，y 同理。之后走现有的 `norm1000_to_abs`。
   - 坐标仍按 0–1000 校验（在视图坐标里校验）。
   - 不存在的视图编号报错，返回错误文字，不抛异常。
   - 返回文字里的坐标一律是原图 0–1000，与现在相同。可以加一句"(coordinates given in view k)"。
   - 长度类参数（半径、像素数）不受影响。
3. **工具说明**：
   - 带坐标的工具多一个 `view` 参数，说明一句即可，例如 "Optional id of an earlier view (0 = the original image); if given, all coordinates in this call are 0-1000 relative to that view."
   - 说明里不出现任何任务词（现有测试钉住）。
   - `zoom` 与 `show` 现在文字里的换算式保留。
4. **日志**：`Workbench.log()` 记每次调用登记的视图编号与区域，以及用了哪个 `view`。
5. **确定性**：同样的调用序列，结果逐位相同。

### 2. 文档

`docs/w1_tools.md` 加一节"视图坐标（`view_frames`，默认关）"：
- 编号规则、换算式、哪些工具带 `view`；
- 给模型的原文逐字照录。

### 3. 测试（`conda run -n visint pytest tests/ -q` 全部通过）

- 默认关时 `specs()` 的 sha256 与改动前相同。
- 开时：
  - `zoom` 一块区域，再用 `view = k` 画框，结果与直接用原图坐标画的同一个框逐位相同；点、多边形同理。
  - 视图编号递增；不存在的编号报错；超过 `MAX_VIEWS` 报错。
  - 覆盖整图的视图换算是恒等。
  - 同样的调用序列，结果逐位相同。
- 与 DEV-0053 的接口：`Workbench(img, disabled_groups=frozenset(), view_frames=True)` 能构造。DEV-0053 只在开关为真时传这个参数。

## 不做

- 不改 `src/w1/loop.py`、`config.py`、`runner.py`、`scripts/run_w1.py`、`configs/w1.json`：这些由 DEV-0053 做。
- 不加新工具，不改现有工具的算法与固定参数。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 带 `view` 的工具清单；
- 给模型的原文；
- 测试结果原文；
- 与本任务书的任何出入。
