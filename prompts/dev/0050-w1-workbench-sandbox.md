[DEV-TASK 0050] W1：工作台（眼、手、贴边、核对、提交）与 Python 沙箱。只用不含学习的算法；与任务无关（D103；接口约定见 `docs/w1_plan.md` §0、§2）

## 背景

- W1（D103）：冻结的 LLM 在多轮回路里调用工具完成指代分割、定位与计数。
- 原则：**语义判断由 LLM 做，底层拟合交给不含学习的算法。**
- 负责人给的工具边界（D103 §二.2，原文照用）：

  | 角色 | 工具 |
  |---|---|
  | 眼 | 裁剪、放大、像素与颜色读数、分块查看 |
  | 手 | 框、点、多边形、样条；光栅化；叠加渲染 |
  | 贴边 | 边缘检测、GrabCut、分水岭、超像素（SLIC）、引导滤波、形态学运算、连通域；参数固定的 CRF |
  | 核对 | 把结果叠加回原图，交给模型再看一次 |
  | 编排 | Python 沙箱，只允许 numpy、scipy、OpenCV 和 scikit-image 中不含学习的函数，不联网 |

  - **禁止**：任何学习型视觉模型或预训练权重，包括 SAM、检测器、深度模型、CLIP、DINO、OCR 模型；OpenCV 的 DNN 模块。
- **工作台必须与任务无关**：所有任务用同一套工具；工具说明里不写任何任务的解法。

## 先读

- `CLAUDE.md`；`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w1_plan.md`；`DECISIONS.md` 的 D103；`docs/lead_answers/2026-10-20-w1-directive.md`
- 可复用的部分：
  - `src/executors/raster.py`：`box_to_mask`、`polygons_to_mask`，即 GT 的光栅化规则；
  - `src/executors/overlay.py`：`render_overlay`、`mask_contour`。它的签名被测试钉住不能改，需要的话另写函数；
  - `src/executors/upsample.py`：`guided_filter`；
  - `src/executors/geometry.py`：0–1000 与绝对坐标的换算；
  - `src/metrics/seg.py`。

## 任务

1. **环境**：
   - 开发机 conda env `visint` 里装 `scikit-image==0.24.0`（`pip install --no-cache-dir`，不改已有的包）。
   - 用已有的 cv2，在完成总结里写明版本。代码只用 OpenCV 4.10 与 5.x 都有的 API。
   - `requirements.txt` 加上 `opencv-python-headless==4.10.0.84` 与 `scikit-image==0.24.0`。HPC 由 RUN-0058 安装。
   - 不装任何 CRF 库。"参数固定的 CRF"如果不靠第三方库实现不了，就不做，写明理由；其余工具都要有。
2. **`src/w1/workbench.py`**：`Workbench`、`ToolResult`、`Submission`、`GROUPS`，签名照 `docs/w1_plan.md` §2。
   - 坐标一律用原图 0–1000 相对坐标，x 在前。
   - 每个工具都有 `ToolSpec`（`src/adapters/chat.py` 的定义，见 DEV-0049；先按约定的字段写，合并时对齐）：名字、一句话说明、JSON Schema 参数。
   - 工具至少包括：
     - **eye**：`zoom(x1,y1,x2,y2)`，返回放大视图，长边固定像素，并写明覆盖范围；`read_pixels(points)`，返回 RGB 值与亮度；`tile_view(rows, cols)`，返回带格号的分块图。
     - **hand**：`draw_box`、`draw_points`（正负点）、`draw_polygon`、`draw_spline`（闭合样条）。都写进一个命名 mask，可新建、并、差、交。`render(name)` 把 mask 叠加到原图返回。
     - **snap**：
       - `edges(…)`：Canny，参数可选，有固定默认；
       - `grabcut(name 或 box, iterations)`：用已有 mask 或框初始化；
       - `watershed(markers)`；
       - `superpixels(n_segments, compactness)`：SLIC；返回带编号的叠加图，并提供"按编号选取"的方式；
       - `refine(name)`：引导滤波加阈值；
       - `morph(name, op, radius)`；
       - `components(name)`：连通域列表，可选保留第 k 个。
     - **check**：`show(name)`，把 mask 的轮廓与填充叠加到**原图**，返回给模型。
     - **code**：`run_python(code)`，见第 3 项。
     - **submit**：`submit_mask(name)`、`submit_box(x1,y1,x2,y2)`、`submit_count(n)`。
   - 出错（参数不对、越界、未知的 mask 名）返回 `ToolResult.error`，不抛异常。
   - `log()` 记每次调用：名字、参数、耗时、是否出错、沙箱代码原文、产生的 mask 的面积。
   - `disabled_groups`：被关掉的组不出现在 `specs()` 里；`call()` 调到时返回错误。
3. **`src/w1/sandbox.py`**：`SandboxConfig`、`run_code(code, *, image, masks, config) -> SandboxResult`。
   - 子进程、`python -I`、空环境变量（不带 HF_HOME 等）、临时工作目录。
   - 限时（默认 20 s）、限内存（默认 4 GB，`resource.setrlimit`）。超时就结束子进程。
   - **import 白名单**，用 `sys.meta_path` 的拦截器加启动前的 AST 检查：
     - 允许：numpy、scipy（子模块都允许）、cv2、skimage、math、itertools、collections、functools、json、re、statistics。
     - 屏蔽：`cv2.dnn`、`cv2.dnn_superres`、`cv2.face`、`cv2.text`、`cv2.CascadeClassifier`、`cv2.HOGDescriptor` 的默认检测器；`skimage.feature.Cascade`、`skimage.data`。
     - 其余一律拒绝，包括 torch、transformers、os、subprocess、socket、pathlib、importlib。
   - 没有 `open`、`exec`、`eval`、`__import__` 可用。
   - 输入：图像（uint8 HxWx3）与命名 mask（bool），经 npz 传入。
   - 输出：stdout（截到 4,000 字符），以及代码里赋给 `result_mask` 的 mask，经 npz 取回，不用 pickle。
   - 这是研究诚信的防线：防止模型借代码调用学习型模型或读权重。它不是对抗性安全边界。在模块说明里写清楚这一点。
4. **`docs/w1_tools.md`**：工具清单。每个工具写名字、参数、返回、所属的组、算法与固定参数。这份文档会进论文的附录。
5. **测试**（`conda run -n visint pytest tests/ -q` 全部通过；`tests/test_w1_workbench.py`、`tests/test_w1_sandbox.py`）：
   - 合成图（几何形状）上：
     - 框、多边形、样条的光栅化与 `src/executors/raster.py` 的规则一致；
     - grabcut、watershed、superpixels、refine、morph、components 在明确的情形下给出预期的 mask，例如纯色圆在噪声背景上；
     - zoom 的覆盖范围与坐标换算对；
     - check 的叠加图尺寸与原图相同。
   - `specs()` 的 JSON Schema 合法；关掉一组之后它不出现，调用时报错。
   - 沙箱：
     - 白名单内的代码跑通，并返回 mask；
     - 每一种被禁止的 import 与调用都被拒（torch、transformers、os、socket、cv2.dnn、CascadeClassifier、skimage.data、open、`__import__`）；
     - 超时与超内存被结束；
     - 环境变量为空。
   - 工作台完全确定：同样的调用序列，结果逐位相同。

## 不做

- 不写回路与任务（DEV-0051），不改适配器（DEV-0049），不改 configs/。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；工具清单；沙箱的规则；
- 测试列表与结果原文；
- 开发机上的 cv2、skimage 版本；
- CRF 做没做、为什么；
- 对 `docs/w1_plan.md` §2 约定的任何偏离或建议。
