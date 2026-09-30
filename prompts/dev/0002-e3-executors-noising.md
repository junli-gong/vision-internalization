[DEV-TASK 0002] E3 用的执行器（叠加渲染、分块拼接、无学习上采样、不确定度选区）与 mask 加噪过程

## 背景

E3a、E3b、E3c 都要用一组与任务无关的"乐器"执行器，E3b 还要一个 mask 加噪过程来免费制造"带错 → 正确"的训练对。本任务把它们做成纯 numpy/scipy 的模块。与模型无关，只跑 CPU。

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含末尾"作为子 agent"一节）
- `.claude/skills/exp-triad-flow/references/lessons.md` 的"仪器"一节
- 简报：
  - §3.7–3.9（扩散的三个思想、叠加反馈、分块与粗网格校正）；
  - §4：第 2 条执行器白名单，第 11 条"多轮流程与任务无关"；
  - §8.4 的上采样、§8.5 的边界细化、§8.6 的分块、§12 的"叠加渲染的一致性"。
- `DECISIONS.md`：D4
- `DECISIONS_NEEDED.md`：N8（选区按不确定度统一）
- 已有代码：`src/constants.py`、`src/executors/`、`src/grid/codec.py`、`src/metrics/seg.py`

## 任务

1. **`src/constants.py` 新增几节**。每条注明依据，没有依据的标"暂定"：
   - 叠加渲染：调色板（至少 8 种易区分的颜色）、alpha、是否画轮廓、轮廓宽度；
   - 分块：默认块大小、重叠像素；
   - 引导滤波：半径、eps；
   - 联合双边上采样：空间与值域 sigma、半径；
   - 不确定度选区：区域数、区域边长；
   - 加噪：严重度的分布与各扰动的强度范围。

   **只新增，不改已有常量。**

2. **`src/executors/overlay.py`**：`render_overlay(image, masks)`，返回 RGB uint8，参数取自 constants。
   - 多个 mask 按顺序取不同颜色，结果确定。
   - mask 外的像素与原图逐位相同。mask 内按 alpha 混合公式计算，整数化规则写明并测试。
   - 轮廓 = mask 减去 3×3 腐蚀结果，用 scipy 实现，画成实色。

3. **`src/executors/tiling.py`**
   - `make_tiles(H, W, tile, overlap)`：返回块坐标列表。覆盖整张图，最后一块贴边对齐，不做填充，结果确定。
   - `stitch(tile_values, tiles, H, W)`：重叠区做平均，权重 = 覆盖该像素的块数。
   - 测试：
     - 能整除与不能整除两种尺寸；
     - 每个像素至少被一块覆盖；
     - 从已知 float64 图切出块再拼回，逐位还原；
     - 常数输入拼出常数。

4. **`src/executors/upsample.py`**（无学习，只用白名单）
   - `guided_filter(guide, src, radius, eps)`：He et al. 的盒式滤波实现，灰度引导和彩色引导（逐像素 3×3 协方差求逆，向量化）都要。
   - `joint_bilateral_upsample(low, guide, …)`：Kopf et al. 2007。
   - `guided_upsample(low_logits, guide_image)`：先双线性放大，再用引导滤波。这是简报 §8.4 的默认做法。另提供 `bilinear_upsample` 作为对照。
   - 测试：
     - 引导图为常数时，引导滤波精确退化为盒式均值滤波；
     - 合成阶跃边缘加噪声时，引导上采样的边界误差（用 Boundary IoU 或边界带内的误差）小于双线性；
     - 彩色版在三通道相同的输入上与灰度版一致（容差写明理由）。

5. **`src/executors/uncertainty.py`**：`select_uncertain_regions(dist, k, size)`。
   - 输入是逐像素分布：二值情形给前景概率 (H, W)，多类情形给 (H, W, C)，例如 64 档深度。
   - 不确定度一律用熵，选出 k 个互不重叠的正方形区域（非极大抑制），平局按固定规则打破。
   - **函数不接受任何"任务类型"参数**（简报 §4 第 11 条、N8）。mask 和深度用同一个函数。
   - 测试：
     - 熵最高的位置被选中；
     - 区域不重叠、不越界；
     - k 大于可选区域数时的行为写明；
     - 二值与多类两种输入。

6. **`src/noising/mask_noise.py`**：mask 加噪过程（简报 §3.7 第 2 条）。
   - 扰动算子：腐蚀、膨胀、边界抖动、随机增删区域、平移、与邻近物体合并。合并要传入其他物体的 mask。
   - 每个算子由严重度 s ∈ [0, 1] 控制。
   - `sample_severity(rng)` 按 constants 里的"噪声日程"采样。
   - `corrupt(mask, severity, rng, others=())` 返回 (带噪 mask, 实际施加的算子与参数的记录)。
   - 纯 numpy/scipy；给定种子，结果确定。
   - 测试：
     - s=0 时逐位不变；
     - 固定一组种子下，与原 mask 的平均 IoU 随 s 单调下降；
     - 各算子语义：膨胀面积不减、腐蚀面积不增、平移的位移与参数一致、合并包含邻近物体的像素；
     - 输出是 bool、形状不变、不越界；
     - fixture 的 H≠W，批内样本互不相同。

7. **tests/**：新增测试全部在 CPU 上 60 秒内跑完。

## 允许改动的文件

- `src/constants.py`（只新增）
- `src/executors/overlay.py`、`src/executors/tiling.py`、`src/executors/upsample.py`、`src/executors/uncertainty.py`
- `src/noising/mask_noise.py`
- `tests/test_overlay.py`、`tests/test_tiling.py`、`tests/test_upsample.py`、`tests/test_uncertainty.py`、`tests/test_mask_noise.py`

**不许碰 `src/adapters/` 和 `requirements.txt`**：同时在跑的 DEV-0003 会改它们。本任务不需要新依赖。

## 约束

- 只用白名单里的东西：numpy、scipy、PIL。不用 GrabCut、分水岭、超像素、CRF，不用任何带学习参数的模块。
- 约定只在 `constants.py` 定义，测试也从那里 import。
- 执行器都是任务无关的"乐器"，函数签名里不出现任务身份。
- 不造通用框架。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿。原有测试保持通过，报告新增测试数与耗时。
- 在一张合成的 H≠W 图上彩排一遍，把数值写进总结：
  1. GT mask 经 `corrupt` 加噪（s=0.5）；
  2. `render_overlay` 渲染；
  3. 分块后拼回，验证逐位还原；
  4. 低分辨率 logit 图分别用 `guided_upsample` 与 `bilinear_upsample` 放大，比较两者的 Boundary IoU；
  5. 在上采样后的概率图上 `select_uncertain_regions`，给出选中区域。
- 逐条列出偏离，写明"暂定"常量的取值理由。
