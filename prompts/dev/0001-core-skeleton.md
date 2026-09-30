[DEV-TASK 0001] 与模型无关的基础件：接口抽象、光栅化与几何、网格编解码、指标、统计、计算记账、运行记录（CPU，不下载任何权重或数据）

## 背景

规格是 `docs/RESEARCH_BRIEF.md`（v0.2）。本任务做 Phase 0（§8.1）里与具体模型无关、E1/E2 马上要用的部分。

以下几块不在本任务里：

- 叠加渲染、分块拼接、无学习上采样、不确定度选区、mask 加噪，归 DEV-0002。
- 模型适配器的实现要等服务器 RUN-0001 报回真实 schema，归 DEV-0003。不要提前对 Qwen3.5 的格式编码。

先读：

- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（你的纪律）与 `references/lessons.md` 的"仪器""评测与报表"两节
- 简报：
  - §3.3；
  - §4，尤其第 2 条执行器白名单、第 6 条与模型无关、第 7 条防泄漏、第 11 条计算记账；
  - §7、§8.1、§8.4 的"显著"定义、§11.1、§11.5、§12。
- `DECISIONS.md` 的 D1、D4

## 任务

0. **环境**
   - 建 conda env `visint`（py3.12），装 numpy、scipy、pillow、shapely、pycocotools、pyyaml、pytest、torch（CPU 版）。
   - 写 `requirements.txt`：两台机器共用的核心依赖，不含 transformers、vllm 等模型栈。
   - 不往用户其他 conda 环境装东西。

1. **包布局照简报 §11.1。** `src/` 本身是包，有 `__init__.py`。子包 `adapters`、`executors`、`metrics`、`grid`、`noising`、`synth`、`probes`、`readout`、`multi`、`rl` 先建空的 `__init__.py`；本任务只实现下面列出的模块。另加 `tests/`，以及让 `pytest` 从仓库根能 `import src` 的最小配置。

2. **`src/constants.py`：跨模块约定的唯一来源（D4）。** 其他模块和测试一律从这里 import，禁止重述数字。每条约定注释写明依据或"暂定"。至少包含：
   - **坐标：**
     - 内部统一用绝对像素、浮点、xyxy，原点左上，x 向右、y 向下；
     - 像素 (i, j) 占据连续区间 [j, j+1) × [i, i+1)；
     - 框光栅化规则：像素中心落在框内即为 1；
     - 多边形光栅化一律走 pycocotools，与 COCO/RefCOCO 的 GT 同一套规则。
   - **网格：**
     - E3b 默认 32，E1 零样本 16；
     - mask → 网格：格内前景面积占比 ≥ 0.5 记 1；
     - 网格 → mask：按格边界 `floor(k*H/N)` 精确回填。
   - **深度分档：**
     - 64 档对数均匀，`D_MIN = 0.5`、`D_MAX = 10.0`（米），越界裁剪，反解码取档的几何中心；
     - 注释写"暂定，E3b 预注册前定"。
   - **NYUv2 评测：**
     - Eigen crop，与 BTS/AdaBins 评测代码一致，在 480×640 上取行 45:471、列 41:601；
     - 有效深度 (1e-3, 10.0]；
     - 预测先裁剪到同一区间。
   - **Boundary IoU：** 膨胀比例 0.02（相对图像对角线）。
   - **ADE20K：** 150 类，`reduce_zero_label=True`，忽略值 255（对齐 mmseg）。
   - **bootstrap：** 默认重采样 1000 次，95%，固定种子。

3. **`src/adapters/base.py`：简报 §8.1 适配器接口的抽象基类和数据结构。** 只有签名、类型注解和 docstring，不实现任何模型，不 import transformers。
   - 抽象方法：
     - `load(model_id, dtype)`
     - `build_inputs(images, instruction, order)`，order 取 `"instruction_first" | "image_first"`
     - `generate(inputs, *, enable_thinking, allowed_token_ids=None, max_new_tokens, **gen_kwargs)`
     - `hidden_states(inputs, layers)`
     - `inject_query_tokens(n, position_mode)`，position_mode 取 `"grid2d" | "seq1d"`
     - `add_grid_tokens(names)`
     - `num_layers()`、`hidden_size()`
   - `enable_thinking` 和 `max_new_tokens` 是必填关键字参数，没有默认值（D1：每次调用都显式设置）。写一条测试：漏传时报 `TypeError`。
   - 层号只能作为参数传入。基类和文档里不出现任何具体层号。
   - `ModelInputs`：在张量之外带元数据，包括每张图的视觉 token 区间与 (h, w) 网格、指令区间、查询 token 区间、order。
   - `GenerationResult`：文本、token ids、生成 token 数、prompt token 数、视觉 token 数、墙钟时间。
   - `HiddenStates`：按层给出图像 token 位置和查询 token 位置的张量，附层号列表。

4. **`src/executors/`（全部无学习，只用简报 §4 第 2 条白名单里的库）**
   - `raster.py`：框 → mask、多边形 → mask（pycocotools）、RLE 编解码。
   - `geometry.py`：
     - 框 IoU，连续坐标，不加 1；
     - 多边形 IoU（shapely）；
     - 坐标格式换算：绝对像素 ↔ 0–1000 归一化 ↔ 0–1。换算显式要求原图宽高；往返换算要有可逆性测试。

5. **`src/grid/codec.py`**
   - mask ↔ N×N 网格。
   - 深度 ↔ 64 档（规则按 constants）。
   - E1 用的"数字字符网格"：文本 ↔ 数组。
     - 序列化：每行 N 个 0/1，行间换行。
     - 解析：从模型输出里找恰好 N 行、每行恰好 N 个 0/1 的块，允许数字之间有空格或逗号。
     - 行数或列数不对、夹杂非法字符时，返回 `None` 和失败原因。不补齐、不截断、不猜。
     - 失败原因是有限的枚举，供统计解析失败率。

6. **`src/metrics/`**
   - 每个指标的 docstring 写明对齐的是哪份官方代码的哪个函数（仓库 + 文件 + 函数名）。
   - **框：** 逐样本 IoU 和 Acc@IoU0.5，IoU ≥ 0.5 记对。
   - **分割：**
     - cIoU：累计交除以累计并。
     - gIoU：逐样本 IoU 的平均；并集为 0 时该样本记 1.0，写明并测试。
     - Boundary IoU：对齐 bowenc0221/boundary-iou-api 的 `mask_to_boundary`，包括它的 1 像素零填充、3×3 腐蚀、迭代次数的取整规则。用 scipy 实现；测试环境若装了 opencv，就逐位对照它的实现（可选依赖，只用于测试）。
     - mIoU：按类在整个数据集上累计交与并，并集为 0 的类不计（nanmean），对齐 mmseg。
   - **深度：** AbsRel、δ1（<1.25），按 constants 的有效区域计算，两种都算：
     - 度量深度；
     - 尺度-平移对齐：在有效像素上对预测做深度空间的最小二乘，拟合 s·pred + t，再裁剪、计算。
   - **法向：** 单位化后的角误差（度），给出平均值、中位数，以及 <11.25°、<22.5°、<30° 的比例；GT 无效像素掩掉。
   - **其他：**
     - 计数准确率（整数精确匹配）；
     - 相对深度点对准确率；
     - 多选题准确率（选项字母规范化后匹配）；
     - 开放词汇检测 mAP：包一层 pycocotools 的 COCOeval。输入是无分数的预测时统一记 score=1.0，写明；输出 AP 与 AP50。
   - **所有指标：** 空集合返回 `None`，不是 0.0；每个指标至少一条对照手算小例子的测试。

7. **`src/stats.py`**
   - 逐样本 bootstrap 95% CI（percentile 法）。
   - 配对 bootstrap：两个系统在同一批样本上的差值 CI。
   - 简报 §8.4 的"显著"判定。输入是两种方法各自逐 seed 的指标，加上逐样本值。两个条件都满足才算显著：
     - |均值差| > 2 × 合并标准差，合并标准差 = sqrt((s_A² + s_B²)/2)，ddof=1；
     - 配对 bootstrap 差值 CI 不含 0。
   - 函数的 docstring 写明这是对简报措辞的实现解释，待 G0 确认。

8. **`src/accounting.py`：计算记账（硬约束 11）**
   - 每次调用记录：生成 token 数、prompt token 数、视觉 token 数、墙钟时间、轮数、调用数。
   - 支持累加与合并。
   - 写 `compute.json` 时字段名固定，用一个 schema 版本常量。

9. **`src/runlog.py`：实验目录与元数据**
   - 创建 `experiments/<phase>/<NNNN_slug>/`：
     - 运行编号不是 4 位数字就拒绝；
     - 目录已存在且非空就拒绝（不覆盖）。
   - 写入：
     - `config.yaml` 快照、seed；
     - git commit，含 dirty 标记与 diff 统计；
     - 数据版本：给定划分文件的 sha256；
     - 主机名、GPU 名（没有 CUDA 就记 `null`）、关键包版本、UTC 时间戳。

10. **tests/**
    - 每条行为至少一个断言。
    - 合成 fixture 故意 H≠W，内容和批内样本也互不相同（lessons：对称的 fixture 抓不到转置、取错样本这类错误）。
    - 全部在 CPU 上 60 秒内跑完。

## 约束

- 不写任何模型相关代码，不 import transformers；不下载任何权重或数据集。
- 执行器只用白名单里的东西：numpy、scipy、shapely、PIL、pycocotools，以及 torch 的非学习算子。
- 不用 GrabCut、分水岭、超像素、CRF（灰名单），不用任何带学习参数的模块。
- 约定只在 `constants.py` 定义，测试也从那里 import。
- 不造通用框架：注册表、插件系统、配置 DSL 都不要。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿。报告测试数与耗时。
- 下面的 import 无报错：

  ```bash
  conda run -n visint python -c "import src.constants, src.adapters.base, src.executors.raster, src.executors.geometry, src.grid.codec, src.metrics, src.stats, src.accounting, src.runlog"
  ```

- 在一张合成的 H≠W 图上彩排一遍：GT 多边形 → mask → 16×16 网格 → 数字文本 → 解析回网格 → 回填 mask → cIoU/gIoU/Boundary IoU。把数值写进完成总结。
- 任何偏离任务书之处，逐条列出待裁定。任务书自相矛盾、或某个定量说法不成立时，同样上报，不要静默改设计。

完成后不要 commit、不要 push，向用户总结改动并说"已完成，请交给管理员检查"。
