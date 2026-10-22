[DEV-TASK 0058] W2 的候选生成器与挑选图渲染（D113；接口见 docs/w2_plan.md §1、§2）

## 背景

- W2-a：LLM 用 E1 的原生格式给框，可选再给几个点；工具在框和点上生成 K 个候选 mask；把候选编号后叠回原图交给 LLM，它只回答一个编号或"都不对"。
- 本任务实现候选与渲染两部分。回路、划分、对照与判定由并行的 DEV-0059 做。
- W1 的数据给出的依据（G-W1）：开发-检查集上，自己的框 0.507，W1-F（框交给默认 GrabCut）0.576。候选池要在 W1-F 的基础上，提供可能更好的选择。
- 候选池的上限，即"最优候选"，决定了挑选最多能做到多好；候选的多样性比单个候选的平均质量更重要。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w2_plan.md`（全文，§0–§2 是本任务的约定）、`docs/lead_answers/2026-10-22-w2-directive.md`
- `src/w1/workbench.py`：`grabcut_mask`、SLIC、分水岭、形态学、光栅化、叠加渲染与字体的现有实现，尽量复用，不复制粘贴；`docs/w1_tools.md`
- `src/w1/baselines.py`：W1-F 怎样调用 `grabcut_mask`
- `src/executors/raster.py`、`src/executors/geometry.py`

## 任务

1. **`src/w2/candidates.py`**，签名照 `docs/w2_plan.md` §1：
   - `Candidate`、`CandidateSpec`、`generate_candidates`，以及 `GENERATOR_TYPES`，即全部生成器类型与参数的说明。
   - 至少实现 §1 列出的几类生成器，每类的参数写进 `CandidateSpec`。
   - **1 号候选必须是 `grabcut_default`，并且逐位等于 W1-F 的输出**：用 `src/w1/baselines.py` 的同一条调用路径，写测试钉住。
   - 失败时用框本身代替，`ok = False`；K 不变。
   - 只用不含学习的算法。
2. **`src/w2/render.py`**：`render_selection(image, candidates, layout, style)`，两种布局 `grid` 与 `separate`，见 §2。
   - 编号大而清楚，叠加不遮住物体的边缘：半透明填充加轮廓。
   - 默认的样式参数写进 `src/w2/render.py` 的常量，并在 `docs/w2_tools.md` 里列出。DEV-0059 会把它们抄进 `configs/w2.json`。
   - 每种布局的输出尺寸，以及在 Qwen3.5 的 `PixelBudget(300)` 下的视觉 token 数：用适配器的图像处理现算，不要估。
3. **默认的候选清单**：在 `src/w2/candidates.py` 里给一个 `DEFAULT_SPEC`，8 个生成器，作为调参的起点。理由写进文档。DEV-0059 会把它写进 `configs/w2.json`。
4. **`docs/w2_tools.md`**：
   - 每种生成器的算法、参数、失败时的处理；
   - 渲染的样式；
   - 640×480 图上每种生成器的单次耗时，用 tests/fixtures 的图在开发机上实测，并注明机器；
   - 写成论文附录的口径。
5. **CPU 用时**：报告 `DEFAULT_SPEC` 在一张 640×480 图上生成 8 个候选的总耗时。W2-c 要在 RefCOCO train 上生成几万句的候选，这个数决定能做多大的训练集。
6. **测试**（`tests/test_w2_candidates.py`、`tests/test_w2_render.py`；`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w2_*.py tests/test_w1_workbench.py -q` 全部通过）：
   - 1 号候选与 W1-F 逐位相同；
   - 每种生成器都能跑通，各种边界情况：框贴边、框极小、没有点、点在框外；
   - 失败时用框代替；
   - 确定性：同样输入逐位相同；中途打乱 OpenCV 的全局随机数也一样；
   - 渲染的尺寸、编号顺序、逐位确定。

## 不做

- 不写回路、划分、对照、判定，不写 `configs/w2.json` 与 `scripts/run_w2.py`：这些由 DEV-0059 做。
- 不改 W1 的代码。只能在 `src/w1/workbench.py` 里把已有的私有函数提成可复用的公共函数，并且要保证 W1 的 specs、文字、图、mask 逐位不变，测试照常通过。这种改动在完成总结里列明。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 生成器清单与 `DEFAULT_SPEC` 的理由；
- 渲染的尺寸与视觉 token 数；
- 耗时；
- 测试结果原文；
- 与 `docs/w2_plan.md` 的任何出入。
