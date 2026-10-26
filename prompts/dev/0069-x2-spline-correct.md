[DEV-TASK 0069] 探索 X-2：让 9B 自己描 B 样条、指出带符号的修正框（LoRA SFT，目标来自真值），以及推理运行器与消融

## 背景

- 负责人授权自主探索（D126-3），见预注册 X 与 `docs/exploration/log.md` 的 X-0、X-1。
- **X-0**：W2-c 的损失 58% 出在框已经很好的句子上，瓶颈是从框到 mask。
- **X-1**：真值模拟的"理想 LLM"在 devcheck2 上，用 W2-c 的框：
  - B 样条 24 个控制点直接光栅化 0.899；12 个控制点作 trimap 种子 0.836；
  - 从 W2-c 的输出起，带符号的局部修正每轮 2 框、3 轮，0.788（grabcut 规则）；
  - W2-c 本身 0.642。
- **证据**：9B 的 C3 臂（E3，真值多边形文本 SFT）能解析的部分 cIoU 0.61；失败几乎都是截断，格式太长。
- **本任务**：训一个单一的 LoRA，在 W2-c 原有的提出、挑选两个任务之外，再学两件事：
  - 描样条：给图、指代句与框，输出 n 个控制点；
  - 指修正框：给当前 mask 的叠加图，输出至多 m 个带符号的框。

  再写推理运行器，把它们拼成几条管线，做消融。
- **边界**（D126-3 硬边界）：
  - 只用不含学习的工具，即 `src/explore/guidance.py`；
  - 只用开源模型，单一权重；
  - 监督只来自真值，以及基座的自蒸馏回放；
  - 照硬约束 11 计算记账；
  - 不碰 test；devcheck3 只能经 `--unseal-devcheck3 G-EX`（D130 的守卫）。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 预注册 X、`docs/exploration/log.md`、`docs/references/llmseg_notes.md`（样条与修正的提示与参数化）
- `src/explore/guidance.py`、`src/explore/oracle_sim.py`（DEV-0068：样条拟合与光栅化、trimap、局部修正、软锁定、effect gate、真值模拟）
- `src/w2/w2c_train.py`、`scripts/train_w2c.py`、`configs/w2c_train.json`、`configs/w2c_replay_train.json`（流的机制、回放流，DEV-0065 已把多流做通用）
- `src/w2/w2c_data.py`（数据行：`gt_rle`、`proposal.box`、`candidates[].rle/iou`）、`src/w2/render.py`、`src/w2/loop.py`、`src/w2/runner.py`、`scripts/run_w2.py`
- `src/data/explore_splits.py`（explore_val、devcheck3 与守卫）、`scripts/gate_explore.py`（输入格式）
- `src/adapters/qwen35.py`（坐标 0–1000 网格，`abs_to_norm1000` / `norm1000_to_abs`）

## 任务

### A 期（先做；做完训练就能开跑）

1. **提示与解析** `src/explore/x2_prompts.py`：两个新任务的提示原文、输出格式、解析与校验，全部写成常量与纯函数。坐标都用 0–1000 网格的整数，同 E1。
   - **描样条**：
     - 输入：图、指代句、提出的框（0–1000）。
     - 提示要说清：物体在这个框内；用 n 个控制点的闭合周期均匀三次 B 样条描它的外轮廓；只输出 JSON `[[x, y], …]`，恰 n 个点。n 是参数，缺省 16。
     - 解析：严格 JSON；点数不对、越出 0–1000 都记为失败，失败原因分类计数。
     - 光栅化用 `guidance` 的同一套，采样点夹在框内。
   - **指修正框**：
     - 输入：一张叠加图，即原图上画出当前 mask（半透明填充加轮廓）与提出的框；再加指代句。叠加图的画法放进 `src/explore/x2_render.py`，风格与 W2 的渲染一致，单张图。
     - 提示要说清：
       - 指出至多 m 个需要修改的局部，每个给符号与框，"+" 表示漏掉、要补，"−" 表示多出、要删；
       - 没有要改的就输出 `[]`；
       - 只输出 JSON `[{"sign": "+", "box": [x1, y1, x2, y2]}, …]`。
       - m 是参数，缺省 2。
     - 解析同上，失败分类计数。
   - 两个提示都关思考，与 W2-c 的提出、挑选相同。
2. **训练数据** `src/explore/x2_data.py` 与 `scripts/build_x2_data.py`：在 CPU 上从 W2-c 的训练数据行（RUN-0067/0069 的 `w2c_data`，16k 句）算目标，不跑模型。
   - **样条行**：
     - 目标：`gt_rle` 的外轮廓按 `guidance` 拟合 n 个控制点，取整到 0–1000；填洞后取最大连通块，与 DEV-0068 的 `parts = 1` 相同。
     - 框：数据行里提出的框，即基座的 E1 框。
     - 提出的框与真值框 IoU < 0.5 的行不用，按任务书统计数量。
     - 每行另记拟合的真值 IoU，即拟合本身的上限。
   - **修正行**：
     - 起点 mask：取数据行 4 个候选里的一个，按确定性规则轮换，覆盖好、中、差三类候选。
     - 目标：起点与真值之差的 8 连通块，限在提出的框内，经 effect gate 过滤（门槛同 `guidance`）。按面积从大到小取前 m 个，每块的外接框取整到 0–1000：漏掉的为 "+"，多出的为 "−"。都过不了门槛时目标是 `[]`。
     - 另造第 2 轮的行：把第 1 轮的真值修正用 `guidance` 的 grabcut 规则加软锁定施加到起点上，得到第 2 轮的起点，再按同样规则算目标。
     - 两轮的比例是参数。
   - 输出 jsonl 加 manifest：总体、过滤、各类的条数与 sha256、拟合上限的分布、目标为 `[]` 的比例。可续跑，除耗时外逐字节可复现。
3. **训练** `configs/x2_train.json`：在 `train_w2c.py` 上加两种流 `spline` 与 `correct`，机制同 DEV-0065 的回放流。
   - **缺省配置**，即这次要跑的：
     - 每步 propose 8、select 8、spline 6、correct 6，加回放 general 3、count 2；
     - 各流按本流的监督 token 数平均；
     - 权重 propose 1、select 1、spline 1、correct 1、general 0.5、count 0.5；
     - 2,000 步，9B，LoRA r32/α64，与 W2-c 相同。
     - W2 的部分（候选、渲染、K = 4）用冻结的 `configs/w2.json`。
   - 叠加图在训练时现画，同 W2-c 的挑选渲染。
   - 没有 spline、correct 两节的配置，`train_w2c.py` 的行为逐字节不变，用 7f53f240 的快照对照测试钉住，同 DEV-0065。
   - 检查点格式与 W2-c 相同，`run_w2.py --setting w2c` 也能加载。
   - **用时估计**：每步多 12 条，按 RUN-0069 的每步 10.9 s 折算，写进文档字符串。超过 12 GPU 小时的话，给出 1,500 步的备选。

### B 期（A 期完成后接着做）

4. **推理运行器** `scripts/run_x2.py` 与 `src/explore/x2_runner.py`：一个检查点，几条管线，每条一个设置名。输出 W2 格式，`scripts/gate_explore.py` 能直接读。
   - **划分**：
     - `--splits-file configs/explore_splits.json` 的 explore_val 三层；
     - W2 的 devtune800、devcheck2；
     - devcheck3 只经 `--unseal-devcheck3 G-EX`；
     - 拒绝 test。
   - **设置**：
     - `x2_select`：提出 → 4 个候选（冻结的 W2 配置）→ 挑选，即 W2-c 的管线换成新检查点；
     - `x2_select_corR`：在 `x2_select` 的输出上做 R 轮修正，R = 1、2、3。用 grabcut 规则、软锁定、effect gate（`guidance` 的缺省）；模型输出 `[]` 就提前停；
     - `x2_spline_raster`、`x2_spline_trimap`：提出 → 描样条 → 光栅化，或作 trimap 种子跑 GrabCut；
     - `x2_select5`：4 个候选加 `spline_trimap`，共 5 个一起挑。挑选的提示与渲染沿用 W2，K 从 4 变 5；挑选没在 K = 5 上训练过，照实标注；
     - `x2_select5_corR`；
     - 诊断：最优候选（4 个、5 个）。
     - 第一个检查点的读数出来之前，提前停、轮数等参数都不调。
   - **消融**，证明能力来自 LLM，至少两个：
     - 修正框换成同数量、同面积分布的随机框；
     - 样条换成框内的椭圆。
   - **记账**：每句的调用次数、各阶段的 token、墙钟，同 W2；多轮的计算量与单轮配对。
   - **用时**：给 9B 在 explore_val 1,000 句上全部设置的估计。
5. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_x2_*.py tests/test_w2c_*.py -q`）：
   - 提示与解析：合法与各类非法输出、坐标换算与适配器一致；
   - 训练数据：样条拟合的上限、修正目标的规则、第 2 轮的构造、确定性、续跑；
   - 训练：tiny 上四流加两条回放流一起跑、续跑、默认配置逐字节不变、检查点能被 `run_x2` 与 `run_w2 --setting w2c` 加载；
   - 运行器：tiny 端到端（各设置）、提前停、拒绝 test、devcheck3 守卫、输出能被 `gate_explore` 读；
   - 不 import 学习型视觉模块。

## 不做

- 不改 W2 已冻结的配置与默认行为，不改 `src/w2/candidates.py`。
- 不改 `src/explore/guidance.py` 的缺省常量（D133 的配置测试钉住）；需要新参数时在 x2 的配置里给。
- 不在开发机上下载数据。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 提示原文；
- 数据规则与条数估计；
- 训练配置与用时估计；
- 运行器的设置与消融；
- 服务器命令：数据生成（CPU）、训练、在 explore_val 与 devtune800 上的评测；
- 测试结果原文；
- 与任务书的任何出入。
