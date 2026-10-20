[DEV-TASK 0052] 论文用图：E1 差距图、E2 逐层曲线、E3 对照表、干扰随尺寸的曲线。10-26 前要能直接放进论文；数据补齐（9B、r128）后重跑同一个脚本即可更新（D103 §五）

## 背景

- 论文按三段式组织（D103 §〇）：
  1. **诊断（E1、E2）**：LLM 知道东西在哪，画不出轮廓，而轮廓信息就在它的中间层。
  2. **内化路线（E3 加 9B）**：训练能教会 LLM 画，但会和语言争抢，这种干扰随规模缩小。
  3. **工具路线（W1）**：本任务不涉及。
- 负责人要求 10-26 前，把下面四件做成可以直接放进论文的版本（CVPR，英文标签）：
  - E1 差距图；
  - E2 逐层曲线；
  - E3 对照表；
  - 干扰随尺寸的曲线。
- 这是**只读结果、只出图表**的任务，不跑任何模型，不改任何实验代码与配置。

## 先读

- `CLAUDE.md`；`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `RESULTS.md`、`reports/gates/G1.md`、`reports/gates/G2.md`：E1、E2 的结论与数字口径
- E1：`experiments/E1/0027_e1_main/`（主评测，含各模型的 refexp_box、refexp_poly、grid16 等的 `metrics.json`）、`0030_e1_posthoc/`、`0033_e1_posthoc2/`（结构强制多边形、自己的框 `ownbox.{json,md}`）；`scripts/mgr/e1_summary.py`、`e1_table.py`
- E2：`experiments/E2/0029_e2_qwen/e2_summary.{json,md}`；`scripts/mgr/e2_summary.py`
- E3：
  - `scripts/mgr/e3_summary.py`；
  - 0.8B、4B 的评测目录（`experiments/E3/0050_hpc_eval_08b_hpcgpu07/`、`0052_hpc_eval_4b_hpcgpu06/`）；
  - 2B 的评测（`0046_e3_eval_2b_h1/` 等）；
  - `docs/analysis/e3_interference_by_size.{json,md}` 与生成它的 `scripts/mgr/interference.py`。
- 9B 的 E3 结果（RUN-0056，`experiments/E3/0056_hpc_9b_eval_hpcgpu06/`）与 0.8B 秩 128（RUN-0054，`experiments/E3/0054_r128_eval_hpcgpu07/`）**现在还没有**。脚本要在它们出现后自动纳入，没有时照常出图并在图注里写明缺。

## 任务

1. 开发机 conda env `visint` 里装 matplotlib（`pip install --no-cache-dir matplotlib`，不改已有的包；写明版本）。
2. **`scripts/figures/make_paper_figures.py`**：
   - 一个命令生成全部图表，输出到 `docs/figures/`，每张图出 PDF（矢量）与 PNG（300 dpi），表出 LaTeX（booktabs）与 Markdown 两种；
   - 另写一个 `docs/figures/README.md`：每张图的数据来源路径、口径、生成时间与 commit。
   - 数字一律从结果文件读，不在脚本里写死任何读数。
   - **E1 差距图**：Qwen3.5 0.8B、2B、4B、9B（以及 E1 里有的其他家族，单独一组或作为参照，按你看到的数据决定，写明理由）。
     - 同一个模型上，"知道在哪"的读数：定位 Acc@0.5、自己的框光栅化后的 cIoU；
     - "画轮廓"的读数：多边形 cIoU、grid16 gIoU、结构强制多边形。
     - 读数之间的差距要一眼可见，并附 CI。
     - 具体用哪几列，以 `RESULTS.md` 与 G1 的口径为准，写进 README。
   - **E2 逐层曲线**：线性探针在各层的读数（深度、法向、语义分割、指代分割），横轴是相对深度（层号 ÷ 总层数），每个尺寸一条线，标出第 0 层与最后一层。口径照 G2。
   - **E3 对照表**：0.8B、2B、4B（9B 有了就加一列组）。行是 C3、E3b+C3、E3b，以及基座，列是：
     - §13 的 8 个 T 文本读数；
     - 指代分割 cIoU 与深度 AbsRel（结构强制网格）。
     - 种子平均与标准差；表注写明机器（D93-1）与被合并核对拒绝而没有读数的种子（N37）。
   - **干扰随尺寸的曲线**：
     - 横轴是参数量（对数）；
     - 纵轴是语言干扰 L，副图是像素干扰 P_seg、P_depth 与相对干扰 R_seg；
     - 每个尺寸画种子平均与各种子的点。
     - 秩 128 的 0.8B 有了就另画一个标记。
     - 数据取 `scripts/mgr/interference.py` 的 JSON 输出，9B 需要时用同一个脚本生成。
3. 风格：
   - 统一字体与配色，对色盲友好；
   - 单栏宽度约 3.3 英寸、双栏约 6.9 英寸；
   - 坐标轴与图例用英文，与论文术语一致：C3、E3b+C3、E3b、grounding、outline、cIoU 等。
4. **测试**（`conda run -n visint pytest tests/ -q` 全部通过；`tests/test_paper_figures.py`）：
   - 用小的假结果目录跑一遍，检查每个输出文件都生成；
   - 读到的数字与假数据一致；
   - 缺 9B 或 r128 时不报错，并在 README 里写明缺。

## 不做

- 不跑模型，不改实验代码、配置与结果文件。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；每张图表用的数据路径与口径；
- 各图的 PNG 路径，管理员要看图；
- 测试结果原文；
- 数据里发现的任何不一致或缺口，比如某个尺寸缺某个读数。
