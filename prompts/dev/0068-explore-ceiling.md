[DEV-TASK 0068] 探索 X-1：各种"LLM 输出通道"的上限（真值模拟，CPU），以及可复用的引导工具

## 背景

- 负责人授权自主探索（D126-3），见预注册 X 与 `docs/exploration/log.md`。
- X-0 的结论：W2-c 在 devcheck2 上的损失 58% 出在框已经很好的句子上（框 IoU ≥ 0.9，347 / 500 句）。
  - 这些句子上，W2-c 0.70，最优候选 0.71；单个生成器只有 0.59–0.64。
  - 瓶颈是从框到 mask。
- 下一步要知道：如果 LLM 给出完全正确的形状信息，确定性工具能把 mask 推到多高。候选的信息有：
  - 前景/背景点；
  - B 样条轮廓；
  - 带符号的局部修正框。

  哪个通道的上限高、需要的 token 少，就先在那个通道上投 GPU。
- 本任务只做工具和真值模拟，不跑任何模型。所写的工具之后会给真正由 LLM 驱动的方法复用，所以要写成干净的库。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 预注册 X，`docs/exploration/log.md`（X-0、X-1）
- `docs/references/llmseg_notes.md`：B 样条的参数化与光栅化（附录 B.2）、带符号局部修正、软锁定（公式 4）、effect gate、提前停止
- `src/w2/candidates.py`（现有的生成器与 GrabCut 的用法；不含学习的工具清单）
- `src/w2/data.py`（`w2_items`）
- `src/w2/loop.py`、`src/w2/runner.py`（候选的进程池：spawn，子进程单线程）
- 计分的 `mask_inter_union`

## 任务

1. **工具库 `src/explore/guidance.py`**：只用不含学习的工具，即 numpy、scipy、scikit-image、OpenCV 的非 DNN 部分；写一个测试钉住不 import 任何学习型模块。
   - **点 → GrabCut**：给框与 k 个前景点、k 个背景点：
     - 框外为确定背景，框内为可能背景；
     - 前景点处半径 r 的圆为确定前景，背景点同理；
     - 跑 GrabCut。r 与迭代次数是参数。
   - **B 样条**：
     - 拟合：mask 的外轮廓 → n 个控制点的闭合周期均匀三次 B 样条（最小二乘），n 是参数；
     - 光栅化：采样 m = min(2000, max(64, round(2(w+h)))) 个点，偶奇填充，与 LLMSeg 附录 B.2 相同；
     - 框约束：采样点逐坐标夹到框内；
     - 可选洞：内轮廓同法拟合后减去。
   - **B 样条 → GrabCut**：
     - 样条内为可能前景，样条外为可能背景；
     - 轮廓两侧宽 w 的带为未知，按 GrabCut 的约定取值；
     - 跑 GrabCut。
   - **带符号的局部修正**：给当前 mask 与若干 (框, 符号)：
     - "+" 在框内把未承诺的像素按局部 GrabCut 或直接填的规则补；
     - "−" 同理删；
     - 带软锁定的承诺场与 effect gate，照 LLMSeg 的公式 4 与附录 B.2；
     - 同一轮先删后补。
     - "补"和"删"的像素怎么定，是工具的事，不看真值；规则写进模块说明。
   - **生成器库 v2**：在 `src/w2/candidates.py` 之外另写，不改它。包括：
     - GrabCut 的参数变体（迭代、框的内缩、GMM 的分量数、颜色空间）；
     - 超像素合并（SLIC、Felzenszwalb）后在框内按与框边的连通性或颜色取前景；
     - 分水岭的变体；
     - 形态学后处理的组合。
     - 每个都有名字与参数，可以按名字列表组成候选池。
2. **真值模拟的"理想 LLM"** `src/explore/oracle_sim.py`：从真值 mask 按确定性规则生成各通道的输入，并给可选的噪声档，模拟不完美的 LLM。
   - **点**：
     - 前景点：真值 mask 距离变换的极大值，依次取，取后挖掉邻域；
     - 背景点：框内、真值外区域的同法；
     - 噪声：坐标抖动 σ（像素，或框尺寸的比例）、错点率。
   - **B 样条**：真值外轮廓的 n 点拟合。噪声为控制点抖动 σ。
   - **局部修正**：当前 mask 与真值之差的连通块，按面积从大到小取前 m 个的外接框：
     - 漏掉的块为"+"，多出的块为"−"；
     - 每轮重算，1、2、3 轮；
     - 噪声：框的抖动、符号错误率。
   - **token 估计**：每个通道的输入按我们的文本格式估输出 token 数。坐标用 0–1000 网格的整数，与 E1 相同；用适配器的 tokenizer 实数，tiny 时用 tiny 的 tokenizer。
3. **上限测量脚本 `scripts/explore_ceiling.py`**：
   - **划分**：devtune800 与 devcheck2，`w2_items` 取样本；以后 explore_val 生成了也能跑。
   - **框**，三种：
     - 真值框；
     - W2-c 的框：devcheck2 取 RUN-0069 的 `refexp_box/propose_w2c/devcheck2`；
     - 基座的框：devtune800 取 RUN-0070 sep_on_k4 的 `refexp_box/baseline_e1/devtune800`，devcheck2 取 RUN-0069 的 `baseline_e1`。
     - 路径用参数给；读框的格式同 W2 的提出。
   - **起点 mask**（修正通道用）：
     - 1 号候选 gc_default；
     - devcheck2 上另用 W2-c 的最终 mask，即 RUN-0069 `refseg/w2c/devcheck2` 的 answer.mask。
   - **网格**：
     - 点：k ∈ {1, 2, 4, 8}，有无背景点；
     - B 样条：n ∈ {8, 12, 16, 24}，直接光栅化与作 GrabCut 种子两种；
     - 修正：轮数 ∈ {1, 2, 3}，每轮 m ∈ {1, 2, 4}，有无软锁定；
     - 生成器库：K ∈ {4（现有）, 8, 12, 16} 的最优候选，按一个固定的名字顺序加入；
     - 噪声档：无、轻、中。具体值写进配置 `configs/explore_ceiling.json`。
   - **输出**（`experiments/X/<运行编号>_ceiling_<节点>/`）：
     - `predictions.jsonl`：每句每个格点的交、并与 IoU，不存 mask；
     - `metrics.json`：每个格点的 cIoU 与 95% CI，按框 IoU 分档（≥ 0.9 与 < 0.9）各报一次，加 token 估计的均值；
     - 一张 markdown 汇总表；
     - `compute.json`：CPU 墙钟。
   - 进程池同 W2：spawn，子进程单线程，`--workers N`。可续跑，除耗时外逐字节可复现。
   - **用时估计**：开发机上用 tiny 测每句每格点的 CPU 时间，按 1,300 句估 HPC 上 8 核与 16 核的墙钟。写进脚本的文档字符串。格点太多就给一个先跑的子网格。
4. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_explore_*.py -q`）：
   - 每个工具在合成图上的行为：
     - 圆与矩形的真值，B 样条拟合的 IoU 随 n 上升；
     - 框约束；
     - 软锁定使反向的修正只消耗承诺、不翻转；
     - effect gate；
   - 不 import 学习型模块；
   - oracle_sim 的确定性与噪声档；
   - tiny 端到端：`scripts/explore_ceiling.py --tiny`；
   - 续跑逐字节相同。

## 不做

- 不改 W2 的代码与冻结的配置，不改 `src/w2/candidates.py`。
- 不碰 test 与 devcheck3：脚本拒绝 test_* 与 devcheck3 的划分名，测试钉住。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 各工具的规则与参数；
- 网格与噪声档；
- token 估计的方法；
- 服务器命令与用时估计；
- 测试结果原文。
