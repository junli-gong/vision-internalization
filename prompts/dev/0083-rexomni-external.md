[DEV-TASK 0083] Rex-Omni 外部参照的运行器：只做检测，在 X-M 的最终报告集与 LVIS 子集上，用与 X-M 相同的 F1 计分（D165 第四节；D177）

## 你的 worktree（不是 Agent 自动建的那种）

- 路径 `/Users/tim/Research/CVPR2027/.claude/worktrees/dev0083`，分支 `dev0083`，建在 origin/main 671599d77。
- 稀疏签出，不含 `experiments/`：开发机磁盘只剩约 6 GB。**不要下载 Rex-Omni 的权重**，开发机上只用 tiny 随机模型。
- 所有命令都先 `cd` 到这个路径。不要碰主仓库 `/Users/tim/Research/CVPR2027`，也不要碰 `dev0082`，那里有另一个开发 agent。
- 不 commit、不 push。
- 测试：`cd <worktree> && HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/ -q -k "<相关>"`。
- 交回前跑一次全量。

## 先读

- `CLAUDE.md`：硬约束第 1 条，学习型 CV 模型只能作单独运行的对照系统，输出绝不进入我们的数据或奖励。
- `.claude/skills/exp-triad-flow/references/dev.md`
- `docs/lead_answers/2026-11-03-xm-directive.md` 第四节
- `docs/references/xm_refs_notes.md` 的 Rex-Omni 一节：许可证、权重、提示、温度、只问正类别
- `PREREGISTRATION.md` 的 XM-3a（F1 的口径）
- `DECISIONS.md` 的 D177
- 先例：LISA 外部对照（DEV-0075）的隔离，即 `src/lisa/`、`scripts/run_lisa.py`、`tests/test_lisa_isolation.py`、`tests/test_lisa_vendor.py`
- X-M 的叶目录格式与计分：`src/xm/runner.py`、`src/xm/metrics.py`（`eval_pair`、`table`、`point`、`paired_bootstrap`）、`scripts/run_xm.py`（gate 与密封）

## 任务

### 1. 官方推理照搬

- 读 Rex-Omni 官方仓库（github.com/IDEA-Research/Rex-Omni）的推理代码与模型卡，照搬以下几项，并在代码注释里写明出处（文件与 commit）：
  - 检测的提示原文；
  - 图像预处理（min/max pixels 或缩放规则）；
  - 生成参数：温度 0、max new tokens 等；
  - 输出的解析：坐标的量化网格与特殊 token 格式，换算回像素框。
- 权重 `IDEA-Research/Rex-Omni`（Qwen2.5-VL-3B 底座）。用 transformers 的 Qwen2.5-VL 类加载。HPC 上环境的 transformers 版本是 Qwen3.5 那一档，应该支持 Qwen2.5-VL。
- 能不装新包就不装。必须装的话，写明包名与版本，只装纯 Python 的，并且 `--no-deps`。
- 许可证：IDEA License 1.0，限非商业研究。我们不再分发权重或输出图。

### 2. 隔离（同 LISA）

- 代码放 `src/rexomni/` 与 `scripts/run_rexomni.py`，运行目录名必须含 `_rexomni`。
- 静态测试，同 `tests/test_lisa_isolation.py`：生成训练数据、标签、奖励与训练的代码，都不 import、不引用 Rex-Omni 的模块、脚本与结果目录。
  - 正对照：模式确实能在 Rex-Omni 一侧找到。
- Rex-Omni 一侧只 import 它需要的东西。可以用 `src/xm/data`（读对）与 `src/xm/metrics`（计分），这是读真值与计分，不是训练；不 import 我们的训练代码与 X-M 的提示、后处理。

### 3. 运行器 `scripts/run_rexomni.py`

- 划分：
  - `xm_tune`：只作冒烟，与它论文里 COCO 的量级对一下；
  - `xm_final`、`lvis_sub`：要 `--xm-gate <gate_xm.json>`，走 src/xm/gate 的基座那条路，不比 manifest，同 run_xm 的 base_* 设置。
  - 不配 `xm_decision`：密封的，已经判完了，拒绝。
- 每个（图像, 类别）调用一次：提示里只给这一个类别，只问图中出现的类别，同 X-M 与 Rex-Omni 的协议。最多保留 100 个框，同 XM-3a。
- 输出叶目录与 X-M 的同格式：`<out>/rex-omni/xm/rexomni_det/<split>/`，内含 predictions.jsonl、metrics.json、config.yaml、meta.json、README.md。
  - 每行的 det 用 `src/xm/metrics.eval_pair` 算，seg 为 null。
  - 这样管理员可以用同一套 `M.table`、`M.paired_bootstrap` 与 X-M 配对比较。
- 不加 NMS，不改它的输出顺序：Rex-Omni 的官方协议没有，F1 按输出顺序贪心匹配。如果官方代码有去重，照它的做法，写明出处。
- 断点续跑（`--resume`，身份相同才续）；`--out` 末级目录名以四位运行编号开头且含 `_rexomni`；`--tiny` 用随机的小 Qwen2.5-VL 配置，在 CPU 上跑通。
- 记录解析失败数、截断数、每对的生成 token 数与用时。

### 4. 测试

- 解析：用官方格式的样例字符串（照官方仓库的示例），覆盖多框、空、格式错、越界；
- 坐标换算的往返；
- 隔离的静态测试；
- 划分与 gate 的拒绝：`xm_decision`、缺 `--xm-gate`、彩排的 gate；
- tiny 端到端：用 `tests/_xm_run_fakes` 的合成对，跑通并写出叶目录，指标能被 `src/xm/metrics.table` 读。

## 不做

- 不改 X-M、X-2 的任何代码、配置与结果。
- 不下载真权重。
- 不写服务器驱动。

## 完成总结要写

- 改了哪些文件；
- 照搬了官方的哪几处，附出处与 commit；
- 依赖与环境：要不要装包；HPC 上权重怎么下载（hf-mirror），以及 sha 怎么记；
- 给管理员的服务器命令行：先在 xm_tune 上冒烟，再跑 xm_final、lvis_sub；
- 用时估计；
- 测试结果原文；
- 待裁的地方。

尽量 3 小时内完成。
