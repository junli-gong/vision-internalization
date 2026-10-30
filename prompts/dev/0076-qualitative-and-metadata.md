[DEV-TASK 0076] 只用已有输出：定性图（U-5）与 test 逐句元数据（U-6），两个只读脚本，在 HPC 上用 CPU 跑

## 背景

- 负责人要求冻结前做两件事（D154，原文 `docs/lead_answers/2026-10-30-lang-table-and-freeze.md`），都只用已有输出、不占 GPU：
  - 定性图；
  - 按划分、COCO 类别、形状复杂度的分档分析。
- 规则已写进预注册，**先读 `PREREGISTRATION.md` 的 U-5（抽样）与 U-6（分档）**。规则在写代码之前就定了，实现时一条都不改；有歧义的地方在总结里提出来，不要自己定。
- 要用的已有输出：
  - X-2（C1）在 test 上的 `x2_spline_trimap` 与 `x2_ellipse_trimap` 叶目录（RUN-0092，`experiments/X/0092_x2_test_hpcgpu06/`），逐句 mask 在 predictions.jsonl 里；
  - LLM 描的 B 样条在 `x2_trace/<划分>/trace.jsonl`。每个约 20 MB，**只在 HPC 上**，仓库里只有 `LARGE_FILES.manifest.tsv` 记的 sha256；
  - W2-c（RUN-0072）的逐句结果在仓库里。
- 开发机没有 RefCOCO 的图与标注，两个脚本都在 HPC 上用 CPU 跑，由服务器执行。开发机上用 `tests/fixtures` 的小数据测。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 U-1、U-5、U-6
- `src/explore/x2_runner.py`：trace 的格式、样条的光栅化、mask 的 RLE
- `scripts/run_w2.py` 与 `src/w2/`：test 划分的取样、图像与真值 mask 的加载
- `src/llmseg/runner.py`：子集的 `load_subsets`、`check_subsets`
- `scripts/figures/make_paper_figures.py` 与 `docs/figures/README.md`：配色、字体、尺寸、caption 的写法

## 任务

1. **逐句元数据 `scripts/analysis/test_metadata.py`**：
   - 句子：W2 test 全部 5,000 句，取样同 `run_w2`，逐 id 相同。
   - 每句输出：id、划分、COCO 类别 id 与名称、大类、图像尺寸、真值 mask 的面积 A（像素）、周长 P、c = P² / A、相对面积 A / (W·H)、连通块数。
     - 周长 P：外轮廓逐像素长度之和，用 OpenCV 的 `findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` 与 `arcLength(closed=True)`。
     - 类别从 RefCOCO 的标注取（ref → ann → COCO 的 category_id），不要猜。
   - 输出 `<out>/test_metadata.jsonl` 与 `summary.json`：句数、各类别与大类的句数、c 与相对面积的分位数、输入文件的 sha256。
   - `--out` 的最后一级目录名以四位运行编号开头。
2. **定性图 `scripts/figures/make_qualitative.py`**：
   - **抽样完全照 U-5**：
     - 每个划分在 300 句子集里用 `numpy.random.default_rng(20260929 + k)` 不放回抽 40 句，k 是划分在 `configs/w2_llmseg_subsets.json` 里的序号；
     - 随机例取抽中顺序的前 3 句；
     - 失败例取池里 X-2 逐句 IoU 最低的 2 句，同分按 id 字典序。
   - 抽样只用 predictions.jsonl 就能算，单独做成一个函数与一个 `--select-only` 模式，开发机上也能跑；输出 `selection.json`，含每条的 id、划分、类型、各系统 IoU。
   - **每条画四格**，每格标 IoU，标题是指代表达式：
     - 原图与 LLM 描的 B 样条，画控制点与曲线；
     - X-2 的最终 mask；
     - 椭圆对照的 mask；
     - 真值。
   - **输出**：
     - 附录图：25 条全部，按划分分页或分图；
     - 正文图：每个划分的第 1 个随机例与第 1 个失败例，共 10 条；
     - PDF 矢量，PNG 300 dpi；
     - 再给 caption（.tex），写抽样规则与数据来源。
   - 读 trace.jsonl 之前核对它与 `LARGE_FILES.manifest.tsv` 的 sha256 相同。
3. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/ -q -k "metadata or qualitative"`）：
   - 用 fixture 小数据算元数据：周长与面积对一个已知形状（矩形、圆）算对；
   - 抽样：同一输入两次结果相同；规则逐条对，可以用合成的 predictions 构造同分；
   - 在合成数据上出图，文件都生成；
   - `--out` 不带运行编号时拒绝。
4. 在总结里给出服务器命令：HPC 登录节点或 06 的 CPU，路径用 `$VI/b200/data` 与仓库里的结果目录；再给出用时估计。

## 不做

- 不改任何已有的结果目录、分析 JSON、配置、推理或计分逻辑。
- 不做分档的统计。那由管理员拿元数据另写脚本，按 U-6 做。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 实现中对 U-5、U-6 有歧义的地方；
- 服务器命令与用时估计；
- 测试结果原文。

尽量 2.5 小时内完成。
