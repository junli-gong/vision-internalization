[DEV-TASK 0075] 外部对照 LISA-7B：在 W2 test 的 1,500 句子集上单独运行与计分（D154；简报 §8.8 确认的 LISA 式对照系统）

## 背景

- 负责人要求统一主表的"外部方法"一组加 LISA-7B（D154，原文 `docs/lead_answers/2026-10-30-lang-table-and-freeze.md`）。
  - 它就是简报 §8.8 所说"经项目负责人确认的 LISA 式对照系统"的型号（D154-2）。
- **硬约束第 1 条**：LISA 把 SAM 接在 LLM 旁边，是"外接"的路线。它只能作为对照**单独运行**，输出绝不进入我们的训练数据、标签、奖励或任何挑选。
- 句子与计分写在预注册 U（`PREREGISTRATION.md` 末尾），先读 U-1、U-2。
  - 句子是 `configs/w2_llmseg_subsets.json` 的 1,500 句，与 LLMSeg 相同；计分与 W2 相同。
- 写法参照 LLMSeg 外部基线（DEV-0066）：官方代码原样引入 `third_party/`，记出处与逐文件 sha256，我们的包装与计分在 `src/` 里。
- 开发机没有 CUDA，跑不了 7B。真跑在 HPC 的 06 上（A100 80GB）。开发机上只用替身模型测流程。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 U 节；`docs/RESEARCH_BRIEF.md` §3.1 与 §8.8；`DECISIONS.md` 的 D147、D154
- `third_party/llmseg/PROVENANCE.md`、`src/llmseg/`（vendor、runner）、`scripts/run_llmseg.py`：引入、核对、子集、计分、输出的写法
- `scripts/run_x2.py` 对 test 的放行（`--test-approval`）

## 任务

1. **引入官方代码**：`https://github.com/dvlab-research/LISA`，钉一个 commit。
   - 只拷推理需要的文件：模型、LLaVA 与 segment_anything 的推理部分、官方 refer_seg 评测要用的预处理与模板，以及 LICENSE。
   - 不在它的目录里运行任何东西。
   - `third_party/lisa/PROVENANCE.md` 记来源、commit、许可证、逐文件 sha256，格式同 `third_party/llmseg/PROVENANCE.md`；运行前核对。官方文件一个字节都不改。
2. **协议**：照官方 refer_seg 评测，即其论文表格的口径。核对官方的 validate 代码，照抄以下各项，并写进 PROVENANCE 的"协议"一节，逐条注明出自哪个文件哪一行：
   - 提示模板与对话模板（含是否在回答里直接给出 `[SEG]`）；
   - 图像预处理：SAM 一侧与 CLIP 一侧；
   - mask 的阈值与还原到原图尺寸的方法；
   - 精度：bf16 或 fp16。
3. **权重**：`xinlai/LISA-7B-v1`，钉 revision；视觉塔照它的 config，通常是 `openai/clip-vit-large-patch14`。
   - 写一个下载清单（repo、revision、文件、sha256），由服务器在 HPC 登录节点下载，放在 `$VI/hf` 下，不碰别人的缓存。
   - 在总结里给出总大小。
4. **运行器 `scripts/run_lisa.py`**：
   - 在**单独的环境**里跑，因为 LISA 要旧版 transformers，不能装进我们的 `$VI/env`。
   - 给出 `third_party/lisa/requirements_lisa.txt`，全部钉版本，并写清在 HPC 上建 venv 的命令。
     - 首选：`--system-site-packages` 继承 `$VI/env` 的 torch，只在 venv 里装钉住的 transformers 等。
     - 备选：完整独立的 venv。
     - 说明为什么这样选。
   - 句子：`configs/w2_llmseg_subsets.json`，先 `check_subsets`；图像与指代表达式用与 W2 相同的加载方式。
     - 这个环境可能 import 不了我们的全部代码。数据读取可以分两步：先在我们的环境里导出一份清单（id、图像路径、表达式、原图尺寸），LISA 环境只读这份清单。
   - **放行**：必须给 `--test-approval D154`，写成常量并注明依据，否则退出码 2、什么都不写；放行时写运行身份与 `test_unlock.jsonl`，同 DEV-0072。
   - **输出**：每句的 mask（RLE）与原始信息，如生成的文本、有没有 `[SEG]`、前向次数、用时。
3. **计分**：在我们的环境里用 W2 的计分（与 RUN-0078 同一函数），把 LISA 的 mask 计成 W2 的叶目录格式。
   - 路径为 `<out>/<模型>/refseg/lisa_7b/<划分>/`，含 predictions.jsonl（`score.inter/union/iou`）、metrics.json（cIoU、gIoU 与 CI，n、complete、failures）、compute.json、config.yaml、README.md。
   - 没有 `[SEG]` 或没有 mask 的句子按空 mask 计，失败数进 failures。
   - 这样 `scripts/mgr/w2_test_table.py` 的 `preds` 能直接读。
4. **隔离**：
   - 结果目录单独，名字带运行编号，例如 `experiments/W2/NNNN_lisa_hpcgpu06`；
   - 加一个测试：`src/` 下任何生成训练数据、标签或奖励的代码都不 import `third_party/lisa`、`src/lisa`，也不读 LISA 的结果目录，例如静态扫描 import 与路径字符串；
   - 在 README 里写明硬约束第 1 条。
5. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/ -q -k "lisa or llmseg"`）：
   - 引入文件的 sha256 核对；
   - 放行：缺编号、编号不对时拒绝，给 D154 时放行并有记录；
   - 用替身模型（不加载权重，返回固定 mask）跑通"导出清单 → 运行 → 计分"，叶目录能用 `w2_test_table.preds` 读；
   - 子集与 `src/llmseg/runner.py` 逐 id 相同；
   - `--out` 不带运行编号时拒绝；
   - 隔离测试。
6. 在总结里给出 HPC 上的完整步骤：建环境、下载权重、导出清单、在 06 上用 `srun --jobid=157296 --overlap` 运行、计分。
   - 再给出 1,500 句的用时与显存估计，写明依据。

## 不做

- 不改我们的模型、训练、数据生成、计分逻辑；不改任何已有配置与结果目录。
- 不碰 E1 多边形与 E3b 网格（DEV-0074）。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 引入的 commit、许可证、文件清单；
- 协议的逐条出处；
- 环境与权重；
- 服务器步骤与用时估计；
- 测试结果原文；
- 风险，例如旧版 transformers 与现有 torch 的兼容性，以及备选方案。

尽量 3 小时内完成。
