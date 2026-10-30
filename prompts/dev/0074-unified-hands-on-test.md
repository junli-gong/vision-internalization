[DEV-TASK 0074] 统一主表的两只"手"：9B 零样本的 E1 结构强制多边形、9B 的 E3b 像素 token 网格，在 W2 test 的 1,500 句子集上运行（D154 放行）

## 背景

- 负责人要一张统一主表（D154，原文 `docs/lead_answers/2026-10-30-lang-table-and-freeze.md`）：所有系统都在同样的 5 个划分 × 300 句上。
  - 这份子集就是 LLMSeg 用的 `configs/w2_llmseg_subsets.json`，1,500 句。
  - 加载、核对与取样的函数在 `src/llmseg/runner.py`：`load_subsets`、`check_subsets`、`test_splits`、`subset_samples`。
- 清单与计分写在预注册 U（`PREREGISTRATION.md` 末尾），先读 U-1、U-2。
- 已有输出的系统直接取子集，不用跑。这里只做清单里要新跑的两只"手"：
  1. **E1 结构强制多边形，9B 零样本**：即 `run_e1.py --subset forced_poly` 里的 `refexp_poly__poly_structured`（至少 3 个点，关思考，D72-3）。现在它只跑 E1 的思考子集。
  2. **9B 的 E3b 像素 token 网格，结构强制解码**：即 `eval_e3b.py` 在 E3 表里的 refseg 读法，单轮，guided 上采样（D83-3）。现在它只跑 E3 的评测集。
- 两者都要在 test 子集上跑，需要负责人的批准：编号 **D154**，写法同 `run_x2.py` 对 D147 的放行（DEV-0072）。
- 第三项新基线 LISA-7B 由另一个任务（DEV-0075）做，这里不碰。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 U 节；`DECISIONS.md` 的 D147、D154
- `scripts/run_e1.py`（forced_poly 一节）、`scripts/eval_e3b.py` 与 `src/e3/evaluate.py`
- `scripts/run_x2.py` 对 test 的放行（`--test-approval`）、`scripts/run_llmseg.py` 与 `src/llmseg/runner.py`（子集与 W2 同口径计分）
- `scripts/run_w2.py` 与 `src/w2/` 的计分、输出格式（predictions.jsonl 的 `score.inter/union/iou`，metrics.json 的 headline）

## 任务

1. **新运行器 `scripts/run_u_hands.py`**：`--system e1_poly_structured | e3b_grid`。
   - 也可以改为给现有脚本加参数，选改动更小、更不容易影响已有输出的做法，在总结里说明理由。
   - 句子：`configs/w2_llmseg_subsets.json`。先 `check_subsets`，再按 split 取样。
     - `--split <W2 的 test 划分名>|all`，`--limit N` 用于试跑，进运行身份。
   - **放行**：必须给 `--test-approval D154`，恰是这个编号，写成常量并注明依据（D154）。否则退出码 2，什么都不写。
     - 放行时把批准编号写进运行身份（config.yaml），并追加一行到 `<out>/<模型>/test_unlock.jsonl`，同 DEV-0072。
   - **推理照搬，不另写**：
     - E1 多边形：调用 forced_poly 现有的推理与解析函数，口径为 E1 冻结的图像口径（D28-3）、关思考、至少 3 个点。
     - E3b：调用 `src/e3/evaluate.py` 现有的单轮结构强制解码、guided 上采样。检查点用 `--checkpoint`，先做合并核对，同 eval_e3b。
     - 不改任何解码、阈值、提示。
   - **计分与输出**：用 W2 的计分（与 RUN-0072、RUN-0078 同一函数），输出 W2 的叶目录格式。
     - 路径为 `<out>/<模型>/refseg/<设置名>/<划分>/`，含 predictions.jsonl、metrics.json（cIoU、gIoU 与 CI，n、complete、failures）、compute.json、config.yaml、README.md。
     - 设置名为 `u_e1_poly_structured` 与 `u_e3b_grid`。
     - 失败（解析不出、空网格）按空 mask 计，失败数进 metrics.json 的 failures。
     - 这样 `scripts/mgr/` 里现有的读法（`w2_test_table.preds`）就能直接读。
   - `--out` 的最后一级目录名必须以四位运行编号开头（CLAUDE.md"服务器驱动的写法"）；`--resume`、`--tiny` 同其他运行器。
2. **其余行为不变**：`run_e1.py`、`eval_e3b.py` 在原有子集上的输出逐字节不变，快照或测试钉住。
3. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/ -q -k "u_hands or run_e1 or e3b or llmseg"`；能跑全量更好）：
   - 放行：缺编号、编号不对时拒绝；给 D154 时放行，并有 test_unlock 记录；
   - tiny 上两个系统都能出叶目录，格式与 W2 的叶目录同一套读法读得出（用 `scripts/mgr/w2_test_table.py` 的 `preds` 读一遍）；
   - 子集取样与 `src/llmseg/runner.py` 逐 id 相同；
   - `--out` 不带运行编号时拒绝；
   - 原有输出不变。
4. 在总结里给出服务器命令（真模型、06、5 个划分）：
   - E1 多边形用基座 9B；
   - E3b 用 `--checkpoint <RUN-0055 的 E3b seed 0 叶目录>/ckpt/final/model`，路径照 `experiments/E3/0056_hpc_9b_eval_hpcgpu06/eval/e3b/qwen3.5-9b/seed0/refseg__structured/config.yaml` 里记的写；
   - 再给出每个系统 1,500 句的用时估计及其依据。

## 不做

- 不改 `src/w2`、`src/e3`、`src/e1` 的推理与计分逻辑；不改任何配置与已有结果目录。
- 不碰 LISA（DEV-0075）。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件，以及为什么选新运行器或改现有脚本；
- 放行的条件与报错；
- 服务器命令与用时估计；
- 测试结果原文；
- 与预注册 U 不一致、或需要管理员定的地方。

尽量 2.5 小时内完成。
