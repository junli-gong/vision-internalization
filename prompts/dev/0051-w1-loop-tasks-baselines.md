[DEV-TASK 0051] W1：回路（按轮批量）、任务与数据划分、评分与计算记账、两条去留基线与 W1-F，以及 `scripts/run_w1.py`（D103；接口约定见 `docs/w1_plan.md` §0、§3）

## 背景

- W1（D103）：冻结的 LLM 在多轮回路里调用不含学习的工具（工作台，DEV-0050），通过适配器的多轮对话接口（DEV-0049）。
- 主任务是指代分割；另有定位与 CountBench 计数。
- 去留判定约在 10-27/28（D103 §三，写于任何 W1 结果之前，不得修改）。判定点是开发-检查集 500 句上 Qwen3.5-9B 的 W1-1，两条都满足才继续：
  1. W1-1 的 cIoU 显著高于同一模型"自己的框"基线：配对 bootstrap 的 95% CI 下界 > 0。
  2. W1-1 的 cIoU 比同一模型在 E1 结构强制多边形上的 cIoU 至少高 0.2。
  - 两条基线都要在这 500 句上，用同一个模型、同一套协议补跑。
- DEV-0049、0050 与本任务**并行开发**。本任务按 `docs/w1_plan.md` 的签名对接；测试里用假适配器与假工作台。合并后由管理员做集成彩排。

## 先读

- `CLAUDE.md`；`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w1_plan.md`（全文）；`DECISIONS.md` 的 D103；`docs/lead_answers/2026-10-20-w1-directive.md`
- E1 的数据、提示、评分：
  - `src/data/loaders.load_refcoco`、`load_countbench`；
  - `src/e1/tasks.select`；`src/data/eval_subsets.py`（`POLY_SPLITS`）；`configs/e1_eval_subsets.json`；
  - `src/data/subsets.seeded_sample`；`src/data/constants.EVAL_SUBSET_SALTS`；
  - `configs/e1_prompts.json`（`PromptConfig.render`）；
  - `src/e1/scoring.score_generation`；
  - `src/metrics/seg.py`（`mask_inter_union`、`sample_iou`、`ciou`、`giou`）；`src/metrics/box.py`；`src/parsing`（`parse_count`、Qwen 的原生框解析）；
  - `src/stats.paired_bootstrap_ci`（配对 cIoU 用 `statistic=ratio_of_sums`）；
  - `src/accounting.Compute`。
- "自己的框"：`scripts/summarize_e1_posthoc.py` 的 `box_polygon`、`own_box_record`、`match_boxes`。
- 结构强制多边形：`src/e1/poly_forced.py`（`PolyGrammar`、`PolyStructureProcessor`）、`src/e1/posthoc_runner.run_forced_poly_task`（现在只认 think 子集）、`configs/e1_posthoc.json` 的 `format_forced_poly`。
- E1 的输出布局与续跑规则：`scripts/run_e1.py`、`src/e1/runner*`。

## 任务

1. **`configs/w1.json`**（新）：回路协议，即 `docs/w1_plan.md` §3 的 `Protocol`。另有：
   - 批大小；
   - 系统提示：对所有任务相同，只说明工作台的用法与坐标约定，不写任何任务的解法；
   - 按任务的提示：只说明任务是什么、输出成什么格式；
   - 数据划分的 salt。
   - 提示词的初稿写进配置，标为"开发-调参集上待定"。负责人要求在开发-调参集上定稿、跑 test 之前冻结。
2. **`src/w1/data.py`**：
   - `w1_items(task, split) -> list[Sample]`，划分如下：
     - `devtune`：从 `poly:refcoco_unc_val`（1000 句）按 sha256 排序（salt `w1_dev_v1`），取前 300；
     - `devcheck`：同一排序的第 301–800 句，与 devtune 不重叠；
     - `test_<ds_split>`：E1 的 `poly:<ds_splitBy_split>` 子集（每个划分 1000 句），划分为 refcoco testA/testB、refcoco+ testA/testB、refcocog test；
     - CountBench：E1 的 eval 条目（442 条）。
   - 去重照 E1。
   - 划分的 id 清单写进 `configs/w1_splits.json`，附 sha256，测试钉住。
3. **`src/w1/loop.py`**：`Protocol`、`EpisodeRecord`、`run_episodes(adapter, samples, task, protocol, prompts, *, batch_size)`，按轮同步：
   - 第 1 轮：system 加 user（任务提示与原图）。
   - 之后每轮：
     - 把还没结束的样本拼成一批，调 `adapter.chat_turn_batch`；
     - 逐条执行作答段里的工具调用（`Workbench.call`），把结果作为 tool 消息（文字加图）追加；
     - 提交了答案（`submit_*`），或作答段没有工具调用而给出了最终答案，这条样本就结束。
   - **上限**：
     - 最多 6 轮、8 次工具调用。超过的调用不执行，返回说明"已到上限，请给出最终答案"。
     - 第 6 轮还没有答案，记失败（`failure = "no_answer"`）。
     - 每轮思考最多 2,048 个 token，到上限强制收尾。
   - **出错时回到原图重画**：系统提示里这样要求，工具里也提供从原图重来的方式。回路本身不做任务相关的修补。
   - **最终答案**：
     - refseg：提交的 mask；
     - refexp_box：`submit_box`，或作答段里 Qwen 原生的 `bbox_2d` JSON；
     - count：`submit_count`，或作答段里的单个整数（E1 的 `parse_count`）。
   - **记录**：每条样本记每轮的思考与作答原文、`thinking_end`、工具调用与结果摘要（图只记尺寸与 sha256，不存图）、最终答案、失败原因。
   - **compute**：提示、生成、补入的 token 数，视觉 token 数，图数，轮数，工具调用数，墙钟，用 `src/accounting.Compute`。
4. **评分**：
   - 用 E1 的函数：refseg 是 cIoU 与 gIoU，失败记 0；refexp_box 是 Acc@0.5；count 是准确率。
   - `metrics.json` 含 bootstrap CI。
   - 另有配对比较的工具函数：两种设置在同一批 id 上的配对 cIoU 差及其 CI（`paired_bootstrap_ci`，`statistic=ratio_of_sums`），给去留判定用。
5. **基线**（`src/w1/baselines.py`；都用同一个模型、E1 的协议，即关思考、贪心、E1 冻结的批大小）：
   - **W1-0**：E1 零样本，即 refexp_poly、refexp_box、count，在 W1 的划分上跑。
   - **自己的框**：refexp_box 的预测光栅化成 mask，用 `own_box_record` 的规则，算 cIoU。
   - **E1 结构强制多边形**：`poly_structured` 变体（`format_forced_poly`）在 W1 的划分上跑。
     - 放宽 `run_forced_poly_task` 对 think 子集的硬要求，允许 W1 的划分；
     - E1 原来的路径与输出保持逐字节不变，用测试证明。
   - **W1-F**（固定流程）：自己的框直接交给 GrabCut（框初始化，迭代次数固定，用工作台的同一个函数），没有 LLM。
   - 输出布局与 W1 的运行相同，便于配对。
6. **`scripts/run_w1.py`**（CLI）：
   - 参数：`--model`、`--task refseg|refexp_box|count`、`--split devtune|devcheck|test_*`、`--setting full|no_eye|no_check|no_snap|baseline_e1|baseline_ownbox|baseline_poly_structured|w1f`、`--out experiments/W1/<NNNN_slug>`、`--batch-size`、`--limit`、`--resume`、`--tiny`。
   - 运行身份写进 `config.yaml`：模型 revision、协议、提示配置的 sha256、划分 id 清单的 sha256、工作台工具清单的 sha256。续跑时必须相同。
   - `--resume` 按 id 续跑。
   - `--tiny` 用 tiny 模型与 fixture 数据，供开发机彩排。
7. **`scripts/w1_gate.py`**：去留判定的计算，照 D103 §三原样实现，规则写死，不留参数。
   - 读 W1-1、自己的框、E1 结构强制多边形三个运行（同模型、同 500 句），给出两条条件各自的读数与是否满足，以及结论"继续 / 放弃"。
   - 三个运行的 id 集合必须相同，不同就报错。
8. **测试**（`conda run -n visint pytest tests/ -q` 全部通过；`tests/test_w1_*.py`）：
   - 划分：300 与 500 不重叠；清单与 sha256 稳定；test 划分与 E1 的 poly 子集逐 id 相同。
   - 回路：用假适配器（按脚本返回 `ChatTurnResult`）与假工作台。
     - 上限（轮数、调用数）照规则生效；
     - 按轮批量时，样本先后结束，批正确收缩；
     - 提交、无答案、解析失败各自记录正确；
     - compute 字段齐全。
   - 评分：手算的小例子上，cIoU、gIoU、Acc@0.5、计数准确率与 E1 的函数一致；配对 CI 的方向对。
   - 基线：
     - 自己的框光栅化与 `own_box_record` 一致；
     - W1-F 在合成图上跑通；
     - 放宽之后的结构强制多边形路径，在 E1 原有的 think 子集上输出与改动前逐字节相同（快照）。
   - `w1_gate.py`：构造的三组读数分别得出"继续"、"放弃（条件 1 不过）"、"放弃（条件 2 不过）"。
   - CLI：`--tiny` 端到端跑通（子进程），用假工作台或 tiny 的真工作台；`--resume` 续跑与一次跑完逐字节相同。

## 不做

- 不改适配器（DEV-0049）与工作台、沙箱（DEV-0050），只按约定调用。
- 不改 E1、E3、E5 的配置与输出。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；回路的规则；划分（附 id 清单的 sha256）；基线的实现；
- 测试列表与结果原文；
- 提示词初稿全文；
- 对 `docs/w1_plan.md` §3 约定的任何偏离或建议；
- 每种设置在 9B 上处理 500 句的粗略用时估计（需要 DEV-0049 的速度估计时，写明假设）。
