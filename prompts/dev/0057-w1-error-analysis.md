[DEV-TASK 0057] W1 的错误分析：重放 W1 开发-检查集的每条样本，得到每一步的 mask，按四类给失败样本分类（D113-4；只用 CPU；10-24 前要结果）

## 背景

- G-W1（`reports/gates/G-W1.md`）：开发-检查集 500 句上，W1-1 0.470，自己的框 0.507，W1-F 0.576。
- 负责人要求（D113-4）：把 W1 开发-检查集里的失败样本分为四类，给出各类占比，用来设计 W2 的候选生成器：
  - 种子点放错；
  - 修改把好的结果改坏；
  - 没有提交；
  - 贴边算法本身的问题。
- W1 的数据可以随便看，不参与 W2 的判定。
- W1 的记录（`predictions.jsonl`）只存了最终提交的 mask，没存中间步骤。工作台逐位确定（DEV-0050 的测试钉住），所以按记录里的调用序列重放，就能得到每一步的 mask。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w2_plan.md`（§4）、`reports/gates/G-W1.md`
- `src/w1/workbench.py`、`src/w1/loop.py` 的记录格式（EpisodeRecord 与每轮的 `tool_calls`）、`src/w1/scoring.py`、`src/w1/data.py`、`src/w1/baselines.py`（自己的框与 W1-F 的记录）
- 数据：`experiments/W1/0063_w1_devcheck_hpcgpu07/qwen3.5-9b/refseg/{full,baseline_ownbox,w1f}/devcheck/predictions.jsonl`，以及 `refexp_box/baseline_e1/devcheck/predictions.jsonl`。图像与真值在 HPC 上，开发机只有 tests/fixtures。

## 任务

1. **重放**：`src/w1/replay.py` 与 `scripts/w1_replay.py`。
   - 对每条样本，用原图新建一个工作台（与运行时相同：`disabled_groups`、`view_frames` 取运行身份 `config.yaml` 里的值）。按记录的顺序，把 `executed = true` 的调用逐个 `Workbench.call(name, arguments)` 重放。被拒的与解析失败的调用不重放。
   - **一致性核对**：每次调用重放出的返回文字，必须与记录里的 `result.text` 逐字相同。提交的 mask 必须与记录里的 RLE 逐位相同。不一致就把这条标记为 `replay_mismatch`，报告条数，不进分类。
   - 每次调用之后，记下工作台里每个 mask 的 IoU（与真值）及其面积。
2. **每条样本的量**（写进 `replay.jsonl`）：
   - `final_iou`：最终提交的 IoU，没有有效提交时为 0；
   - `best_iou`：整个过程中任何一个 mask 的最高 IoU，以及它出现在第几次调用、是哪个 mask 名；
   - `seed_box`：第一次 `grabcut` 调用的 `box` 参数（原图 0–1000）。没有 grabcut 时，取第一个被创建的 mask 的外接框，并注明来源。
   - `seed_box_iou_gt`：`seed_box` 与真值框的 IoU；`seed_box_iou_own`：与自己的框的 IoU；
   - `first_snap_iou`：第一次贴边调用（grabcut、watershed、select_superpixels）产出的 mask 的 IoU；
   - `w1f_iou`、`ownbox_iou`，取自基线记录；
   - 失败码、轮数、调用序列（只记工具名）。
3. **分类**（`scripts/w1_replay.py` 的 `--classify` 输出 `error_analysis.{md,json}`）：
   - **失败样本**有两种定义，两套都报：
     - F1：`final_iou < 0.5`；
     - F2：`final_iou < w1f_iou − 0.05`，即比 W1-F 差。
   - 每条失败样本按下面的先后归入**唯一**一类：
     1. **没有提交**：失败码不为空（no_answer、wrong_submission、not_submitted 等）；
     2. **修改把好的结果改坏**：`best_iou − final_iou ≥ 0.10`，即过程中有过明显更好的 mask；
     3. **种子点放错**：`seed_box_iou_gt < 0.5`，即 LLM 给工具的起点就不对；
     4. **贴边算法本身的问题**：其余，即种子是对的，工具却没做出好的 mask。
   - 每类给条数、占比，以及各类里 `best_iou`、`seed_box_iou_gt`、`w1f_iou` 的中位数。
   - 另报描述性的量：
     - 全部 500 句 `seed_box_iou_own` 的分布，即写进工具的框比原生框粗多少；
     - "没有提交"那一类里，如果交了过程中最好的那个 mask，cIoU 会是多少；
     - 每种工具调用之后 IoU 的变化（升、降、不变的次数与中位变化）。
   - 阈值写成命令行参数，默认值如上，报告里写明。
4. **测试**（`conda run -n visint pytest tests/ -q` 里与 W1 相关的部分；`HF_HUB_OFFLINE=1`）：
   - 用 tests/fixtures 的小数据与一个脚本化的假回路记录，验证：
     - 重放与原记录逐字一致；
     - 不一致时会被标记；
     - 四类的判定顺序与边界值；
     - 输出文件齐全。
   - 再用 `run_w1.py --tiny` 真跑一小段，重放它的输出，0 条不一致。
5. **服务器上怎么跑**：写进 `scripts/w1_replay.py` 的文档字符串。
   - 只用 CPU，在 HPC 节点上经 `srun --overlap` 跑，`CUDA_VISIBLE_DEVICES=` 置空。
   - 读 `--run` 指向的 RUN-0063 目录，输出到 `--out`。
   - 估计用时。

## 不做

- 不改 `src/w1/workbench.py`、`loop.py` 等已有代码。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 重放与分类的规则，以及与本任务书的任何出入；
- 测试结果原文；
- 服务器上的命令与估计用时。
