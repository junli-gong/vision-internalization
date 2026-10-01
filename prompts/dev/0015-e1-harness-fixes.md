[DEV-TASK 0015] E1 评测框架的修订：按开发子集的截断率调整生成上限、验收的三种一致口径、诊断补充、CountBench 人工问句（可选）

## 背景

RUN-0012 在 8 个模型的开发子集上跑了批处理验收，见 `experiments/E1/0012_acceptance/accept_summary.md`。暴露出三类问题：

1. **生成上限太小**。截断一律判失败（E1 协议），所以：
   - NYU 点对与 BLINK 在所有模型上 100% 截断，准确率为 0；
   - CV-Bench、EmbSpatial 也大量截断；
   - 原始输出显示，模型是先写解释、再给答案。

   各任务的截断率见汇总表第二节。
2. **验收判据**：框按"逐位相等"判一致，bf16 下坐标末位常变，定位在 8/8 模型上不达标，而指标层面一致。判据由项目负责人在 N15 决定；本任务先把三种口径都算出来。
3. **诊断**：多边形任务里 Qwen 常给出 `bbox_2d` 或 `point_2d` 而不是多边形；Gemma 对"detect {指代表达}"常拒绝作答。这些是模型行为，照实记作解析失败，但要能分开统计。

协议允许依据开发子集上的截断率调整 `max_new_tokens`，不看准确率（E1 预注册草稿 §3），开发子集也可以用来开发解析器（D11-6）。本任务只做这些。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 E1 草稿，`DECISIONS.md` 的 D28-5、D32
- `DECISIONS_NEEDED.md` 的 N15
- `experiments/E1/0012_acceptance/`：汇总表与各模型的 `predictions.jsonl`，看原始输出
- 已合入的 `src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`

## 任务

### 1. 调整 `max_new_tokens`（只依据截断率）

在 `configs/e1_prompts.json` 里修改下面这些值，并在调整记录里逐条写下：原值、新值、依据（RUN-0012 开发子集上的截断率，按模型列出）、日期。

| 任务 | 原值 | 新值 |
|---|---|---|
| refexp_box | 128 | 256 |
| count | 64 | 256 |
| ovd | 512 | 2048 |
| rel_depth_nyu、rel_depth_blink、spatial_cvbench、spatial_embspatial | 64 | 1024 |

- refexp_poly 与 grid16 保持 1024。截断率 ≤ 6%；Qwen 小模型在 grid16 上写满 1024，是退化的输出，不是预算不够。
- 配置的钉死 sha256 相应更新。

### 2. 验收的三种一致口径（N15）

`--acceptance` 同时计算并报告三种口径：
- **(a) 逐位**：现行规则（D32）。
- **(b) IoU ≥ 0.9**：框与多边形、网格一样，两个预测的 IoU ≥ 0.9 算一致。
- **(c) 按判定结果**：
  - 定位：IoU 0.5 下两遍对错一致；
  - 检测：每张图在 IoU 0.5 下的 TP 与 FP 计数相同；
  - 多边形与网格：仍用 IoU ≥ 0.9；
  - 计数与多选：值相等；
  - 两遍都解析失败算一致。
  - 检测需要 GT。开发子集用的是 RefCOCO 图，核对 RefCOCO 落地的 `instances.json` 是否含这些图的全部 COCO 标注。含就用作检测的 GT；不含就写明，检测的 (c) 记 null。

- 新增 `--agreement {bitwise,iou90,outcome}`，决定 `all_pass` 用哪种口径，默认先用 `bitwise`（D32），项目负责人答复 N15 后再改默认值。
- 汇总表按三种口径各给一列一致率与是否通过。

### 3. 诊断补充

- **多边形**：解析失败为 `malformed`、且 JSON 里是 `bbox_2d`、`box_2d`、`point_2d` 一类的键时，诊断里单列为 `wrong_shape:<键名>`。只作诊断，失败原因不变。
- **拒绝作答**：回答里没有任何坐标或数字、且含 "I'm sorry"、"cannot" 一类句式的，诊断里单列为 `refusal`。
  - 规则要写明，正则固定；
  - 只作诊断，不改变失败原因。
- **关思考时自发写出思考标记**：InternVL 在关思考时偶尔会写 `<think>`。单列计数。

### 4. CountBench 的人工问句（可选，默认关）

- big_vision 仓库（Apache-2.0）的 `big_vision/datasets/countbenchqa/data/countbench_paired_questions.json` 里有人工写的问句。
- 在数据落地脚本里加一个源 `countbenchqa_questions`：按固定的 commit 下载这个文件，写清单。这一处允许改 `src/data/land/` 与 `scripts/data/land_e1.py`，只做加法。
- E1 框架加 `--count-questions {heuristic,countbenchqa}`，默认 `heuristic`，项目负责人批准后再改。用人工问句时，提示模板不变，只是 `{object}` 换成问句里的对象，或直接用整句问句；二选一，写明。
- 先在开发机上用 fixture 测通；真实文件在服务器上落地。

## 允许改动的文件

- `src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`、`tests/test_e1_*.py`
- `src/data/land/` 与 `scripts/data/land_e1.py`：只加 `countbenchqa_questions` 这一个源
- 对应的 fixture

**不许碰**：`src/adapters/`、`src/parsing/`、`src/metrics/`、`src/health/`、`src/synth/`。

## 环境与验收

- 用 conda 环境 `visint`，不装包，不下载权重。
- 验收：
  - `conda run -n visint pytest tests/ -q` 全绿；
  - `bash scripts/pytest_server_parity.sh` 全绿；
  - `run_e1.py --tiny --acceptance` 对三种口径各出一张表。
- 总结里写明：
  - 调整记录全文；
  - (c) 口径对检测是否可算；
  - 诊断规则的正则；
  - CountBench 人工问句的用法。
