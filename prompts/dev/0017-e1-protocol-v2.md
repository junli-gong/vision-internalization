[DEV-TASK 0017] E1 协议 v2：受限字母读数、格式与能力分开记、验收 v3（冻结批配置 + ±δ）、CountBench 人工问句、拒答诊断、检测开发图的完整标注

## 背景

项目负责人的第二批答复（D36）改了 E1 的协议，本任务把 E1 框架改成 v2。规格以 D36 的第 1、2、6 条为准，要点：

1. **受限字母读数**（D36-6）：
   - 选择题与点对任务（CV-Bench、EmbSpatial、BLINK、NYU 点对）的标题改为受限字母读数：只生成一个 token，限制在选项字母上，同一字母的带空格与不带空格写法都算。
   - 自由作答放在固定子集上作诊断，用现有的 `think:<任务>` 子集的 id，关思考，上限 1024。
2. **三列**（D36-6）：所有 T 通道任务都报告：
   - 严格读数（标题）；
   - 有效率（解析成功且没有拒答的比例）；
   - 条件读数（只在有效回答上计算）。
3. **验收 v3**（D36-1）：
   - 批配置（批大小、填充方向、排序）冻结为协议的一部分。
   - 验收两条：
     - (i) 同一配置重跑，逐位相同；
     - (ii) 两种不同的批组成之间，指标的配对差 CI 落在 ±δ 以内。δ：准确率类 2 个点，IoU 类 0.02。
   - 样本量：短输出 n ≈ 300（按输出类型合并任务），多边形与网格 n ≈ 500。
   - 两种批组成之间的差异记作"数值噪声底"，写进结果。
   - 不再以 batch-1 为参照；RUN-0012/0013 的 batch-1 对照照实保留在报告里。
4. **CountBench**（D36-2）：默认改用 big_vision 的人工问句。报告里注明这套问句出自 PaliGemma，Gemma 家族可能见过。
5. **Gemma 定位的拒答诊断**（D36-6）：
   - 按被指代物体的 COCO 类别（RefCOCO 的 ref 里有 category_id），分"人物"与"非人物"统计拒答率；
   - 另给出非人物对象上的条件读数。
6. **检测的开发图标注**（D36-1）：
   - 补落地开发子集所用 RefCOCO 图在 COCO `instances_train2014` 里的全部标注：Range 只取需要的部分，带清单。
   - 然后验收里检测也按判定结果比，即每张图在 IoU 0.5 下的 TP 与 FP。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D32、D33、D36
- `PREREGISTRATION.md` 的 E1 草稿
- `experiments/E1/0013_acceptance/accept_summary.md`
- 已合入的 `src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`、`src/data/`

## 任务

1. **受限字母读数**：
   - 用适配器的 `generate_batch(..., allowed_token_ids=<全部字母写法>, max_new_tokens=1)`。argmax 等价于"两种写法取较大的 logit"。
   - 另一个任务（DEV-0016）在加 `option_logits` 接口，那个能给出概率。E1 先用 `generate_batch` 这条路径；接口合入后，如果要加概率，再另行安排。
   - 字母集合取本题的实际选项。
   - NYU 点对与 BLINK 的选项是 (A)、(B)，按现有提示的选项写法。
   - 解析：把生成的 token 映射回字母，不经过文本解析。
2. **三列报告**：
   - metrics.json 与 README 的表都加"有效率"与"条件读数"两列。
   - "有效"= 解析成功，且没有被判为拒答（refusal 诊断规则见 D33）。
3. **验收 v3**：
   - `--acceptance` 改为：同一批配置跑两遍，看是否逐位相同；再换一种批组成（例如另一个批大小，或另一种样本顺序，规则写明）各跑一遍，按 ±δ 判。
   - 汇总表给出：数值噪声底（两种批组成之间的指标差与 CI）、逐位重跑是否相同、是否通过。
   - 冻结的批配置写进 `configs/e1_prompts.json` 或单独的协议文件，并进 sha256。
   - 短输出按输出类型合并任务凑够 n ≈ 300，合并规则写明。多边形与网格各用 500 条（RefCOCO 开发子集有 600 句）。
4. **CountBench** 默认改为 `countbenchqa`；启发式保留为可选。
5. **拒答诊断**：按人物与非人物分列；非人物上的条件读数。
6. **检测开发图的标注**：新增落地源（只做加法），并在验收里启用检测的判定结果比较。
7. 相应更新预注册需要的数字与说明：写在完成总结里，由管理员改预注册。

## 允许改动的文件

- `src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`、`tests/test_e1_*.py`
- `src/data/land/` 与 `scripts/data/land_e1.py`：只加检测开发图标注这一个源
- 对应的 fixture

**不许碰**：`src/adapters/`（DEV-0016 在改；只调用现有接口）、`src/health/`、`src/parsing/`、`src/metrics/`、`src/synth/`。

## 环境与验收

- 用 conda 环境 `visint`，不装包，不下载权重。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `bash scripts/pytest_server_parity.sh`
  - `run_e1.py --tiny` 跑全部任务
  - `--acceptance --tiny`
- 总结里写明：
  - 冻结的批配置；
  - 两种批组成的定义；
  - 短输出合并的规则；
  - 受限字母读数的字母写法；
  - 三列的定义；
  - 拒答诊断的规则。
