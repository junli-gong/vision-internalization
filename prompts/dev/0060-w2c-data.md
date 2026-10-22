[DEV-TASK 0060] W2-c 的训练数据：在 RefCOCO 系列的 train 划分上用 9B 出框、用 W2 的候选生成器出候选、用真值定标签（D113 第二节第 4 条；接口见 docs/w2_plan.md §5）

## 背景

- W2-c 训练 LLM 做挑选。数据由真值得到，不涉及任何学习型模型：
  - 同一个模型先用 E1 协议出框；
  - W2-a 冻结的候选生成器出 K 个候选；
  - 真值 IoU 最高的候选编号作标签。
- 训练与评测接入由并行的 DEV-0061 做。数据格式见 `docs/w2_plan.md` §5，按它对接。
- 候选生成器与渲染的最终参数，要等 W2-a 在 devtune800 上调参、冻结之后才定（约 10-24）。所以脚本读 `--w2-config` 指定的配置，默认 `configs/w2.json`，并要求 `frozen: true`；`--tiny` 不受限。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w2_plan.md`（全文，§5 是本任务的约定）、`docs/lead_answers/2026-10-22-w2-directive.md`
- `src/w2/{config,data,loop,runner,candidates}.py`、`scripts/run_w2.py`：怎样出框（复用 E1 的运行器）、生成候选（含 1 号失败当空的规则、`--candidate-workers` 的进程池）、算 IoU
- `src/data/train_loaders.py` 与 E3 训练池的去重：`dedupe/train_exclusions.json`、`configs/e3b_train.json` 的 dedupe 一节
- `src/data/subsets.py`（`hash_rank`）、RefCOCO 的加载器

## 任务

1. **`src/w2/w2c_data.py` 与 `scripts/build_w2c_data.py`**：
   - **总体**：RefCOCO（UNC）、RefCOCO+（UNC）、RefCOCOg（UMD）三者的 train 划分的全部句子。
   - **去重**：
     - 按 COCO 图像 id，排除 W2 全部评测集的图：`configs/w2_splits.json` 的 devtune800、devcheck2、5 个 test 子集；
     - 再沿用 E3 训练池的剔除清单。
     - 各步的条数与 sha256 写进 `manifest.json`。测试钉住"与任何评测图都不重叠"。
   - **抽样**：按 `hash_rank(salt='w2c_train_v1', purpose='w2c_train')` 升序取前 N 句，N 由 `--n` 给，默认 16,000。
   - **出框**：base 9B、E1 refexp_box 协议，即关思考、贪心、E1 冻结的批大小，复用 E1 的运行器，输出与续跑照 E1。
   - **候选**：用 `--w2-config` 的候选清单，复用 `src/w2` 的生成与"1 号失败当空"的规则，`--candidate-workers N` 进程池并行。
   - **标签**：各候选与 GT mask 的 IoU，`label` = IoU 最高的编号，并列取编号小的。
   - **输出**：按 `docs/w2_plan.md` §5 的格式写 jsonl。
     - 分片、可续跑：每片 1,000 句，写完一片才算数；
     - 除耗时外逐字节可复现。
   - 运行身份（config.yaml）要记：模型与 revision、w2 配置的 sha256、候选清单的 sha256、抽样规则、去重清单的 sha256。
   - 统计（`metrics.json`）：
     - 出框的失败率；
     - 候选各编号作为最优的次数；
     - 最优候选的平均 IoU；
     - 1 号（即 W1-F）的平均 IoU；
     - 标签的分布。
2. **服务器上的命令**：写进脚本的文档字符串。
   - 先出框：GPU，9B，用时估计。
   - 再出候选：CPU，`--candidate-workers 8`，用时估计；RUN-0065 实测每句约 4.5 s。
   - 两步可以拆开跑，也可以一条命令跑完。
3. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w2c_data.py tests/test_w2_*.py -q`）：
   - fixtures 上 tiny 端到端；
   - 去重，即与评测图不重叠；
   - 抽样可复现；
   - 标签的规则，包括并列与 1 号失败；
   - 续跑逐字节相同；
   - 配置未冻结时非 tiny 拒跑。

## 不做

- 不写训练与评测：DEV-0061 做。
- 不改 W1、W2 已有的行为：现有测试照常通过。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 数据格式与 `docs/w2_plan.md` §5 的任何出入；
- 服务器命令与用时估计；
- 测试结果原文。
