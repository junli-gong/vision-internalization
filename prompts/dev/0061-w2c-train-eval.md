[DEV-TASK 0061] W2-c 的训练与评测接入：在 E3 的训练框架上加一种只有文本的监督"提出的框 + 挑选的编号"；run_w2.py 支持带 LoRA 检查点的 W2-c 设置（D113 第二节第 4 条；接口见 docs/w2_plan.md §5）

## 背景

- W2-c 在 9B 上做 LoRA 监督微调，两类文本目标：
  - 提出：E1 的定位提示加原图 → GT 框，与 C3 的框监督逐字相同；
  - 挑选：挑选图加挑选提示 → 编号文本。
- 文本监督参照 E3 的 C3：没有伤害语言，还有提升。W2-c 只训练文本形式的决策，不加任何新 token。
- 训练数据由并行的 DEV-0060 生成，格式见 `docs/w2_plan.md` §5。本任务先用 fixtures 造几条符合格式的假数据测试。
- 判定（D113 第三节）：
  - W2-c 是第二个主配置，最晚 11-01 判定，用 `scripts/w2_gate.py --main-name W2-c`；
  - W2-c 必须带检查点；
  - 另要做健康检查（简报 8.11），并报告定位与计数的 T 文本读数，证明语言没有受损。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w2_plan.md`（全文，§5 是本任务的约定）
- E3 的训练：
  - `scripts/train_e3b.py`、`src/e3/{train,data,c3,config}.py`、`configs/e3b_train.json`；
  - 尤其是 C3 的 text 监督：`box_target`、样本日程、`chunked_lm_loss`、LoRA、检查点格式、分段运行与续跑。
- E3 的评测：
  - `scripts/eval_e3b.py`：T 文本的评测怎样加载 C3 的检查点（合并核对 `src/e3/merge_check.py`）；
  - 健康检查 `configs/health_check_items.json` 与对应的代码。
- W2：`src/w2/{config,loop,runner}.py`、`scripts/run_w2.py`、`scripts/w2_gate.py`（W2-c 的核对：必须带检查点）。

## 任务

1. **训练**：
   - 新配置 `configs/w2c_train.json`：
     - LoRA 与优化器同 C3：r 32、alpha 64、lr 2e-4，学习率日程照 C3；
     - 每步 16 条：8 条"提出"加 8 条"挑选"，两种损失 1 : 1；
     - 步数默认 2,000，可配；种子 0；
     - `w2_config` 写死要用的冻结配置路径与 sha256，冻结前先放占位，由管理员填。
   - 在 E3 的训练框架里加一种监督 `w2c`，或者另写 `scripts/train_w2c.py` 复用 `src/e3/train.py` 的机制。哪种侵入小选哪种，理由写进完成总结。
     - **"提出"的样本**：DEV-0060 数据里的每一句，用 `box_target(GT 框)`，提示是 E1 refexp_box 的原文，与 C3 相同。
     - **"挑选"的样本**：`candidates` 不为空的句子。
       - 用冻结的 `render`（布局与样式）在线渲染：从 RLE 解码，按 W2 的规则，1 号失败时画空叠加；
       - 提示用冻结的挑选提示；
       - 目标是编号的十进制文本，例如 `"3"`；关思考，用非思考的对话模板。
       - 标签为 null 的句子不用于挑选。
     - 样本日程、续跑、分段运行（`--stop-after-steps`）、检查点（LoRA，与 C3 同格式）、`train_summary.json`、计算记账：都照 E3。
     - 只用训练集，不读任何评测集。
2. **评测接入**：
   - `scripts/run_w2.py` 加 `--checkpoint <LoRA 检查点>` 与设置 `w2c`。
   - W2-c 的提出与挑选都用加载了检查点的模型，都关思考。候选生成器、渲染、K 都用 W2-a 冻结的那一份；"都不对"或解析失败交 1 号。
   - 加载检查点时走 E3 的合并核对，口径同 E3 的 T 文本评测。
   - 运行身份记检查点的路径与 sha256。
   - W2-c 自己的"提出"一步，要用训练过的模型重新出框，输出放在同一个 `--out` 下与 base 的框分开的目录，不能与 base 9B 的 `baseline_e1` 混用。
   - 确认 `scripts/w2_gate.py --main-name W2-c` 能读这个运行，与同一 devcheck2 上的 `w1f`（base 9B）配对。
3. **健康检查与 T 文本读数**：
   - 写清怎样用现有的 E3 评测脚本，对 W2-c 的检查点跑：
     - 健康检查（简报 8.11）；
     - 定位（refexp_box）与计数（count）的 T 文本读数，口径同 E3 的 C3 对照表。
   - 能原样用就不写新代码，只在文档里写命令；不能原样用，就做最小的改动。
4. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w2c_*.py tests/test_w2_*.py tests/test_e3*.py -q` 里与改动相关的部分）：
   - tiny 训练：小型随机模型、CPU，几步就跑通，能续跑，检查点能被 `run_w2.py --setting w2c` 加载；
   - 挑选样本的渲染与 W2-a 评测时逐位相同；
   - 目标文本格式；
   - 只读训练集；
   - E3 现有的训练与评测行为不变，测试照常通过。

## 不做

- 不生成训练数据：DEV-0060 做。
- 不改 E3 已有的行为。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 训练的设计：新监督还是新脚本，及理由；
- 服务器上的训练、评测、健康检查命令与用时估计。按 9B 的 C3 实测：每步约 7.3 s，峰值约 28.6 GB。
- 测试结果原文；
- 与 `docs/w2_plan.md` §5 的任何出入。
