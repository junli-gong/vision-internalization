[DEV-TASK 0042] E3b+C3 混合训练（文本样本流与 C3 逐样本相同）；C3+预热（在 C3 之上做网格格式预热，存成单个 LoRA）

## 背景（D83，负责人 2026-10-08 的答复，原文 `docs/lead_answers/2026-10-08-n31.md`）

- RUN-0038（2B 种子 0）：只用网格或稠密监督训练 LoRA 的 E3b、E3a，把 T 文本整个冲掉了；只学稀疏文本的 C3 保住了。
- 负责人决定加"保住语言"的变体：
  - **E3b+C3**：0.8B、2B、4B 各 3 个种子，是 H1 修订版的主臂；判据是 E3b+C3 对 C3。
  - **C3+预热**：E5 的新对照起点，与 E3b+C3 的文本曝光相同，只差稠密监督。
  - E3a+C3 另开任务，等 DEV-0041 关于 E3a 是否有 bug 的结论。
- **文本曝光必须逐样本相同**（D83-2 第 1 条）：
  - E3b+C3 里的 C3 部分，要用与 C3 完全相同的样本流：同样的样本、同样的顺序、同样的种子。
  - 稠密样本是在此之外额外加上的，步数加倍。
  - 两种损失的权重事先固定为 1:1，不调。学习率日程的长度不同，照实记下。
- 先读：
  - `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
  - `DECISIONS.md` 的 D56、D65、D70、D71、D75、D83
  - `PREREGISTRATION.md` 的 E3（已批准 v1，§2、§9）
  - `src/e3/`（config、data 的 `Schedule` 与 `ExampleBuilder`、train、warmup、c3）、`scripts/train_e3b.py`、`configs/e3b_train.json`
  - `src/adapters/checkpoint.py`（只读，第 2 节可能要加函数，见"允许改动的文件"）
  - `src/e5/stack.py` 的 `resolve_start` 与 `merge_start_lora`：只读，了解 E5 怎么读起点

## 任务

### 1. E3b+C3（`scripts/train_e3b.py --supervision grid+text`，输出变体名 `e3b_c3`）

- **样本流**：
  - 种子 s 的运行，第 t 步（t = 0 … 2·S − 1；S 是 E3b、C3 各自的步数 4,000）取：
    - **文本 8 条** = C3 种子 s 的样本流第 8t … 8t + 7 个槽位（`Schedule(supervision="text", seed=s)`）；
    - **网格 8 条** = E3b 种子 s 的样本流第 8t … 8t + 7 个槽位（`Schedule(supervision="grid", seed=s)`），含修正对，比例同 E3b。
  - 合计每步 16 条，共 8,000 步。全程的文本样本恰好是 C3 运行的 64,000 条，顺序相同；网格样本恰好是 E3b 运行的 64,000 条，顺序相同。
  - 注意：两条流的槽位与任务、图像的对应只由种子与 k 决定，所以同一步里同一张图会以两种监督各出现一次。在说明里写明这一点，不要另换数据种子去避开它。
- **损失**：每步的损失 = 文本样本的逐 token 平均损失 + 网格样本的逐 token 平均损失，1:1，事先固定。
  - 各自的"逐 token 平均"与 C3、E3b 单独训练时的定义相同：文本按 C3 的权重，网格按 `weights_for`。
  - 训练日志分别记两种损失、两种监督的 token 数。
- **学习率**：同 E3b，2e-4，warmup 3% 加余弦，按 8,000 步排。运行身份与 README 写明"日程长度是 C3、E3b 的 2 倍"。
- **微批**：每个微批只含一种监督也可以，例如 4 个微批里 2 个文本、2 个网格，这样形状更稳；你定，写明理由。批组成进运行身份。
- **配置**：`configs/e3b_train.json` 新增一节，例如 `mixed`：两种监督的条数、损失权重、步数的来历。版本号与 sha256 随之改变。**已有的 E3b、C3、C1′、预热的运行身份必须逐字节不变**：测试钉住。
- 检查点：与 E3b 同一种（LoRA 加网格 token 行），能直接给 `eval_e3b`、`run_e1 --checkpoint`，以及 E5 的起点（第 3 节）。
- **测试**：
  - **逐样本相同**：E3b+C3 种子 s 的前若干步里，文本样本的（样本 id、epoch、text_kind、输入与目标 token），与 C3 种子 s 对应槽位的逐条相同；网格样本与 E3b 种子 s 的逐条相同。
  - 损失是两者之和，权重 1:1，用合成的小例子核对数值。
  - 续跑逐字节相同。
  - 已有变体的运行身份不变。

### 2. C3+预热（`scripts/train_e3b.py --supervision warmup --start-checkpoint <C3 的 ckpt/final/model>`，变体名 `c3_warmup`）

- 先把 C3 的 LoRA 合并进 base，再在其上做网格格式预热：新 LoRA 加网格 token 行。预热的数据、停止规则、留出集与现在的预热完全相同。
- **存成一个检查点，E5 当作普通起点读**。建议做法：把 C3 的 LoRA（秩 32）与预热的 LoRA（秩 32）按秩拼接成一个秩 64 的 LoRA，即 B = [B_c3, B_w]、A = [A_c3; A_w]，缩放保持 alpha/r 不变。
  - 拼出来的合并后增量 = ΔW_c3 + ΔW_w，与"先合并 C3、再合并预热"逐位相同（fp32 下核对）。
  - 网格 token 行取预热之后的。
  - manifest 记下父检查点（C3 manifest 的 sha256）与拼接方式。
  - 如果你有更干净的做法（例如 E5 侧支持"父起点"），说清楚并写进总结，但要保证第 3 节能用。
- 预热的停止规则照旧，在留出集上判；结果写进 `train_summary.json` 的 warmup 一节。
- **测试**：
  - 拼接后合并的权重，与依次合并的逐位相同；
  - C3+预热的检查点能被 `factory.load_trained(merge=True)` 加载，网格受限解码可用；
  - 父检查点的 sha256 不符时拒绝。

### 3. 给 E5 用的准备（只做能在 `src/e3`、`src/adapters/checkpoint` 里做的部分）

- E5 的新起点（D83-5）：`e3b_c3`（E3b+C3 的检查点）与 `c3_warmup`（第 2 节的检查点）。E5 侧的登记（`configs/e5_train.json` 的 starts、`src/e5/stack.resolve_start` 的变体核对）由下一个任务（DEV-0043）做。
- 本任务只需保证：两种检查点的 manifest 与训练叶目录的 config.yaml 记着 `variant`（`e3b_c3`、`c3_warmup`）与 `seed`，与现有起点同一种格式，E5 的 `_leaf_config` 能读到。

## 允许改动的文件

- `src/e3/`（config、data、train、warmup、c3，除 `evaluate.py` 以外，那个文件 DEV-0041 在改）、`scripts/train_e3b.py`、`configs/e3b_train.json`
- `src/adapters/checkpoint.py`：只加函数，用于 LoRA 拼接或保存秩 64 的 LoRA；已有函数的行为不变
- 对应的测试与 fixture

**不许碰**：`src/e3/evaluate.py`、`scripts/eval_e3b.py`（DEV-0041 在改）、`src/adapters/` 的其余文件、`src/e1/`、`src/readout/`、`src/e5/`、`src/synth/`、`configs/e1_*.json`、`configs/e3a_train.json`、`configs/e5_*.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 两套全量测试并行时不要同时用两个 `conda run`。直接用 `/opt/miniconda3/envs/visint/bin/python -m pytest …` 与 `PATH=/opt/miniconda3/envs/visint/bin:$PATH PYTHON=/opt/miniconda3/envs/visint/bin/python bash scripts/pytest_server_parity.sh …`，各写各的日志。
- 验收：
  - 普通与 server-parity 两套全量测试，在最后一次改动之后各跑一遍；
  - tiny 彩排：
    - `train_e3b --tiny --supervision grid+text`；
    - `train_e3b --tiny --supervision text`，然后在它之上 `--supervision warmup --start-checkpoint …`；
    - 两个检查点各跑一次 `eval_e3b --tiny` 与 `run_e1 --tiny --checkpoint --limit 4`。
- 总结里写明：
  - 样本流的做法与逐样本核对的结果；
  - 损失与学习率日程；
  - 微批的组成；
  - C3+预热的存法与逐位核对；
  - 新配置的 sha256；
  - 所有偏离。
