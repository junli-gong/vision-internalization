[DEV-TASK 0038] E5 主比较的小档修正（步数下限 + val 选检查点 + 配种子）；E2 探针接训练后检查点与指定层（E3 的机制测量）；跨机器训练一致性的比对工具

## 背景

- 负责人 D77-3：E5 主比较固定 2 个 epoch 时，1K 档只有约 125 步，小数据量上量到的混着"样本少"与"优化步数少"，而 H5 的主信号恰在 1K 与 4K。修法：
  - 1K 与 4K 的训练步数设下限，至少 1,000 步；
  - 用 **val** 选检查点，防止过拟合；
  - 16K 与 64K 照旧 2 个 epoch；
  - 配种子：E5 的第 i 个种子用 E3 的第 i 个种子作起点。
- D77-2：E3 训完后，每个尺寸 × 变体取 1 个种子，在 A800 上重跑 E2 的线性探针，只做深度与指代分割，只看最后一层与 E2 里最好的那一层。`run_e2` 现在只能读 base 模型。
- D72-4：开训前要做跨机器一致性检查。同一个种子、同一批数据，A800 与 B200 各 50 步，损失曲线应在 bf16 噪声以内一致。需要一个比对两份 `train_log.jsonl` 的工具，给出判据。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D50、D63、D64、D72、D75、D77；`PREREGISTRATION.md` 的 E2（v1 与 §5a）、E3（已批准 v1，§10a）、E5（主比较的设计）
- `src/e5/`（train_config、train_data、train、trained_eval、stack）、`configs/e5_train.json`、`scripts/train_e5.py`、`scripts/eval_e5_trained.py`
- `src/e2/`、`scripts/run_e2.py`、`configs/e2_probes.json`；`src/adapters/factory.load_trained`；`src/readout/`（E3a 检查点的格式）

## 任务

### 1. E5 小档修正（`src/e5/`、`configs/e5_train.json`）

- **步数**：每个数据量档的步数 = max(按 epoch 算出的步数, `min_steps`)。
  - `min_steps` 只对 1K、4K 生效，取 1,000，写进配置。
  - 16K、64K 不受影响，仍是 2 个 epoch。
  - 学习率调度按实际步数排。
- **val 选检查点**：只对受下限影响的档。
  - 每 `val_every` 步在 E5 的 **val 划分**（生成器的 val，不是 test）上做一次受限解码的评测，指标与 test 的标题口径相同；
  - 每类固定 n 条，写进配置；
  - 留下 val 上最好的那一步的检查点作为 `final/`，记录选中的步与全部 val 曲线；
  - test 只在选定的检查点上评一次。
  - 选择规则写清楚，例如"五类标题指标的平均"，或者"各类分别不变的总分"，选一个、写理由。
- **配种子**：`train_e5.py --seed i` 时，起点检查点要求来自 E3 的种子 i。核对起点 manifest 里记的种子，不符就报错；可用 `--allow-seed-mismatch` 放开，记进运行身份。
- **测试**：
  - 1K 档的步数 = 1,000，16K 档不变；
  - val 选择挑的是 val 上最好的一步，test 没被读到（加断言）；
  - 种子不配时报错；
  - 续跑逐字节相同。
- 新的用量推算：2B，3 个起点；1K、4K、16K 各 3 个种子，64K 1 个种子；在 B200 上，按 A800 的 2 倍算。

### 2. E2 探针接训练后检查点与指定层（`src/e2/`、`scripts/run_e2.py`）

- `run_e2.py --checkpoint <dir> [--layers L24 L8 …]`：
  - E3b、C3、C1′ 的检查点：`load_trained(merge=True)` 之后取隐藏状态；
  - E3a 的检查点：只用其中的 LoRA（`final/model`），不用查询与读出；
  - C1、C2 的 LLM 权重没动：直接报错，提示"引用 E2 的 base 读数"，不重复跑。
- 先做合并核对，与 E3 评测同一套（`src/e3/merge_check`）。
- **指定层**：只训练给定层上的探针。探针设置与 E2 v1 完全相同：同一个配置、同一套超参与步数、同一个种子规则。
- **输出**：运行身份记检查点 manifest 的 sha256，输出目录带检查点标签，规则同 `run_e1 --checkpoint-label`。
- **测试**：
  - 小检查点上跑通；
  - 给 base 时与原来的 `run_e2` 在同一层上逐字节相同；
  - C1/C2 被拒；
  - 合并核对不过时拒绝。

### 3. 跨机器训练一致性的比对工具（`scripts/compare_train_logs.py`）

- 输入：两份 `train_log.jsonl`，同一个运行身份、同一个种子，不同机器。先核对两边 `config.yaml` 的运行身份除机器字段外相同。
- 输出：逐步的 loss 与 grad_norm 对比，相对差的最大值与中位数，以及判定。
- **判据**（写进脚本与说明，管理员审）：
  - 前 50 步每步 loss 的相对差 ≤ 2%，中位数 ≤ 0.5%；
  - 最后 10 步平均 loss 的相对差 ≤ 1%；
  - 没有 NaN 或 Inf。
- 阈值的依据：bf16 训练里不同 GPU 的 kernel 差异，引用公开说明，或者在开发机上同一台机器换线程数实测一下噪声量级。拿不准就把阈值标为暂定。
- **测试**：同一份日志给"一致"；人为加偏的日志给"不一致"。

## 允许改动的文件

- `src/e5/`、`configs/e5_train.json`、`scripts/train_e5.py`、`scripts/eval_e5_trained.py`
- `src/e2/`、`scripts/run_e2.py`：只加选项、不改已有行为，E2 配置的 sha256 不变
- `scripts/compare_train_logs.py`（新）
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/e3/`、`src/readout/` 的逻辑（只读复用）、`src/synth/`、`configs/e1_*.json`、`configs/e2_probes.json`、`configs/e3a_train.json`、`configs/e3b_train.json`、`configs/e5_prompts.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。开发机磁盘紧张，约 12 GB 空闲，临时输出用完即删。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `train_e5 --tiny`，1K 档带 val 选择
  - `run_e2 --tiny --checkpoint`
  - `compare_train_logs.py` 用两份 tiny 日志
- 总结里写明：
  - val 选择的规则；
  - 新的 E5 用量推算；
  - E2 接检查点的做法与核对；
  - 一致性判据与依据；
  - E5 配置的新 sha256；
  - 所有偏离。
