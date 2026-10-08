[DEV-TASK 0040] E5 的 N = 0 读数："只评起点"的评测入口

## 背景

- D81-1：负责人决定 E5 主比较要跑 N = 0 的点。
  - 做法：起点检查点不经 E5 训练，直接在 E5 的 test 上评。
  - 用途：区分截距（E3b 一开始就更高）与斜率（E3b 学得更快）。规则见 `PREREGISTRATION.md` E5 §8 第 2 项。
  - 起点：基座加预热（`base_warmup`）与 E3b，种子 0–2。N = 0 排在 E5 最前面跑，顺带检验评测入口是否正确。
- 现在 `scripts/eval_e5_trained.py` 只接 E5 检查点：起点加 E5 LoRA，读 `<ckpt>/final/e5_stack.json`。没有"只有起点、没有 E5 LoRA"的入口。
- 先读：
  - `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
  - `DECISIONS.md` 的 D50、D59-4、D72-1、D73、D75、D78、D81
  - `PREREGISTRATION.md` E5 §8（已批准 v2）
  - `src/e5/trained_eval.py`、`src/e5/stack.py`（`resolve_start`、`check_seed_pairing`、`setup_start`、`start_merge_check`、`load_stacked`）、`src/e5/train_config.py`、`scripts/eval_e5_trained.py`、`scripts/train_e5.py`
  - `configs/e5_train.json` 的 starts 与 eval 两节

## 任务

### 1. "只评起点"（`scripts/eval_e5_trained.py`、`src/e5/trained_eval.py`，必要时 `src/e5/stack.py`）

- 新的调用方式，与现有的 `--checkpoint`（E5 检查点）二选一：

  ```
  python scripts/eval_e5_trained.py --start-only --init {base_warmup,e3b} --start-checkpoint <起点的 ckpt/final/model> \
      --model qwen3.5-2b --seed <i> --out experiments/E5/<NNNN_slug> [--task all] [--subset eval|dev] [--resume] [--limit N]
  ```

- **加载**：与训练后评测完全同一条路径，只是没有 E5 LoRA：
  - 建 base；
  - `resolve_start` 与核对（构造方式、manifest、网格 token、变体）；
  - 合并起点 LoRA，`setup_start` 加网格 token 行。
  - 不能另写一套加载。
- **只允许 `base_warmup` 与 `e3b`**（D81-1）。`e3a` 的网格 token 行是新加的、没训练过，`base` 没有起点检查点，两者都报用法错误。
- **配种子**：`--seed` 必须等于起点训练叶目录 config.yaml 的 seed，规则同 `train_e5.py`（`check_seed_pairing`）。不符就拒绝，可用 `--allow-seed-mismatch` 放开，并记进运行身份。
- **评测前的核对**：用起点检查点自己的合并核对，即 `train_e5.py` 开训前做的那一项，规则见 `stack.start_merge_check`。已通过的照样复用。
  - 没有 E5 LoRA，所以训练后评测的"E5 LoRA 合并前后"核对不适用。在输出里写明跳过了哪一项、为什么。
- **评测本身**：样本清单、提示（指令原文加 E3b 网格指令的结尾句）、允许集合、max_new_tokens、停止 token、批组成、解析、评分、汇总与门槛，全部与训练后评测同一套函数，一行不另写。
- **输出**：
  - 写进 `<out>/trained/<起点>/<模型>/n0/seed<k>/<任务>[__dev]/`，`n0` 是数据量档的标记，与训练后评测的 `n<数据量>` 同一层；
  - 文件与训练后评测相同；
  - 运行身份记 `start_only: true`、`n_train: 0`、起点 manifest 的 sha256 与权重指纹。
- **续跑**同训练后评测。

### 2. 测试

- **关键的一条**：同一个 tiny 起点，"只评起点"的 predictions 与一个"E5 LoRA 为恒等"的训练后评测逐字节相同。
  - LoRA 的 B 初始化为 0 时就是恒等；用训练 0 步、或刚初始化就保存的 E5 检查点都行，你选一个最干净的做法，在总结里说明。
  - 这条钉住"两条路径只差 E5 LoRA"。
- `--init e3a`、`--init base` 被拒；`--checkpoint` 与 `--start-only` 同时给出被拒。
- 种子不配被拒；`--allow-seed-mismatch` 放开后记进运行身份。
- 不用 `--start-only` 时，现有的训练后评测输出与改动前逐字节相同，用快照或现有测试钉住。
- 续跑逐字节相同。

### 3. 彩排

开发机 tiny：

- `train_e3b.py --tiny --supervision warmup` 与 `--supervision grid` 各训一个起点；
- 两个起点各跑一次 `eval_e5_trained.py --tiny --start-only … --limit 2`；
- 训练叶目录用默认布局，同时也用"按叶目录结构搬到别处、旁边放 config.yaml"的布局各试一次（B200 的训练队列就是这样存最终检查点的，见 `scripts/server/train_queue.sh` 的 finalize）。

## 允许改动的文件

- `scripts/eval_e5_trained.py`、`src/e5/trained_eval.py`：只加选项，不改已有行为
- `src/e5/stack.py`：只在需要把现有加载拆成"起点"与"E5 LoRA"两步时动，已有函数的行为不变
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/e3/`、`src/readout/`、`src/synth/`、`src/e5/` 的其余文件（scoring、protocol、token_grid、train*、runner），所有 `configs/*.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。临时输出放系统临时目录，用完即删。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 第 3 节的彩排
- 总结里写明：
  - 加载路径怎么复用（与训练后评测的差别只在哪里）；
  - 恒等对照用的是哪种做法、逐字节核对的结果；
  - 跳过了哪项核对、为什么；
  - 所有偏离。
