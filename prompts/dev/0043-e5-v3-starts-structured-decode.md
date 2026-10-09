[DEV-TASK 0043] E5 v3：新起点（E3b+C3、C3+预热）登记；E5 全部网格任务改用结构强制解码（N = 0、val 选择、test 评测）

## 背景（D83；E5 预注册 §8a v3）

- E5 主比较的起点改为三个：
  - `e3b_c3`：E3b+C3，内化且保住了语言；
  - `c3_warmup`：C3+预热，文本曝光相同，只差稠密监督；
  - `base_warmup`：基座+预热，参照。
- H5 的主判据改为 E3b+C3 对 C3+预热。E3a 与"基座、不预热"退出主比较。
- **E5 的全部网格任务以结构强制解码为正式口径**：二值的 c、d、e，33 个符号的起伏 a、b；适用于 N = 0、训练中的 val 选择、test 评测。
  - 规则：每个格子位置只允许本任务的值 token，行末只允许行分隔，最后只允许停止 token。
  - 理由（D83-3）：a、b 类的值分布摊开，贪心受限解码会像 E3b 的深度那样提前停止，结果可能全部无效。
- 依赖：
  - DEV-0041 的结构强制模块（`src/grid/structured_decode.py` 或它实际的名字，见 D84）；
  - DEV-0042 的两种新检查点（变体 `e3b_c3`、`c3_warmup`；C3+预热是秩拼接的单个 LoRA，见 D85）。
  - 两者都已合入 main，先读它们的总结与 D 条目。
- 先读：
  - `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
  - `DECISIONS.md` 的 D75、D78、D81、D82、D83，以及 DEV-0041、DEV-0042 合入时的 D 条目
  - `PREREGISTRATION.md` E5 §8、§8a
  - `src/e5/`：train_config、stack、train、trained_eval、val_select、token_grid；`configs/e5_train.json`；`scripts/train_e5.py`、`scripts/eval_e5_trained.py`

## 任务

### 1. 新起点（`configs/e5_train.json` 的 starts、`src/e5/stack.py`）

- 新增 `e3b_c3`：`variant: e3b_c3`，`grid_tokens: from_checkpoint`。
- 新增 `c3_warmup`：`variant: c3_warmup`，`grid_tokens: from_checkpoint`。
  - LoRA 是秩 64：C3 的秩 32 与预热的秩 32 拼接而成。确认 `merge_start_lora` 与权重指纹对秩 64 照常工作；不行就在 `stack.py` 里补，测试钉住"合并后与依次合并 C3、预热逐位相同"。
- `e3a`、`base` 两个起点保留在配置里，不删；运行脚本里标注它们不属于 v3 主比较。
- `--start-only`（N = 0，DEV-0040）的允许起点改为 `base_warmup`、`e3b_c3`、`c3_warmup`。`e3b` 是否保留由你判断，写明理由。
- 配种子、变体核对照旧：读起点训练叶目录的 config.yaml。

### 2. 结构强制解码（`src/e5/trained_eval.py`、`src/e5/val_select.py`，必要时 `src/e5/token_grid.py`）

- 新增解码方式 `structured`。按任务格式取允许集合：二值任务每格是 2 个值 token；起伏任务每格是 33 个（背景加 32 档）。行末是行分隔，最后是停止 token，长度 n(n + 1) + 1。复用 DEV-0041 的模块，不另写一套。
- **默认值的处理**：
  - `configs/e5_train.json` 新增 `decode` 一节，v3 的正式值是 `structured`，写明依据 D83-3。配置版本号与 sha256 随之改变。
  - 训练时的 val 选择与 test 评测都读这一节。运行身份记解码方式。
  - `--decode restricted` 保留作诊断。
- **合并核对**（`stacked_merge_check`）照旧比较受限解码一致性，它核对的是合并前后，与解码方式无关。如果你认为也要加一项结构强制下的核对，写明理由。
- **零样本（`scripts/run_e5.py`、RUN-0032）不动**：那是数字文本格式，不是网格 token。
- **测试**：
  - 结构强制下，输出长度恒为 1057，每个位置的 token 都在该位置的允许集合里；
  - 不加结构约束时，与 restricted 逐 token 相同，用 tiny 模型；
  - val 选择用结构强制时，选出的步与分数可复现；
  - `decode = restricted` 时，与改动前逐字节相同（快照或现有测试）；
  - 续跑时解码方式不同就拒绝混写。

### 3. 彩排

开发机 tiny：

- `train_e3b --tiny --supervision grid+text`，得到 e3b_c3 的起点；
- `train_e3b --tiny --supervision text`，再在它之上 `--supervision warmup --start-checkpoint …`，得到 c3_warmup；
- 两个起点各跑一次 `eval_e5_trained --tiny --start-only … --limit 2`；
- 再各跑一次 `train_e5 --tiny --init … --n-train 10`（带 val 选择）与 `eval_e5_trained --tiny --checkpoint … --limit 2`；
- 起点用"按训练叶目录结构搬到别处"的布局（`scripts/server/train_queue.sh` 的 finalize）。

## 允许改动的文件

- `src/e5/`（除 scoring、protocol、runner 以外）、`configs/e5_train.json`、`scripts/train_e5.py`、`scripts/eval_e5_trained.py`
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/e3/`、`src/readout/`、`src/synth/`、`src/e5/scoring.py`、`src/e5/protocol.py`、`src/e5/runner.py`、`scripts/run_e5.py`、`configs/e1_*.json`、`configs/e3*.json`、`configs/e5_prompts.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 两套全量测试并行时不要同时用两个 `conda run`。直接用 `/opt/miniconda3/envs/visint/bin/python -m pytest …` 与 `PATH=/opt/miniconda3/envs/visint/bin:$PATH PYTHON=/opt/miniconda3/envs/visint/bin/python bash scripts/pytest_server_parity.sh …`，各写各的日志。计时类测试（`tests/test_synth_*::test_speed_*`）在 CPU 被占满时可能失败；只有这类失败时，单独重跑确认，并写进总结。
- 验收：两套全量测试，在最后一次改动之后各一遍；第 3 节的彩排。
- 总结里写明：
  - 新起点的登记与秩 64 的核对；
  - 结构强制的接入点：N = 0、val、test；
  - 新配置的 sha256；
  - 所有偏离。
