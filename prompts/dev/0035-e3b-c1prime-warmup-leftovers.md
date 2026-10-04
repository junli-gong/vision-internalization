[DEV-TASK 0035] E3b 的 C1′ 对照、"基座 + 网格格式预热"、三项遗留（事后读数接 --checkpoint、网格允许集合只留一份、eval_e3b 透传 --config）

## 背景

- **C1′**（D36-5）：冻结 LLM，只训练新增网格 token 的嵌入行与输出行。它与 E3a 的 C1 同构，量出冻结的 LLM 本来能表达多少。
- **网格格式预热**（D36-5，N5 第 2 处）：E5 的主比较是"E3b 模型"对比"基座 + 网格格式预热"。
  - 预热只用纯符号任务：用文字给出格子坐标，让模型输出网格 token，不含任何视觉内容；
  - 练到格式错误率低于 1% 为止。
- **遗留**：
  - D68-4(a)：事后读数（去偏字母读数、free）要能在训练后的检查点上跑。D66-3 要求 E3 在四个选择题类任务上的迁移结论同时看全量去偏读数与 free 200 条；
  - D68-4(b)：`check_training_interfaces` 改为 import `src/e1/forced.grid_format_ids`，0、1、换行的构造只留一份；
  - D67-4(a)：`eval_e3b.py` 把 `--config` 透传给合并核对。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D36、D45、D56、D65、D66、D67、D68
- `src/e3/`（train、tokens、targets、data、evaluate、merge_check）、`scripts/train_e3b.py`、`scripts/eval_e3b.py`、`src/e1/posthoc*.py`、`src/e1/forced.py`、`scripts/run_e1.py`、`scripts/check_training_interfaces.py`、`configs/e3b_train.json`

## 任务

### 1. C1′（`train_e3b.py --supervision grid --variant c1prime`，或者其他清楚的开关）

- 冻结 LLM：不注入 LoRA，LLM 的全部参数不训练；只训练网格 token 的嵌入行与输出行，沿用 `trainable_token_rows` 的 fp32 主副本。
- 数据、步数、批、学习率调度同 E3b 的网格监督。token 行的学习率沿用配置的 `token_rows`；如果冻结 LLM 时需要另定，写明理由。
- 检查点只有 token 行，没有 LoRA。`eval_e3b` 要能加载并评测：合并核对里没有 LoRA 的部分跳过；restricted 照做，要写明合并核对在这里的含义。
- **测试**：
  - 训练前后 LLM 的参数逐位不变，只有网格 token 行变；
  - 小模型上过拟合检查能学会 8×8 网格，或者如实报告学不会。冻结 LLM 本来就可能学不会，这正是 C1′ 要量的。

### 2. 网格格式预热（`--supervision warmup`）

- **任务**：纯文本输入，不给图像。用文字给出要涂的格子坐标，例如"Rows 3–5, columns 10–12 are foreground"，或者逐格列出；措辞写明。目标是对应的 32×32 网格 token 序列，与 E3b 网格监督同一套 token、同一种格式。
  - 坐标集合由程序随机生成：矩形、多个矩形、简单形状。种子固定，与任何视觉数据无关。
  - 深度网格也要预热：用文字给出若干区域的深度档号，输出 64 档的深度网格。两种都做，混合比例写明。
- **训练**：LoRA 与 token 行，同 E3b 的网格监督。停止规则："格式错误率低于 1%"。
  - 每 N 步在固定的留出集上做受限解码，统计格式错误：解析失败、行数或列数不对；
  - 连续两次低于 1% 就停，并记下步数；
  - 设一个步数上限，写明。
- **输出**：检查点，与 E3b 检查点同一种格式，`eval_e3b` 与 `run_e1 --checkpoint` 都能加载。
- **测试**：小模型上能跑通，停止规则按设计触发；输入里没有图像。

### 3. 遗留三项

- **(a) 事后读数接 `--checkpoint`**：`run_e1.py --subset debiased|free --checkpoint <dir> --checkpoint-label <标签>`。
  - 加载方式与 T 文本评测相同：`load_trained(merge=True)`，先做合并核对；
  - 格式强制（forced）仍只用于 base 的 Qwen，不配 `--checkpoint`；
  - 测试：`--tiny` 的检查点上 debiased 与 free 都跑通；base 运行的输出逐字节不变。
- **(b) 网格允许集合只留一份**：`check_training_interfaces.py` 改为 import `src/e1/forced.grid_format_ids`，停止 token 另加。输出 JSON 逐字节不变，测试钉住。
- **(c) `eval_e3b.py` 透传 `--config`** 给合并核对，加测试：非默认配置时，核对读的是同一份配置。

## 允许改动的文件

- `src/e3/`、`scripts/train_e3b.py`、`scripts/eval_e3b.py`、`configs/e3b_train.json`（只为 C1′ 与预热新增条目，已有条目不动；sha256 会变，写进总结）
- `src/e1/posthoc*.py`、`src/e1/debiased.py`、`scripts/run_e1.py`：只为 (a)
- `scripts/check_training_interfaces.py`：只为 (b)
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/readout/`、`scripts/train_e3a.py`、`scripts/eval_e3a.py`、`configs/e3a_train.json`（DEV-0034 在并行做 E3a）、`configs/e1_prompts.json`、`configs/e1_posthoc.json`、`configs/e2_probes.json`、`configs/e5_prompts.json`、`src/e2/`、`src/e5/`、`src/synth/`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - C1′ 与预热的 `--tiny` 训练加 `eval_e3b --tiny`
  - `run_e1 --tiny --subset debiased --checkpoint …`
- 总结里写明：
  - C1′ 的参数组与学习率；
  - 预热任务的措辞、生成规则、混合比例、停止规则与步数上限；
  - E3b 配置的新 sha256；
  - 三项遗留的做法；
  - 预热与 C1′ 在 A800 上的用时推算（按 RUN-0026 的实测：0.8B 网格 3.6 秒一步，9B 13.8 秒一步）。
