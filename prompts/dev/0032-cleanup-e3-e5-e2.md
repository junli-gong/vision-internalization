[DEV-TASK 0032] 收尾：训练后检查点的 restricted 核对用自己的网格 token、E3 去重补 T 文本评测集、E5 配置说明改为四类、E2 运行记录写明探针卷积的精度设置

## 背景

D62、D64、D65 记下了四项遗留，都不改任何已定的实验口径：
- D65-4(a)、D56-5：`check_training_interfaces --checkpoint` 的 restricted 核对还在用 E1 的 0、1、换行文本 token，并且拒绝新增的网格 token。对训练后的 E3b 检查点，应改用检查点自己的网格 token 与 E3b 的允许集合。
- D65-4(b)：`configs/e3b_train.json` 的 `dedupe.required_eval_sets` 只要求 `refcoco/eval` 与 `nyuv2/test`。T 文本评测（`run_e1 --checkpoint`）还会用到 E1 的其他评测集，要一并要求比较过。
- D62-4：`configs/e5_prompts.json` 里思维链的说明还写"按自然收尾与强制收尾分开报"，实际已按 `thinking_end` 四分类报（D62）。
- D64-4：E2 探针的卷积走 cuDNN，PyTorch 默认允许 TF32。按 D64 不改设置，但要把实际的设置写进运行记录。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D50、D56、D62、D64、D65
- `scripts/check_training_interfaces.py`、`src/e3/`（`tokens`、`merge_check`、`evaluate`）、`src/e2/runner.py`（运行记录的 audit）、`src/e2/data.py`（去重清单的读取）、`configs/e3b_train.json`、`configs/e5_prompts.json`
- `experiments/P0/0026_a800_delta/results_0026/dedupe/train_exclusions.json`（看 `eval_sets` 的实际名字）

## 任务

### 1. 训练后检查点的 restricted 核对（D56-5）

- 给了 `--checkpoint`，且检查点带网格 token 时：restricted 核对改用检查点自己的网格 token 与允许集合，即 `tokens.allowed_ids(task)`，与 `eval_e3b` 的单轮生成同一个集合。
  - 样本与提示用 E3b 的单轮评测：分割任务的开发子集。
  - 判据沿用 v1（D50）的 restricted 一节，阈值不变。
- 不带网格 token 的检查点（例如 C3）与不给 `--checkpoint` 的 base 核对，行为与现在逐字节相同，测试钉住。
- `src/e3/merge_check.py` 的 `checks` 可以加上 restricted。默认是否加，写明理由：
  - 加上更严，评测前多一道核对，代价是多用一些时间，要给出估计；
  - E1 的 T 文本评测用的是 base 的文本 token，那里不需要。
- 测试：
  - 用 DEV-0025 的过拟合小检查点，restricted 核对通过；
  - 构造一个坏检查点：合并前后网格 token 的 logits 分家，restricted 核对不通过。

### 2. E3 去重要求补上 T 文本评测集（D65-4(b)）

- `configs/e3b_train.json` 的 `dedupe.required_eval_sets` 补上 T 文本评测会用到的集合。
- 名字与 RUN-0026 剔除清单的 `eval_sets` 完全一致，例如 `coco_val2017/e1sub500`、`countbench/eval`、`cvbench/test`、`embspatial/test`、`blink_reldepth/val`、`odinw13/eval`。以实际文件为准，逐个核对。
- 加一条测试：用 RUN-0026 那份 `train_exclusions.json`（仓库里有）读取时，补上之后的要求能通过，不会挡住训练。
- E3b 配置的 sha256 会变，写进总结。E3 还没有正式运行。

### 3. E5 配置的思维链说明（D62-4）

- `configs/e5_prompts.json` 里所有"自然收尾与强制收尾分开报"之类的说明，改为按 `thinking_end` 四分类（natural、forced、not_opened、unclosed）分开报，与 E1 同口径（D62）。
- 只改说明文字，不改任何参数。E5 配置的 sha256 会变，写进总结。E5 的正式运行还没有开始。

### 4. E2 运行记录写明探针卷积的精度设置（D64-4）

- `logs/run_k.json` 的 audit，以及 `--estimate` 的 estimate.json 表头，加一项：运行时的
  - `torch.backends.cudnn.allow_tf32`、`torch.backends.cudnn.deterministic`、`torch.backends.cudnn.benchmark`；
  - 新接口 `torch.backends.cudnn.conv.fp32_precision`（若存在）；
  - `torch.backends.cuda.matmul.fp32_precision`（作用域外的全局值）。
- 只记录，不改任何设置。E2 配置的 sha256 不变；运行身份 config.yaml 不变，测试钉住。

## 允许改动的文件

- `scripts/check_training_interfaces.py`、`src/e3/`、`configs/e3b_train.json`
- `configs/e5_prompts.json`：只改说明文字
- `src/e2/runner.py`、`src/e2/estimate.py`、`scripts/run_e2.py`：只为第 4 项
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/e5/` 的逻辑、`src/synth/`、`src/data/`、`configs/e1_prompts.json`、`configs/e2_probes.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `check_training_interfaces.py --tiny --checkpoint <过拟合小检查点> --checks restricted merge`
  - `eval_e3b.py --tiny`
  - `run_e2.py --tiny --task depth` 与 `--estimate --tiny`
- 总结里写明：
  - restricted 核对对检查点的做法与判据；
  - merge_check 默认是否加 restricted、理由与用时；
  - E3b、E5 配置的新 sha256；
  - `required_eval_sets` 的最终清单；
  - E2 运行记录新增的字段。
