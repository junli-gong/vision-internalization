[DEV-TASK 0030] E3b 第二部分：评测（单轮网格、叠加反馈多轮与计算记账、评测前的合并核对、训练后模型的 T 文本评测、C3 改用原生框格式）

## 背景

- DEV-0025（D56）做了 E3b 训练框架的第一部分：词表、目标、修正对、C3、LoRA 训练循环。本任务做评测。
- 硬约束与已定的规则：
  - **单轮结果是主要数字**；多轮流程（叠加反馈迭代 K = 1、2、4）的结果附计算量，并与匹配计算量的单轮对比（硬约束 11、简报 §8.5、§12）。多轮流程在看测试结果之前固定。
  - **评测前必须合并 LoRA，并在开发子集上核对合并前后一致**（简报 §12）。用 `scripts/check_training_interfaces.py --checkpoint`，判据按 D50 的 v1。
  - **可比的评测一律在 B200 上做**（D55、D59-10）：训练在 A800，检查点经 Drive 传回 B200。
  - **T 文本通道**：训练后的模型要用 E1 的同一套协议评测，跨通道迁移要看这一项。冻结口径、批配置 64、三列。
  - **C3 的框**：D56 定正式运行前改用各家族原生的框格式加 E1 的定位提示。DEV-0027 已加 `format_boxes`。
  - 叠加图用唯一入口 `feedback_image`，训练与评测共用。
- 训练接口：`factory.load_trained(dir, merge=True)`、`generate_restricted`、`tokens.allowed_ids`、`end_of_turn_token_id()`。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §4（硬约束 11）、§8.5、§12
- `DECISIONS.md` 的 D42、D45、D50、D55、D56、D59、D60
- `src/e3/`、`scripts/train_e3b.py`、`src/e1/`（评测框架）、`scripts/run_e1.py`、`scripts/check_training_interfaces.py`、`src/accounting.py`

## 任务

### 1. 网格评测（`src/e3/evaluate.py`、`scripts/eval_e3b.py`）

- **加载**：`factory.load_trained(<检查点>, merge=True)`。开跑前先跑一遍合并核对：`check_training_interfaces --checkpoint` 的同一套函数，n 可配置；不通过就报错退出。
- **单轮**：
  - 输入：图 + 指令；
  - 生成：`generate_restricted`，允许集合用 `tokens.allowed_ids(task)`，`max_new_tokens = 1057`；
  - 解析：`parse_target`；
  - 打分：网格上采样到原图后算指标。
- **评测集**：
  - 分割：E1 的 RefCOCO val（eval 总体）。指标 cIoU、gIoU、Boundary IoU；按 person 与非 person 分列。
  - 深度：NYU 官方 654，Eigen crop，64 档。指标 AbsRel、δ1。
  - 都带 bootstrap CI。
- **上采样**：默认用无学习的引导上采样（`src/executors/upsample.py`），并另报只用双线性的结果（简报 §8.5）。
- **叠加反馈多轮**（K = 1、2、4）：
  - 第 k 轮把上一轮的网格经 `feedback_image` 渲染成叠加图，与原图一起送入，指令用训练时的修正模板；
  - 每轮的输出单独记录与评分；
  - 计算记账：每轮的 prompt token、生成 token、视觉 token、GPU 墙钟；
  - **匹配计算量的单轮对比**：在同样的总计算量下，单轮可以做什么，例如多次采样取多数，或者更大的图像预算。可行的方案列出来，选一个简单、确定的实现，写明。
- **输出**：`experiments/E3/<编号>/eval/<变体>/<模型>/seed<k>/<任务>/`，内容是 predictions、metrics、compute、config、README。支持续跑。

### 2. 训练后模型的 T 文本评测（E1 协议）

- `scripts/run_e1.py` 加 `--checkpoint <dir>`：用 `load_trained(merge=True)` 替代 base 加载，其余协议不变（冻结口径、批配置、三列）。
- 运行身份记检查点的 manifest sha256；评测前同样先跑合并核对。
- 另统计一项：自由生成里冒出网格 token 的条数（DEV-0025 的建议），写进 metrics 的诊断。
- 测试：`--tiny` 下，用训练框架产出的小检查点跑通全部任务。

### 3. C3 改用原生框格式

- `src/e3/c3.py` 的框目标改为 `adapter.format_boxes(...)` 生成的本家族原生格式，提示用 E1 定位任务的提示原文，从 `configs/e1_prompts.json` 读，不另写。
- 多边形目标保持 E1 的统一格式。
- 加一条测试：C3 的框目标，用本家族的 `parse_boxes` 解析回来，与 GT 框在格点精度内一致。
- C3 的配置改动写进 `configs/e3b_train.json`，附 basis（D56、D60）。

### 4. 测试（CPU 小模型，`--tiny`）

- 单轮与多轮在三个家族上跑通，多轮的记账字段正确，K = 1 时与单轮的定义一致（写明是否相同）。
- 过拟合检查点：用 DEV-0025 的过拟合测试流程训出一个 8×8 网格的小检查点，评测得到满分。
- 合并核对不通过时，评测拒绝运行（构造一个坏检查点）。
- `run_e1.py --tiny --checkpoint` 跑通。

## 允许改动的文件

- `src/e3/`、`scripts/eval_e3b.py`、`configs/e3b_train.json`
- `scripts/run_e1.py` 与 `src/e1/`：只为 `--checkpoint`，其余行为不变，E1 配置的 sha256 不变
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e2/`、`src/e5/`、`src/synth/`、`src/data/`、`configs/e1_prompts.json`、`configs/e5_prompts.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `eval_e3b.py --tiny`
  - `run_e1.py --tiny --checkpoint`
- 总结里写明：
  - 评测的完整流程与命令；
  - 多轮的提示与记账；
  - 匹配计算量对比的实现与理由；
  - C3 改动之后的目标样例（三家族）；
  - 在 B200 上的预计用时：0.8B 与 9B 各一个检查点，单轮加多轮，再加 T 文本评测。
