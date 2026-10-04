[DEV-TASK 0033] E1 事后读数：去位置偏好的字母读数（N22 → (b)）与格式强制诊断（多边形预填、网格受限解码）

## 背景

G1 已通过（D66）。负责人要求两项事后读数，口径已写进 E1 预注册 §10，写于运行之前。实现必须**逐字照 §10**：
- **10.1 去偏字母读数**：NYU 点对、BLINK、CV-Bench、EmbSpatial 四个任务的新标题口径。
- **10.2 格式强制诊断**：Qwen 四个尺寸，多边形预填 `[[`；网格受限解码，固定长度。

两项都在 B200 上跑，可比的评测都在 B200 上（D55）。主口径的提示、批配置、图像口径都不变；**E1 主评测的输出与配置 sha256 不能变**（`f3353de0…`）。新口径要么放进配置里新增的、不影响旧条目哈希的位置，要么另起一份配置，二选一并写明。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `PREREGISTRATION.md` 的 E1 §3、§4、§10
- `DECISIONS.md` 的 D36、D37、D43、D60、D66
- `src/e1/`（runner、tasks、letters、scoring）、`scripts/run_e1.py`、`configs/e1_prompts.json`
- `src/adapters/base.py`：`option_logits`、`generate_batch`、`generate_restricted`，以及三家族怎样构造输入

## 任务

### 1. 去偏字母读数（§10.1）

- 新的读数种类，例如 `letter_debiased`，作为四个任务 eval 子集的另一种读法。对 dev 子集也能跑，验收用。
- **选项重排**：
  - k 个选项，做全部 k 种循环移位。字母位置不变（A、B、…），选项内容按移位换位置。
  - NYU 点对：选项文字 "A is closer"、"B is closer" 指的是图上的标记，图不变，只换两行的顺序。
  - BLINK 与 CV-Bench 的提示是数据集自带的整段文字。要把选项行可靠地拆出来再按新顺序拼回，拆分规则写明。**原顺序拼回必须与原提示逐字节相同**，所有样本都要断言；拆不出的样本报错，不能静默跳过。
  - EmbSpatial 用配置里的 option_format 重拼。
- **读数**：
  - 每种顺序下，用现有的受限字母读数机制取每个合法字母的 logit，两种写法取较大的，同 D36-4；
  - 只在合法字母上取 softmax，映射回选项内容；
  - k 种顺序取平均，最大者为答案，平票取原顺序里靠前的选项。
  - 优先复用 `option_logits`；用 `generate_batch(allowed_token_ids)` 只能拿到 argmax，不够。
- **批配置**：同一批的字母集合相同（letter_batches），批大小 64，左填充。k 种顺序的批组成怎样排，写明，并保证确定性。
- **输出**：
  - 每条记录：每种顺序的字母概率、映射后的平均概率、答案与是否正确；
  - metrics：去偏读数的严格读数与 CI；原字母读数在同一批上的对照，复用 RUN-0027 的结果，或在同一次运行里顺带算出，二选一写明；
  - 另记每种顺序的单独准确率与"最常见字母占比"。
- **有效性检查**（§10.1 的两条规则）：写一个汇总，在 free 的 200 条 id 上逐格比较去偏读数与自由作答（RUN-0027 的 free 结果），标出满足两条规则的格子。可以放在 `scripts/` 里作为汇总脚本，不改 run_e1 的主流程。

### 2. 格式强制诊断（§10.2）

- **适配器**：加一个"作答开头预填"的能力，例如 `generate_batch(..., answer_prefix=...)`。
  - 预填的文字接在助手回合开头（关思考时在思考段之后），作为输入的一部分，不计入生成；解析时拼上预填。
  - 三个家族都要实现，本次只用到 Qwen，并加测试：
    - 不给预填时，输入与现在逐 token 相同；
    - 给了预填时，等价于把预填当成已经生成的前几个 token，即用 greedy 生成预填文字后续写。小模型上逐 token 核对。
  - 这是对 `src/adapters/` 的改动，只加参数、不改默认行为，写明改了哪些函数。
- **多边形**：refexp_poly，预填 `[[`，提示不变，上限同 E1 多边形；解析时拼上预填，规则同 E1。
- **网格**：grid16，提示不变。
  - **主读法**：允许集合是 `0`、`1`、换行三个 token（与 `check_training_interfaces` 的 grid16 允许集合同一套构造），**不允许停止 token**，固定生成 271 个 token；解析同 E1。
  - **结构强制**：每行第 17 个位置（下标 16、33、…）强制换行，其余位置只允许 `0`、`1`，固定 271 个 token。逐位置的允许集合用 logits 处理器实现，不改模型。
- **样本**：思考子集的同一批 id，各 200 句。
- **报告**：严格读数、有效率、条件读数；与同一批 id 的主评测（RUN-0027）、思考子集（RUN-0028）逐条配对，汇总脚本给出配对差与 bootstrap CI。

### 3. 命令行与输出

- 例如 `run_e1.py --subset debiased`、`--subset forced`，或其他清楚的方式。输出目录规则同 E1，带运行编号。
- 支持续跑，`--tiny` 能跑通。
- `--estimate` 能给出两项在 B200 上的用时估算，或者在总结里按 RUN-0027 的实测吞吐推算：去偏读数约是原字母读数的 k 倍，格式强制约是 400 条长生成。

## 允许改动的文件

- `src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`（只能新增条目，旧条目不动；或者另起配置文件）
- `src/adapters/`：只为预填，加参数、不改默认行为
- `scripts/` 下新的汇总脚本
- 对应的测试与 fixture

**不许碰**：`src/e2/`、`src/e3/`、`src/e5/`、`src/synth/`、`src/data/`。另有 DEV-0032 在并行改 `scripts/check_training_interfaces.py`、`src/e3/`、`configs/e3b_train.json`、`configs/e5_prompts.json`、`src/e2/runner.py`，不要动它们。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 两种新读数 `--tiny` 跑通
  - 主评测：`run_e1.py --tiny --task all --subset eval` 的输出与改动前逐字节相同
- 总结里写明：
  - E1 配置 sha256 有没有变；新增条目或新配置的 sha256；
  - 选项拆分规则，以及原顺序逐字节还原的断言；
  - 适配器预填的实现与测试；
  - 结构强制的 logits 处理器；
  - B200 上两项的用时估算；
  - 有效性检查汇总脚本的用法。
