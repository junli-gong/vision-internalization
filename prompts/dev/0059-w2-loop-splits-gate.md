[DEV-TASK 0059] W2 的回路、新的开发-检查集、对照与判定脚本（D113；接口见 docs/w2_plan.md §0、§3）

## 背景

- W2-a 分两轮：
  1. 提出：E1 协议的原生框，可选再给几个点；
  2. 挑选：K 个候选编号后叠回原图，LLM 只回答一个编号或"都不对"。
- "都不对"或解析失败时，交 1 号候选，即 W1-F。
- 去留规则（D113 第三节，不得修改）：
  - 新的开发-检查集 500 句，9B；
  - 主配置的 cIoU 对 W1-F 的配对 bootstrap **97.5%** CI 下界 > 0 才算通过。
- 候选生成器与渲染由并行的 DEV-0058 做（`src/w2/candidates.py`、`src/w2/render.py`，签名见 `docs/w2_plan.md`）。本任务先用符合签名的假实现写测试，合入时换成真的。
- W1 的代码完整保留，在其上开发，不改 W1 的行为。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/w2_plan.md`（全文）、`docs/lead_answers/2026-10-22-w2-directive.md`
- W1：`src/w1/{config,data,loop,runner,baselines,scoring}.py`、`scripts/run_w1.py`、`scripts/w1_gate.py`，以及它们的测试；`configs/w1.json`、`configs/w1_splits.json`
- 适配器：`src/adapters/chat.py`、`src/adapters/qwen35.py` 的 `chat_turn_batch`（D104）
- E1：`src/e1/runner.py`、`configs/e1_prompts.json`（refexp_box 的协议）
- 去重与子集：`src/data/subsets.py`（`hash_rank`）、RefCOCO 的加载器与 `subset='eval'` 的去重规则
- DECISIONS.md 的 D103、D104、D107、D108、D111、D113

## 任务

1. **配置 `configs/w2.json`**，`frozen: false`：
   - 提出：`mode` 取 `box`（E1 协议，默认）或 `box_points`；`box_points` 的提示原文、点的个数上限。
   - 候选：`CandidateSpec`，默认取 DEV-0058 的 `DEFAULT_SPEC`。DEV-0058 合入前先放占位，合入时由管理员换。
   - 渲染：`layout` 与 `style`。
   - 挑选：提示原文；思考开关；`thinking_budget ≤ 2048`；`answer_budget`；解析规则。
   - 随机挑选的 salt；批参数。
   - `src/w2/config.py` 校验。
   - 提示原文与任务无关：只说明有 K 个编号的候选，问哪个最好地覆盖 `{expression}`，回答编号或 none。不写解题方法。
2. **划分 `configs/w2_splits.json`**，由 `src/w2/data.py` 生成与核对：
   - `devtune800`：W1 的 devtune 300 加 devcheck 500，从 `configs/w1_splits.json` 原样取。
   - **`devcheck2`**：
     - 总体是 `load_refcoco('refcoco', 'val', subset='eval')` 的全部句子（E1 的 `refcoco_unc_val`，10,634 句，sha256 `1783dd7f…`）；
     - **按图像**排除 W1 那 800 句所在的图；
     - 再按 `hash_rank(salt='w2_dev_v1', purpose='w2_devcheck')` 升序取前 500 句；
     - 清单、图像排除规则、各步的条数与 sha256 写进文件。
   - 开发机上没有 RefCOCO 的数据，只有 tests/fixtures。所以：
     - 生成器要能在 HPC 上用 CPU 跑：`python -m src.w2.data --build --data-root … --out <路径>`，由服务器在 RUN 里执行；
     - 开发机上用 fixtures 测试逻辑；
     - 在完成总结里写明服务器上的命令。
   - test 子集：同 W1，即 E1 的 poly 子集，每个划分 1,000 句。
   - **冻结保护**：未冻结的 `configs/w2.json` 拒跑 `devcheck2` 与 test，退出码 2，同 D108。`devtune800` 不受限；`--tiny` 不受限。
3. **回路与运行器**：`src/w2/loop.py`、`src/w2/runner.py`、`scripts/run_w2.py`。
   - 设置：`w2a`、`random_pick`、`oracle_pick`、`w1f`、`ownbox`、`baseline_e1`，见 `docs/w2_plan.md` §3。
   - `w2a` 遇到定位、计数任务时按 E1 单轮作答，即按输出类型路由。
   - 提出一步的框直接复用同一输出目录里的 `baseline_e1` 运行：没跑完就先跑完，照 W1 的 `ensure_box_run`。
   - 候选生成一次，存进 `candidates.jsonl`（每个候选的 RLE、名字、`ok`、参数、耗时），`w2a`、`random_pick`、`oracle_pick`、`w1f` 共用。
   - 挑选一步按批调 `chat_turn_batch`，不给工具。批的组成只取决于顺序与提示长度，同 W1。
   - 记录、续跑、运行身份（`config.yaml`）、`metrics.json`（cIoU、gIoU 与 bootstrap 95% CI）、`compute.json`：都照 W1 的约定。
   - 每条样本还要记：每个候选的 IoU、挑中的编号、`fallback`（none、parse_failure 或空）。
   - `metrics.json` 另报四项：挑中最优候选的比例、挑中 1 号的比例、"都不对"的比例、解析失败的比例。
4. **判定 `scripts/w2_gate.py`**：
   - 规则写死：主配置减 W1-F 的 cIoU 差，配对 bootstrap，统计量是比值之和（同 `w1_gate.py`），**置信水平 0.975**，重抽样 **10,000** 次，种子取 `src/constants.BOOTSTRAP_SEED`；下界 > 0 才算通过。
   - 判定点核对：9B、`devcheck2`、n = 500、非 tiny、不加 `--limit`、`configs/w2.json` 已冻结。
   - 输出 `gate.json`、`gate.md`。不是判定点时标"彩排，不作数"。
   - 两个主配置（W2-a、W2-c）各跑一次；"至少一个通过"不在脚本里合并，由管理员在门报告里写。
5. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w2_*.py tests/test_w1_*.py -q` 全部通过）：
   - 假候选、假适配器：
     - 每种设置的结果；
     - 回退规则；
     - 解析：编号、none、乱写；
     - 随机挑选可复现；最优候选与真值一致；
     - 续跑逐字节相同；
     - 冻结保护。
   - 划分生成：在 fixtures 上的逻辑——按图像排除、`hash_rank` 的顺序、与 W1 不重叠。
   - `w2_gate.py`：
     - 判定点核对；
     - 0.975 的水平与 10,000 次重抽样确实生效；
     - 用构造的数据检验通过与不通过两种情况。
   - `run_w2.py --tiny` 端到端：小模型、CPU、fixtures，全部设置都 rc=0。

## 不做

- 不写候选生成器与渲染的真实现：DEV-0058 做，这里只用符合签名的假实现。
- 不改 W1 的行为：W1 的测试全部照常通过。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 配置的各项与默认值；
- 划分生成的服务器命令；
- 判定脚本的规则原文；
- 测试结果原文；
- 与 `docs/w2_plan.md` 的任何出入。
