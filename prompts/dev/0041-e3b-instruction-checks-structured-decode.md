[DEV-TASK 0041] 急：E3b 的指令检查、网格的结构强制解码、LoRA 缩放的派生检查点；E3a 不看指令的排查（只读）

## 背景（D83，负责人 2026-10-08 的答复，原文 `docs/lead_answers/2026-10-08-n31.md`）

- RUN-0038（2B 种子 0）：E3b 网格单轮的指代分割 cIoU 0.544，但还没做过"去掉指令"的检查。E3a 做了，配对差约 0，即 E3a 不看指令。如果 E3b 也不看，0.544 可能只是"显著物体分割"。
- E3b 的深度网格 0 / 654 有效：贪心受限解码在随机的行上提前停止。负责人定了：**结构强制解码**作为 E3b+C3 与 E5 所有网格任务的正式口径；现有各臂另报它作诊断。
- 负责人要一组 LoRA 缩放的诊断：0.25、0.5、0.75，每档同时量 T 文本、网格指代分割、结构强制下的网格深度。
- 这些都要在几小时内跑到 B200 上。**请优先做第 1、2 节**，第 3 节其次，第 4 节是只读调查，写进总结即可。
- 先读：
  - `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
  - `DECISIONS.md` 的 D50、D65、D80、D83
  - `scripts/eval_e3b.py`、`src/e3/evaluate.py`、`configs/e3b_train.json` 的 eval 节
  - `src/e1/forced.py`：E1 的结构强制（GridStructureProcessor，经 `generate_batch` 的 `logits_processor` 传入）
  - `src/readout/evaluate.py`：E3a 的两项泄漏检查（shuffle、no_instruction），E3b 的"去掉指令"要与它同一个定义
  - `src/adapters/base.py` 的 `generate_restricted` / `generate_batch`、`src/adapters/checkpoint.py`（只读）

## 任务

### 1. E3b 的指令检查（`scripts/eval_e3b.py`、`src/e3/evaluate.py`）

- 新选项 `--instruction-check {none,swap}`，只用于 refseg：
  - `none`：去掉指代表达。定义与 E3a 的 `no_instruction` 相同：同一个模板、同一种去法，从 `src/readout/evaluate.py` 读出来复用，不另写。
  - `swap`：把指代句换成**同一张图里另一个物体**的指代句。
    - 候选：同一张图、不同标注 id 的指代。取法事先固定，例如按样本 id 的哈希在候选里选，种子固定；没有候选的样本跳过并计数。
    - 每条预测同时对两个目标评分：换上去的那个物体，以及原来的物体。
- 输出：
  - 结果写在与正常评测并列的目录，例如 `refseg__instr_none`、`refseg__instr_swap`；
  - metrics.json 给出 cIoU、gIoU：swap 分别对新目标与原目标；
  - 与正常评测在同一批 id 上的配对差与配对 bootstrap CI；
  - swap 的跳过条数。
- `--limit N`：只取评测清单的前 N 条，N 取批大小的整数倍。B200 上用 2,048 条，每项约 10 分钟。
  - 配对比较时，正常评测那一侧取已有的全量结果里同一批 id，不重跑。如果批组成不同导致不能逐条配对，说清楚怎么处理。
- 解码与正常评测相同，默认就是当前的受限解码。另可配第 2 节的 `--decode structured`。
- 运行身份记下检查的种类、候选的取法、跳过条数。

### 2. 网格的结构强制解码（新模块，例如 `src/grid/structured_decode.py`；`eval_e3b.py` 加 `--decode {restricted,structured}`）

- **结构强制**：每个位置的允许集合固定。
  - 第 r 行的格子位置：只允许本任务的值 token（refseg 2 个，depth 64 个）；
  - 行末：只允许行分隔；
  - 最后一个位置：只允许停止 token。
  - 长度恒为 n(n + 1) + 1 = 1057。
  - 不注入任何任务知识，只强制网格的形状（D83）。
- 实现成可复用的 logits processor 或等价的东西，**E5 的网格（二值 c、d、e，起伏 a、b）以后也要用**。所以按"每行的值 token 集合、行分隔 id、停止 id、n"参数化，不写死 E3b 的任务。
- 贪心。批组成与 `restricted` 相同。
- 先确认 `generate_batch(allowed_token_ids=…, logits_processor=…)` 的路径与 `generate_restricted` 在不加结构约束时逐 token 相同（现有的 `restricted_consistency` 应已覆盖），再在其上叠加结构约束。**不改 `src/adapters/`**。
- 运行身份与 metrics 记 `decode`。`--resume` 时解码方式不同就拒绝混写。不给 `--decode` 时与现在逐字节相同。
- 多轮与匹配计算量的单轮也要能用 `structured`：采样时同样逐位置限制。若工作量大，先只支持单轮（`--k-values 1`），在总结里写明。

### 3. LoRA 缩放的派生检查点（`scripts/scale_lora_ckpt.py`，新）

```
python scripts/scale_lora_ckpt.py --checkpoint <训练叶目录>/ckpt/final/model --scale 0.5 --out <新目录>/ckpt/final/model
```

- 写一个新的检查点：LoRA 的增量整体乘以 s（例如只把 `lora_B` 乘 s），网格 token 行与其余内容原样保留。
- manifest 按仓库的规则重写，逐文件 sha256；另记 `derived: {from: 原 manifest 的 sha256, lora_scale: s}`，这样运行身份里看得出它是派生的。用现有的 `src/adapters/checkpoint` 函数写，不另起格式。
- 新检查点能直接给 `eval_e3b --checkpoint` 与 `run_e1 --checkpoint`，两者的合并核对照常做。
- 旁边放一份原叶目录 `config.yaml` 的拷贝，布局同 `scripts/server/train_queue.sh` 的 finalize，这样需要叶目录信息的读取方也能用。
- **测试**：
  - s = 1：合并后的权重与原检查点合并后逐位相同；
  - s = 0：合并后，除网格 token 行以外与 base 逐位相同；
  - s = 0.5：抽一个模块，合并后的增量是原增量的 0.5 倍，fp32 下误差在 1e-6 量级。

### 4. E3a 不看指令的排查（只读，写进总结；不改代码）

负责人要先排除 bug，再定性：

- **查询 token 能不能注意到指令**：
  - 读 `src/readout/` 与适配器构造输入的部分，写清楚 16×16 个查询 token 在输入序列里的位置（图像 token、指令文本、查询三者的先后），以及注意力掩码。
  - 如果查询排在指令之前，因果掩码下它看不到指令，那就是 bug。
  - 在 tiny 模型上做一个最小验证：同一张图、两句不同的指令，查询位置的隐藏状态是否不同。
- **逐层混合权重**：管理员已经读过，不用再做。2B 种子 0 的 `readout.json` 里，softmax 之后 25 层的权重是 0.030–0.044，几乎是均匀初始化的 0.04；50% 深度以后的层合计约 0.5。所以"权重集中在前几层"这条解释不成立。请集中查上一条：查询的位置与注意力掩码。
  - 读出文件在 `/private/tmp/claude-501/-Users-tim-Research-CVPR2027/66adb8f0-dadd-4263-acc8-6bc423ab4433/scratchpad/e3a_2b_s0_readout/`，需要时可读。
- 结论写进总结：是不是 bug、证据是什么、修的话改哪里。不要动手改；要改的话由管理员另开任务（DEV-0042 的 E3a+C3 一并修）。

## 允许改动的文件

- `scripts/eval_e3b.py`、`src/e3/evaluate.py`：只加选项，不改已有行为
- 新模块（结构强制解码）、`scripts/scale_lora_ckpt.py`（新）、第 4 节的只读脚本（放 `scripts/mgr/` 或 `scripts/diag/`）
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/readout/`、`src/e5/`、`src/synth/`、`src/e3/` 中除 `evaluate.py` 以外的文件、所有 `configs/*.json`、`scripts/run_e1.py`、`scripts/train_*.py`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- **两套全量测试要并行跑时，不要同时用两个 `conda run`**：会互相覆盖输出。直接用 `/opt/miniconda3/envs/visint/bin/python -m pytest …` 与 `PATH=/opt/miniconda3/envs/visint/bin:$PATH bash scripts/pytest_server_parity.sh …`，各写各的日志。
- 验收：
  - 普通与 server-parity 两套全量测试，在最后一次改动之后各跑一遍；
  - tiny 彩排：`train_e3b --tiny`，然后 `eval_e3b --tiny` 的 `--instruction-check none`、`swap`、`--decode structured` 各一次；`scale_lora_ckpt.py` 0.5 之后再 `eval_e3b --tiny` 与 `run_e1 --tiny --checkpoint --limit 4` 各一次。
- 总结里写明：
  - 两项指令检查的定义与 swap 的候选规则；
  - 结构强制的实现与核对（不加约束时与 restricted 逐 token 相同）；
  - 派生检查点的格式与三条测试的结果；
  - 第 4 节的结论；
  - 所有偏离。
