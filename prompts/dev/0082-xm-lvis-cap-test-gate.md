[DEV-TASK 0082] X-M 通过 G-XM 之后的两处小改：LVIS 子集每图最多 3 个类别（XM-2b）；run_x2 凭 gate_xm.json 放行 X-M 的检查点跑 RefCOCO 的 W2 test 子集（D165 第五节、D177）

## 你的 worktree（不是 Agent 自动建的那种）

- 路径 `/Users/tim/Research/CVPR2027/.claude/worktrees/dev0082`，分支 `dev0082`，建在 origin/main 671599d77。
- 稀疏签出，不含 `experiments/`：开发机磁盘只剩约 6 GB，不要把它签出来。
- 所有命令都先 `cd` 到这个路径。不要碰主仓库 `/Users/tim/Research/CVPR2027` 的任何文件，也不要碰另一个 worktree `dev0083`，那里有另一个开发 agent 在做 Rex-Omni。
- 不 commit、不 push。
- 测试：`cd <worktree> && HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/ -q -k "<相关>"`。不要用 conda run，太慢。
- 交回前跑一次全量：`HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/ -q`。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/lead_answers/2026-11-03-xm-directive.md`（第二、五节）
- `PREREGISTRATION.md` 的 XM-2、XM-2a、XM-2b
- `DECISIONS.md` 的 D147、D165、D177
- `src/xm/data.py`（`lvis_pairs`、`_pairs`、`hash_rank` 的用法、`FINAL_GATED`、`check_xm_gate`）、`src/xm/gate.py`
- `scripts/run_x2.py`（`check_test_gate`、`TEST_APPROVAL`，以及 test 划分怎么放行）、`prompts/server_hpc/0092-hpc-06-x-test/x_test.sh`

## 任务

### 1. LVIS 子集每图最多 3 个类别（预注册 XM-2b，原文："每张图在它的正类别里按 hash_rank（salt `xm_lvis_cats_v1`）最多取 3 个类别"）

- 在 `src/xm/data.py` 里实现，在载入 `lvis_sub` 的对时生效：
  - 每张图的正类别（有标注实例的类别），用 `src/data/subsets.hash_rank` 排序。键是类别 id 的十进制字符串，purpose 里带上 image_id，salt 用 `xm_lvis_cats_v1`。取前 3 个。
  - 常量写成 `LVIS_MAX_CATS_PER_IMAGE = 3`、`LVIS_CAT_SALT = "xm_lvis_cats_v1"`。
- 不改 `configs/xm_splits.json`，它的 sha256 已写进预注册。上限只在载入时生效。
- 统计里记上限前后的对数与被去掉的对数。
- 非穷尽类别的处理不变：没配上的预测忽略（DEV-0080）。
- COCO 的划分，以及 `stats_pairs` 对 COCO 的结果，一字不变，用测试钉住。

### 2. run_x2 凭 gate_xm.json 放行 X-M 的检查点跑 test

- 现在 test 只认 G-EX：`--test-gate <gate_explore.json>`，加 `--test-approval D147`。这一套一字不变，现有测试全部照过。
- 另加一条路，给 X-M 用：
  - 命令行是 `--xm-gate <gate_xm.json> --test-approval D165`。D165 第五节是负责人批准 X-M 通过后上 test 的原文。
  - 核对用 `src/xm/gate.check_xm_gate(path, checkpoint=<--checkpoint>)`：结论"通过"、非彩排、判定点全真、candidate 的 manifest = 本次检查点的 manifest。
  - 设置必须是 X-2 的设置。
  - 运行身份里记 gate_xm 的路径、sha256、批准编号。
- `--xm-gate` 与 `--test-gate` 不能同给；给了 `--xm-gate` 却不是 test 划分，也拒绝。
- 其余划分（explore_val、devtune800、devcheck2……）的行为不变。

### 3. 测试

- **LVIS**：合成的 LVIS 文档上，每张图至多 3 个类别；同样的输入结果确定；不到 3 个类别的图全保留；COCO 划分不受影响。
- **run_x2**：
  - X-2 原有的 test 放行测试全部照过；
  - 新路径在这几种情况下都拒绝：结论不是"通过"、彩排、判定点有假、检查点不符、批准编号不是 D165、与 `--test-gate` 同给、不是 test 划分；
  - 合格时放行，运行身份里有 gate_xm 的记录。
- 用 tiny 跑一次 run_x2 的 test 划分，走新路径：用 tests 里已有的 fixture 造 gate_xm.json 与 tiny 检查点，不要另起一套。

## 不做

- 不改 X-2 的配置、样本、结果、提示。
- 不改 gate_xm.py、gate_explore.py。
- 不写服务器驱动，由管理员写。

## 完成总结要写

- 改了哪些文件，附关键 diff；
- LVIS 上限的具体写法，键与 purpose 的格式；
- 给管理员的服务器命令行：
  - X-M 在 W2 test 的 5 个划分上，设置 x2_spline_trimap 与 x2_ellipse_trimap，同 RUN-0092；
  - X-M 与基座在 lvis_sub 上，同 RUN-0109 的写法；
- 测试结果原文；
- 待裁的地方。

尽量 2 小时内完成。
