[DEV-TASK 0081] X-M 的训练与判定：COCO 两条训练流、训练配置、50 步吞吐探测、判定脚本 gate_xm

## 背景

- 负责人新开 X-M 线（D165，要点原文 `docs/lead_answers/2026-11-03-xm-directive.md`，先读全文）。预注册在 `PREREGISTRATION.md` 的 XM 节，先读 XM-4、XM-5。
- 做法同 X-2：单一 LoRA，从基座开始，配方同 `configs/x2_train.json`。X-2 的各条样本流与每步条数不变，另加 COCO 的两条流：
  - **列框**：调用 1，给图与类别名，目标是该类全部实例的框列表；
  - **每实例样条**：调用 2，给图、类别名与框列表，目标是每个框一条 16 控制点 B 样条。
- 并行在做的两个任务：
  - DEV-0079：数据层 `src/xm/data.py`，含训练目标（框列表与每实例样条）的分片 jsonl；
  - DEV-0080：提示 `src/xm/prompts.py`、指标 `src/xm/metrics.py`、运行器 `scripts/run_xm.py`。
- 接口见下。你按接口写，测试用自己的 fixture。合入时以前两个任务的实现为准，接口名不要改。

## 开工前（省盘）

开发机磁盘紧，另有两个开发 agent 同时在跑。开工前在**你自己的 worktree 里**执行 `rm -rf experiments/`，可省约 6 GB。不要 `git add` 这些删除，也不要动主仓库。只跑你自己的测试与 X-2 训练的相关测试。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/lead_answers/2026-11-03-xm-directive.md`、`PREREGISTRATION.md` 的 XM 节、`DECISIONS.md` 的 D165 与 D166、`DECISIONS_NEEDED.md` 的 N47
- X-2 的训练：`scripts/train_w2c.py`、`src/w2/w2c_train.py`（样本流、日程、按流归一的损失、`--stop-after-steps`、续跑）、`configs/x2_train.json`、`src/explore/x2_data.py`
- X-2 的判定脚本：`scripts/gate_explore.py`（gate_explore.json 的键、彩排标记、判定点核对）
- 配对 bootstrap：`src/stats.py`

## 接口（与 DEV-0079、0080 共用）

- 数据：`src/xm/data.py` 的 `XMPair`、`XMInstance`、`load_pairs(split, data_root)`。训练目标的分片 jsonl 每行至少含：
  - `pair_id`、`image`、`width`、`height`、`category`；
  - `boxes`：0–1000 网格整数，阅读顺序，最多 30 个；
  - `splines`：与 boxes 一一对应，每条 16 个 `[x, y]`；
  - `fit_iou`。
- 提示：`src/xm/prompts.py` 的 `call1_prompt(category)`、`call2_prompt(category, boxes)`，以及两个目标的序列化函数 `format_boxes(boxes)`、`format_splines(splines)`。你放同名的最小桩，合入时以 DEV-0080 为准。
- 判定结果文件 `gate_xm.json` 的键（DEV-0080 的运行器读）：`verdict`、`rehearsal`、`checks`、`candidate.manifest_sha256`。

## 任务

1. **COCO 训练样本**（`scripts/build_xm_data.py`）：
   - 从 DEV-0079 的训练目标里，按 hash 抽够 2,000 步所需的对数，再留 20% 余量。
   - 生成两条流的样本行：
     - **列框**：提示用 `call1_prompt`，目标用 `format_boxes`；
     - **每实例样条**：每条样本最多 12 个框。超过的，按阅读顺序切成几条，每条的提示只列它那几个框，目标是它们的样条。这是为了控制显存（A100 40 GB）；推理时仍一次最多 30 个。
   - 输出分片与 manifest，可续跑。`--out` 末级目录名以四位运行编号开头。
2. **训练流与配置**：
   - 在 `src/w2/w2c_train.py` 的样本流里加 `coco_det`、`coco_inst` 两种，加法照 X-2 加 spline、correct 两流的方式。**X-2 原有的流、条数、损失归一一字不变**，用测试钉住 X-2 的配置与样本逐字节不变。
   - 新配置 `configs/xm_train.json`：等于 `configs/x2_train.json`，另加 `coco_det: 6`、`coco_inst: 6`（每步），损失权重 1，按流归一；version 写 `xm-train-v1`。
   - `scripts/train_w2c.py` 加 `--xm-data <build_xm_data 的输出>`。
   - 每条流可以单独设微批大小：`coco_inst` 缺省 2，其余照 X-2，以防显存超出。
3. **吞吐探测**：沿用 `--stop-after-steps 50`，并在训练日志里按流记每步用时与显存峰值，方便驱动取第 5 步以后 step_s 的中位数。
4. **判定脚本 `scripts/gate_xm.py`**（预注册 XM-5，**照录 D165 第五节原文实现，不改阈值**）：
   - 输入：
     - 判定集上 X-M 的 `xm_spline_trimap`、`xm_box_grabcut` 与 `base_det` 的叶目录（DEV-0080 的格式）；
     - 2,300 句开发数据上 X-M 与 X-2 seed 0 的指代分割叶目录（`run_x2` 的格式，同 `scripts/mgr/seed_variance.py` 读的那些）；
     - 大题量 `lb_compare`：X-M 对 X-2 seed 0，同节点。
   - 四条：
     - (1) 实例分割 F1：X-M 主设置 − `xm_box_grabcut`，CI 下界 > 0；
     - (2) 检测 F1：X-M − `base_det`，CI 下界 > 0；
     - (3) cIoU：X-M − X-2，CI 下界 > −0.02；
     - (4) MMLU-Pro 与 IFEval：X-M − X-2，CI 下界都 > −2 点。
     - (1)(2) 用 DEV-0080 的按图像重抽的配对 bootstrap；(3) 用逐句配对 bootstrap，口径同 `scripts/mgr/x_test_table.py`。三者都是 95%、10,000 次、种子 `src/constants.BOOTSTRAP_SEED`。(4) 取 `lb_compare` 里的 CI。
   - 输出 `gate_xm.json` 与 `gate_xm.md`：每条的差与 CI、是否满足、总结论"通过 / 不通过"、判定点核对、输入文件的 sha256。
   - `--rehearsal` 时标"彩排，不作数"。
   - 第 4 条的写法另设一个 `--criterion4` 开关：缺省 `d165`（原文）。只有在负责人答复 N47 之后，管理员才会决定是否加别的写法，这里先不实现别的。
   - 判定集的读取要给门名 G-XM（DEV-0079 的守卫）。
5. **测试**（`HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/ -q -k "xm or w2c_train or x2_train"`）：
   - 样本：最多 12 个框的切分、提示与目标的格式；
   - X-2 的配置与样本逐字节不变；
   - tiny 上用 `configs/xm_train.json` 训 3 步不报错，日志有按流的用时；
   - 判定：合成的输入覆盖四条各自通过与不通过；彩排标记；缺输入时拒绝；
   - `--out` 不带运行编号时拒绝。
6. 在总结里给出服务器命令：
   - 建 COCO 训练样本；
   - 50 步探测；
   - 正式训练：9B，06，`srun --jobid=157296`；
   - 判定；
   - 给出用时估计，按 X-2 的 17.4 s/步与样本条数折算，写明依据。

## 不做

- 不改 X-2 的任何配置、样本与结果；不改 `gate_explore.py`。
- 不写数据层（DEV-0079）与推理、指标（DEV-0080）。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 配置的 diff（相对 x2_train.json）；
- 服务器命令与用时估计；
- 测试结果原文；
- 待裁的地方。

尽量 3 小时内完成。
