[DEV-TASK 0037] E5 主比较的训练框架：从不同起点（基座 + 预热、E3b、E3a）在 E5 训练集上做 LoRA 微调，学习曲线的数据量档，训练后用 E5 的评分在 test 上评测

## 背景

- E5 的主比较（简报 §8.8；E5 预注册 §3、§5；D36-5、N5）：在**同一份 E5 训练集、同样的预算**下微调几个起点，比较学习曲线。
  - 基座 + 网格格式预热：对照，DEV-0035 已实现预热；
  - E3b 模型：主比较；
  - E3a 模型：第二条证据。
- 负责人 D72-4：E5 主比较先做 2B，3 个种子，在 B200 上训练与评测，2B 的 E3 模型一出来就开始，约 10 月 11–12 日。所以这个框架现在就要做好。
- 现状：`src/e5/` 只有零样本、思维链、平凡基线、外部预测的评测与评分，没有训练。E3b 的训练循环、网格 token、受限解码、检查点都在 `src/e3/`，可以复用。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §3.10、§7.4、§8.8；`PREREGISTRATION.md` 的 E5（全文）与 E3 草稿 v0
- `DECISIONS.md` 的 D23、D24、D28-13、D36、D39、D59-4/5、D61、D62、D65、D66-5、D70、D71、D72
- `src/synth/`（生成器、划分、`TASK_GENERATOR_VERSIONS`）、`src/e5/`（runner、scoring、protocol、prompts）、`configs/e5_prompts.json`
- `src/e3/`（train、tokens、targets、data、warmup、evaluate、merge_check）、`scripts/train_e3b.py`、`scripts/eval_e3b.py`、`src/readout/`（E3a 检查点的格式）

## 任务

### 1. E5 训练数据（`src/e5/train_data.py`）

- 从生成器的 **train 划分**取样。划分按场景种子与物体身份（D23-2），与 val、test 不相交，加一条测试钉住。
- 五类任务 a–e。a–d 有图像，e 是纯文本。
- **数据量档**：1K、4K、16K、64K，暂定，写进配置。各类等量，按档内均匀覆盖难度档。
- 小档是大档的前缀，同一个种子下嵌套，这样学习曲线上的点可以配对比较。
- 还要有"只用 a–d 训练"的配置，用于图像到文本的迁移：在 e 上测（简报 §8.8 第 3 条）。
- 样本由 (数据种子, k) 唯一决定，无状态。d（physics）只在规范平台上生成（D24），非规范平台上只允许 `--tiny`。

### 2. 输出格式：训练后的系统用像素 token 网格（简报 §8.8）

- 目标是 32×32 的 E3b 网格 token 序列，与 E3b 同一套 token、同一种格式，`src/e3/targets`。
  - c、d、e 是二值 mask，用二值 token。
  - a、b 的起伏网格：背景加 32 档起伏（D28-13；E5 提示配置里零样本用 33 个符号）。要映射到 E3b 的网格 token：背景用二值的 0，32 档起伏用 64 档深度 token 里的哪 32 个，或者另一种映射。选一个，写明理由。
  - 评分时把 token 映射回 E5 的格子表示，用 `src/e5/scoring` 的同一套评分，指标与零样本完全可比。
- **E3a 起点没有网格 token**：先加上与 E3b 同一套网格 token 行，用与 E3b 相同的初始化，再微调。写明这一处与 E3b 起点的差别，供预注册讨论。
- 提示：用 E5 提示配置里任务的原文，加 E3b 网格指令的结尾句；放进配置，不改 `configs/e5_prompts.json` 的已有条目。

### 3. 起点与训练（`scripts/train_e5.py`、`configs/e5_train.json`）

- `--init`：
  - `base_warmup`：DEV-0035 预热的检查点；
  - `e3b`：E3b 检查点；
  - `e3a`：E3a 检查点，加网格 token 行；
  - `base`：基座，加网格 token 行，不预热。只作诊断。
- **起点检查点的 LoRA 怎么处理**：默认先合并进基座，作为新的起点权重，再注入一个新的 LoRA。理由是预算对齐：每个起点在 E5 上都训练同样的一份新 LoRA。这一点必须做到：
  - 合并之后的模型与检查点合并评测时一致（D50 的合并核对）；
  - 不把合并后的整模型存盘（D34）。存的是"起点检查点的引用 + 新 LoRA"，评测时按顺序合并。

  如果你认为"继续训练原 LoRA"更合理，写明理由，二选一实现。
- **训练**：LoRA r32，同 E3b，受限交叉熵，只在网格 token 上算损失；网格 token 行继续训练，fp32 主副本。
  - 学习率、步数、批大小写进配置，暂定。
  - 学习曲线的每个数据量档按**固定的 epoch 数**训练，还是按固定步数，选一个，写明理由。预注册会冻结它。
- **种子**：每个运行一个，命令行给出。续跑、记账、检查点同 E3b。
- `--n-train` 选数据量档，`--tasks` 选任务子集。

### 4. 评测（`scripts/eval_e5_trained.py`，或者给 `run_e5.py` 加 `--checkpoint`）

- 加载：起点检查点 + E5 的 LoRA，按顺序合并，先做合并核对。
- 受限解码到网格 token，单轮，max_new_tokens = 1057。
- 在 E5 的 test 划分上评分，用零样本的同一套评分与判据（D59-4、D72-1）：mask IoU、起伏误差、三条基线的配对 CI。b 的判定统计环绕的 [20,180] 三档（D72-1）。
- 输出：`experiments/E5/<编号>/trained/<起点>/<模型>/n<数据量>/seed<k>/<任务>/`，支持续跑。

### 5. 测试（CPU 小模型，`--tiny`）

- 四种起点在三个家族上都能训练、评测跑通。用 DEV-0025、DEV-0035、DEV-0034 的小检查点当起点。
- 过拟合检查：每类几条样本，训练后在训练样本上接近满分。
- 训练集与 test 划分不相交。
- 嵌套数据量档：1K 是 4K 的前缀。
- 评分与零样本同一套代码：同一个网格，两条路径给出相同分数。
- 起点 LoRA 合并后的一致性：合并核对通过。坏起点（合并前后分家）时拒绝训练。
- 续跑逐字节相同（CPU）。

## 允许改动的文件

- `src/e5/`（新增训练相关的模块；已有的评测与评分逻辑只读复用，不改已有行为与输出）、`scripts/train_e5.py`、`scripts/eval_e5_trained.py`（或 `run_e5.py` 的 `--checkpoint`，只加选项、不改已有行为）、`configs/e5_train.json`（新）
- 只读复用 `src/e3/`、`src/readout/`、`src/synth/`。要改它们，先在总结里说明，只做不改变已有输出的小改动。
- 对应的测试与 fixture

**不许碰**：`configs/e1_prompts.json`、`configs/e1_posthoc.json`、`configs/e2_probes.json`、`configs/e3a_train.json`、`configs/e3b_train.json`、`configs/e5_prompts.json` 的已有条目、`src/adapters/`、`src/e1/`、`src/e2/`。另有 DEV-0036 在并行改 `src/e1/`、`scripts/run_e1.py`，不要动。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 开发机磁盘紧张，剩约 10 GB：临时输出放 pytest 的 tmp 或 worktree 里，用完即删。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 四种起点的 `--tiny` 训练与评测
- 总结里写明：
  - 训练集的构造与划分核对；
  - 起伏到网格 token 的映射与理由；
  - E3a 起点加网格 token 的做法；
  - 起点 LoRA 的处理方式与理由；
  - epoch 还是步数；
  - B200 上的用时推算：2B，四个数据量档 × 三个起点 × 3 个种子，按 RUN-0026 的 E3b 实测，B200 约是 A800 的 2 倍，这个倍数待实测；
  - 所有留给预注册定的选择。
