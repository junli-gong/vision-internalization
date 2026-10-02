[DEV-TASK 0019] LISA-7B-v1 在 E5 上的推理脚本（外部对照，隔离运行）+ 拒答正则单一来源 + E5 提示配置搬到 configs/

## 背景

- D36-5：LISA-7B-v1 作为 E5 的外部对照，是演示，不是成功条件。
  - 预期它只能分割当前图像里的物体，所以在 a、b、d 上不高于原样复制，在 c 上不高于"可见部分"。
  - 它的输出绝不进入我们的数据、标签或奖励（硬约束 1）。
- RUN-0015（`reports/0015-lisa-smoke.md`）在 B200 上跑通了 LISA 的独立 venv。必须用两处处理，见 report §4：
  - **AddedToken 的 `__str__` 补丁**：venv 里没有 tokenizers；
  - **vision_tower 改成本地快照路径**：离线时 Hub id 解析不到。
- DEV-0018（D39）定了外部预测的目录格式（`src/e5/external.py`），以及导出评测集输入的命令 `run_e5.py --export-inputs`。
- 本任务写 LISA 那一侧的推理脚本：读导出的输入，写出外部预测目录。
- 另顺手做两件小事：
  - 拒答正则在 E1、E5 各有一份，合并成单一来源；
  - E5 的提示配置搬到 `configs/`。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D36（第 5 条）、D39
- `reports/0015-lisa-smoke.md` 全文，以及 `experiments/P0/0015_lisa_smoke/probe_scripts/` 里的冒烟脚本（`p19_lisa_smoke.py` 等）
- `src/e5/external.py`、`scripts/run_e5.py` 的 `--export-inputs`、`tests/test_e5_external.py`
- LISA 官方仓库 `chat.py` 的推理流程（commit 见 RUN-0015 的 `lisa_commit.txt`）：在服务器上，或按那个 commit 在网上读；只读，不改，不下载权重

## 任务

### 1. `scripts/external/lisa_e5_infer.py`

在 LISA 的 venv 里运行（`$PROJ/ext/lisa/venv`，transformers 4.31），不在我们的项目 venv 里。

```bash
$PROJ/ext/lisa/venv/bin/python scripts/external/lisa_e5_infer.py \
  --lisa-src $PROJ/ext/lisa/LISA --inputs <export 目录> --out <外部预测目录> \
  --lisa-weights <快照路径> --clip-weights <快照路径> [--limit N] [--resume]
```

- **依赖**：
  - 只 import LISA 自己的代码（经 `--lisa-src` 加进 sys.path）与它的依赖；
  - **不 import 我们仓库 `src/` 下的任何模块**，因为两个环境的依赖版本不同。
  - 外部预测格式的常量（标记文件名、字段名）在脚本里另写一份；由测试核对它与 `src/e5/external.py` 一致。
- **加载**：与冒烟脚本同一条路径，bf16，GPU：
  - 两处处理照搬：AddedToken 补丁只在没装 tokenizers 时打；vision_tower 改成本地路径；
  - 每次运行记下用了哪几处。
- **提示**：一个固定模板，写成脚本里的常量，并记进输出目录的 `run_config.json`：

  ```
  {instruction} Please output segmentation mask.
  ```

  - `{instruction}` 是导出的 E5 指令原文：其中已经说明了变换和被点名的物体。
  - 后半句是 LISA 训练时用的说法。
  - 套 LISA `chat.py` 的对话模板与图像预处理，不改。
- **生成**：greedy，`max_new_tokens=512`，和冒烟脚本一样。
- **mask**：
  - 取第一个 `[SEG]` 的 mask，阈值与 `chat.py` 相同（> 0），尺寸为原图；
  - 没有 `[SEG]`，或 mask 全空时，写一行 failure，原因分开记：`no_seg`、`empty_mask`、异常类名。
- **任务范围**：a、b、c、d。e 类没有图像，跳过，在 `run_config.json` 里写明。
- **输出**：符合 `src/e5/external.py` 的目录格式：
  - 标记文件写明系统 `lisa-7b-v1`、LISA 的 commit、权重 revision、提示模板、用到的处理；
  - 每个任务一个 `<任务>.jsonl`；
  - mask 写成**未压缩 RLE**，按 COCO 的列优先顺序，只用 numpy。LISA 的 venv 没有 pycocotools。
  - 另记每条的墙钟、生成的文本、`[SEG]` 个数。
- **断点续传**：`--resume` 时跳过已经写出的 id，重复 id 报错。
- **隔离**：
  - 只读 `--inputs`、`--lisa-src`、两个权重目录；
  - 只写 `--out`；
  - 不碰 `$PROJ/data` 与我们的任何训练或数据目录。

### 2. 测试（开发机上跑不了 LISA，用替身）

- 用一个假模型替身代替 LISA：给定输入，返回固定的 mask 和文本，跑通整个脚本流程，包括续跑与 failure 行。
- 写出的目录用 `src/e5/external.py` 的读取函数读回来：
  - mask 逐像素相同；
  - failure 原因保留；
  - 标记文件通过校验。
- 脚本里另写的常量与 `src/e5/external.py` 一致。
- 静态检查：脚本不 import `src/`。
- 隔离测试：`tests/test_e5_external.py` 的静态允许清单加上这个脚本，并写明理由：它是外部系统那一侧，产出外部预测，不读我们的训练数据。动态隔离测试保持不变。
- `run_e5.py --export-inputs`（`--tiny` 数据）→ 替身脚本 → `run_e5.py --system external` 的端到端流程，在开发机上跑通。

### 3. 拒答正则的单一来源

- 比较 E1（`src/e1/scoring.py` 的 REFUSAL_RE 一类）与 E5（DEV-0018 另写的一份）的拒答正则：
  - **逐字相同**：搬到 `src/parsing/refusal.py`，两处都 import 它，行为不变。加一条测试，核对 E1 现有测试里的拒答例子判定不变。
  - **不同**：不改，在总结里列出差异，由管理员定。
- E1 的提示配置 sha256 不能变（`4dd027b4…`）。

### 4. E5 提示配置搬到 `configs/e5_prompts.json`

- 内容逐字节不变（sha256 不变），只改 `src/e5/prompts.py` 的 `PROMPTS_PATH` 与相关测试。
- 已有的运行目录里记的路径不追溯修改。

## 允许改动的文件

- 新建：`scripts/external/lisa_e5_infer.py`、`tests/test_e5_lisa_infer.py`、`src/parsing/refusal.py`
- `tests/test_e5_external.py`：只改静态允许清单
- `src/e1/scoring.py`、`src/e5/scoring.py`：只改拒答正则的来源
- `src/e5/e5_prompts.json` → `configs/e5_prompts.json`；`src/e5/prompts.py`：只改路径
- 对应的测试与 fixture

**不许碰**：`src/adapters/`、`src/synth/`、`src/data/`、`src/health/`、`configs/e1_prompts.json`，以及 E1、E5 的其他逻辑。

## 环境与验收

- 用 conda 环境 `visint`，不装包，不下载权重，不联网下载数据。为了读 LISA 的代码可以联网读 GitHub 上的源码。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 第 2 节的端到端替身流程
- 总结里写明：
  - 脚本的完整用法与服务器上的运行步骤（导出 → LISA → 评分，三条命令）；
  - 提示模板；
  - 输出格式举例；
  - 拒答正则的比较结果；
  - 预计在 B200 上的用时：按冒烟实测的推理 0.8 s/条，评测集约 1,500–1,900 条。
