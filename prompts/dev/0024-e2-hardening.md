[DEV-TASK 0024] E2 开跑前的加固（独立审查的发现）+ 训练接口核对的 v1 判据（D50）+ base 的 trainable_token_rows 桩（D45-5）

## 背景

- E2 开跑前的独立审查（D53，报告原文存在 `docs/reviews/2026-10-02-e2-hidden-state-review.md`）没有发现会让结论出错的问题，但提了下面几条。本任务把它们做掉。
- 一并做两件小事：
  - D50 冻结的训练接口核对判据（v1）写进脚本；
  - D45-5 的遗留：`base.ModelAdapter` 补 `trainable_token_rows` 的桩。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 审查报告全文（上面的路径）、`DECISIONS.md` 的 D45、D49、D50、D52、D53
- `src/e2/`、`src/data/train_loaders.py`、`src/data/train_dedupe.py`、`scripts/check_training_interfaces.py`、`src/adapters/base.py`

## 任务

### 1. 去重剔除清单（审查的小问题 A、B、C；硬约束 7）

- **A**：`require_dedupe=True` 时，`set_name` 必须是剔除清单 `sets` 里的一个键。缺键就报错，说明这个训练集没有比较过。fixture 补上四个训练集的键，并加一条测试。
- **B**：
  - 剔除清单写上 `eval_sets`（参与比较的评测集名单）、各输入清单的摘要（sha256、主机、集合与张数）和 `max_hamming`；
  - `run_e2` 开跑前断言：当前任务的评测集在 `eval_sets` 里，不在就报错；
  - `config.yaml` 记下剔除清单的 `max_hamming` 与 `eval_sets`。
  - 旧格式的清单（没有 `eval_sets`）照样能读，但 `run_e2` 用真模型时拒绝。
- **C**：`load_nyu_depth_train` 加一条断言：每条的基础场景名不在官方 test 场景集合里（D11-2）。

### 2. 网格顺序的确定性测试（审查 §3 的建议 a、b、c）

- **(a) 按模型自己的元数据核对顺序**，确定性，不靠 argmax：
  - Qwen：第 k 个图像 token 的 M-RoPE (H, W) 等于 (k // w, k % w)，处理器里每 m² 个连续 patch 落在同一格；
  - Gemma：由 `image_position_ids` 算出的 pooler 核下标是行主序；
  - InternVL：对下标张量做 pixel_shuffle，第 k 个 token 的 4 个 patch 正好是第 (k // 16, k % 16) 格的 2×2 窗口。
  - 覆盖 640×480、480×640、683×512 三种尺寸。写法参考审查脚本：`docs/reviews/2026-10-02-e2-scripts/order_source.py`，只读。
- **(b) Qwen 与 InternVL：全格 × 每个子 patch 的白块扫描**，两种朝向，作为慢测试。单个测试超过 30 s 的话，标成 slow，并默认跑一个子集。
- **(c) Gemma**：扫描放在 pooler 输出上，或者旁路编码器。不在 `vision_features` 上用 argmax，原因见审查报告。
- 现有的 5 格白块测试保留。

### 3. 解读用的两个读数（审查的 I1、I2）

- **I1：最后一层归一化之前的读数**：
  - 适配器的 `hidden_states` 加一个可选的层键，取最后一个 decoder 层的原始输出（final norm 之前），例如 `layers=[..., "pre_norm_last"]` 或一个单独的关键字参数，接口写明；
  - 三个家族都实现，测试核对它等于最后一层的原始输出，且 norm(它) 等于层号 N 的读数；
  - E2 的特征键是 `L{N}_pre`，默认加进层集合，曲线上单列。
- **I2：按光栅位置分开的指标**：
  - E2 的评测另外按图像上半与下半各报一遍标题指标，并按 token 光栅下标的四分位各报一遍；
  - 深度、语义分割、法向按像素所在的行或位置划分，指代分割按目标质心划分；
  - 写进 metrics 与 README 的表，不改标题指标。

### 4. 训练接口核对的 v1 判据（D50）

- 把 `scripts/check_training_interfaces.py` 的判据改成 D50-3 的 v1：
  - **restricted**：教师强制 ≥ 0.999；关思考时 exact_rows ≥ 0.95；不一致处 logit 差的最大值 < 2.0。
  - **merge**：教师强制 ≥ 0.98；不一致处差的中位数 ≤ 0.5；生成的逐位一致率只报不判。
  - **前提**：只在开思考时判，与 base 相同的行 ≤ 0.5。
- 判据版本号写进输出。
- 用 RUN-0021 的 4 份 JSON（`experiments/E3/0021_train_iface_check/`）重新判一遍，作为测试：4/4 通过。

### 5. `base.ModelAdapter.trainable_token_rows` 的桩（D45-5）

照 base 里其他训练接口的写法：占位方法，报 `NotImplementedError`，附接口说明。

## 允许改动的文件

- `src/e2/`、`scripts/run_e2.py`、`configs/e2_probes.json`：只为第 1、3 条
- `src/data/train_loaders.py`、`src/data/train_dedupe.py`、`scripts/data/dedupe_report.py`：只为第 1 条
- `src/adapters/`：只为第 3 条 I1 与第 5 条，现有行为不变
- `scripts/check_training_interfaces.py`、`src/adapters/training.py`、`src/adapters/checkpoint.py`：只为第 4 条
- 对应的测试与 fixture

**不许碰**：`src/e1/`、`src/e5/`、`src/synth/`、`configs/e1_prompts.json`、`configs/e5_prompts.json`、E1 的落地源。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `run_e2.py --tiny --task all`
  - `check_training_interfaces.py --tiny` 三家族
- 总结里写明：
  - 新的剔除清单格式；
  - `pre_norm_last` 的接口；
  - 按位置分开的指标的定义；
  - 慢测试的用时；
  - v1 判据对 RUN-0021 的重判结果。
