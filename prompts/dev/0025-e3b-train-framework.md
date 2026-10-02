[DEV-TASK 0025] E3b 像素 token 通道的训练框架（第一部分：词表、目标、数据管线、训练循环、C3 对照；评测另派）

## 背景

- 简报 §8.5（E3b，检验 H1，与 E3a 并列）：
  - **输出**：指令在前。LLM 按光栅顺序输出 32×32 的网格，行间用分隔符；网格 token 是新增的一小组 token；解码时 logits 只在这组 token 上计算。
  - **训练数据**：
    - GT 网格：从零预测；
    - 加噪过程生成的修正对：带错的叠加图作输入，干净网格作目标，严重程度按噪声日程采样。
  - **深度**：把 0/1 换成 64 档对数深度。
  - **对照 C3**：同样的图像、同样的 LoRA，只用稀疏的文本监督（框、多边形、相对深度陈述），匹配曝光量。
  - 尺寸与种子同 E3a：先做 0.8B、2B、4B，每个配置 3 个种子。
- 硬约束（简报 §4）：
  - 能力必须长在 LLM 权重里，不用任何学习型 CV 专家模型；
  - E3b 最多新增 300 个 token，不加额外解码器，任何架构改动先批准；
  - 多轮流程要做计算记账；
  - 训练与评测用同一套叠加渲染参数（§12）；
  - 数据按 D11 划分并去重。
- 已经就绪的部件：
  - **适配器的训练接口**（D42、D45、D50、D54）：
    - `training_forward`；
    - 唯一的 `restricted_logits`、`generate_restricted`、`chunked_lm_loss`；
    - `add_grid_tokens`、`token_table`、`restore_tokens`、`trainable_token_rows`；
    - `lora_target_modules`、`inject_lora`、`save_trained`、`load_trained`。
  - **执行器与加噪**（D14）：`src/executors/overlay.py`（叠加渲染）、`src/noising/mask_noise.py`（`corrupt`、`sample_severity`）。
  - **编解码**：`src/grid/codec.py`（`mask_to_grid`、`depth_to_bins`、64 档对数深度）。
  - **训练数据与加载器**（D49、D54）：RefCOCO 训练图、NYU BTS 深度、ADE20K；没有去重剔除清单就报错。
- 本任务只做框架与 `--tiny` 彩排。超参取简报的默认值，并写成配置。按简报 §9，LoRA 的秩与放置、数据混合与课程属于自由探索；正式取值由 E3 预注册冻结。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §3.3、§3.4、§3.8、§4、§8.5、§9、§12
- `DECISIONS.md` 的 D10、D11、D14、D17、D21、D34、D42、D45、D49、D50、D54
- `src/adapters/base.py`（训练接口）、`src/adapters/training.py`、`src/adapters/checkpoint.py`、`src/executors/`、`src/noising/`、`src/grid/codec.py`、`src/data/train_loaders.py`、`src/e2/`（运行器结构可参照）

## 任务

### 1. 网格词表与目标（`src/e3/tokens.py`、`src/e3/targets.py`）

- **词表**：二值 2 个、深度 64 档、行分隔 1 个，共 67 个新 token，名字过 `check_grid_token_names`。用 `add_grid_tokens` 加入，`token_table()` 记进运行身份。词表写进配置，并核对累计不超过 300。
- **目标序列**：32×32 网格按光栅顺序，每行之后加行分隔，最后接本家族的停止 token。
  - 分割：`mask_to_grid` → 二值 token；
  - 深度：`depth_to_bins` → 64 档 token，无效像素的处理写明。
- **双向函数**：目标 ↔ token 序列 ↔ 网格。测试覆盖往返与长度（32 × 33 + 1）。

### 2. 训练样本（`src/e3/data.py`）

- **任务**：
  - 指代分割：RefCOCO 家族训练集，指令在前；
  - 深度：NYU BTS 训练集，指令在前，指令固定一句；
  - 其余任务按配置可加。
- **修正对**：
  - 从 GT mask 用 `corrupt(mask, severity, rng)` 生成带错的 mask，severity 按 `sample_severity` 的日程；
  - 用 `render_overlay` 渲染叠加图，作为第二张图（原图 + 叠加图）；
  - 指令写明"叠加图是候选结果，请输出修正后的完整网格"，目标是干净网格。
  - 修正对占的比例写进配置，默认 0.5。
  - 深度的修正对：叠加图的渲染方式写明，例如伪彩色；渲染参数与评测共用同一个函数。
- **C3 对照**（`--supervision text`）：同样的图像与指令集合，目标换成稀疏文本：
  - 分割：多边形（E1 的格式）与框；
  - 深度：相对深度陈述，例如两个点谁更近，点按固定规则从 GT 抽样。
  - 用 `chunked_lm_loss` 在全词表上训练。
  - "匹配曝光量"的定义写明：每步的图像数相同、总步数相同，并记下监督 token 数。
- **固定**：样本顺序、修正对的随机数、严重程度，全部由种子与样本 id 决定（无状态），续跑可复现。

### 3. 训练循环（`src/e3/train.py`、`scripts/train_e3b.py`、`configs/e3b_train.json`）

- **顺序**：加载 base → `add_grid_tokens` → `inject_lora`（秩 32，`lora_target_modules()`）→ `trainable_token_rows(网格 token)`。顺序约束沿用适配器的检查。
- **损失**：
  - 网格监督：`training_forward` 加 `restricted_logits`（只在网格 token 与停止 token 上）的交叉熵，只算目标位置；
  - C3：`chunked_lm_loss`。
- **优化**：
  - AdamW，两个参数组：LoRA，以及网格 token 行（weight_decay = 0，D45-3）；
  - warmup 加余弦；bf16；梯度检查点；梯度累积；
  - 全部超参写进配置，带依据。
- **检查点**：每 N 步 `save_trained`，只存 LoRA、token 行、manifest，不存合并后的权重（D34）。
  - 续跑要用的优化器状态另存，默认只留最新一份；9B 的嵌入行优化器状态约 4 GB，写明。
- **续跑**：从最新检查点接着跑，与一次跑完的结果在 CPU 上逐位一致。
- **记账**：
  - 每步的样本数、监督 token 数、前向与反向的 GPU 墙钟、显存峰值；
  - 总计写进 `compute.json`，按 `src/accounting.py`。
- **输出目录**：`experiments/E3/<编号>/<变体>/<模型>/seed<k>/`，内容是 config.yaml、logs、检查点目录与 README。检查点目录不入库，`.gitignore` 加上 `experiments/E3/**/ckpt/`。

### 4. 测试（CPU 小模型，`--tiny`）

- 三家族 × {网格分割、深度、修正对、C3 文本} 各跑几步，loss 有限、梯度只落在 LoRA 与网格行上。冻结行逐位不变，用 `save_trained` 的冻结行核对。
- **过拟合测试**：4 条样本训练到网格逐格正确，用 `generate_restricted` 生成，再用 `parse` 解析。这一条证明训练与推理的链路是通的。
- 保存、加载、合并之后，生成结果与合并前相同（CPU fp32）。
- 续跑逐位一致；样本顺序与修正对可复现。
- 词表预算与名字检查；叠加渲染与评测共用同一个函数（测试钉住）。
- 加载器缺少去重清单时报错。

## 允许改动的文件

- 新建 `src/e3/`、`scripts/train_e3b.py`、`configs/e3b_train.json`、`tests/test_e3_*.py` 与 fixture
- `.gitignore`：只加 `experiments/E3/**/ckpt/`

**不许碰**：`src/adapters/`（需要的接口写进总结）、`src/e1/`、`src/e2/`、`src/e5/`、`src/data/`、`src/executors/`、`src/noising/`、`src/grid/`、`src/synth/`、`configs/` 里已有的文件。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `train_e3b.py --tiny` 三家族全部变体
- 总结里写明：
  - 词表与 token id 布局；
  - 目标序列的格式与长度；
  - 修正对与 C3 的具体做法；
  - 超参默认值与依据；
  - 检查点的体积，0.8B 与 9B；
  - 每步显存与吞吐的估算方法；
  - 评测部分（第二部分）需要的接口与建议。
