[DEV-TASK 0034] E3a 框架：查询 token + 逐层加权混合 + 共享薄解码器 + 开放词汇与标量、向量头；C1、C2 对照；泄漏检查；训练与评测（第一部分）

## 背景

- E3a 是 D 通道的稠密读出（简报 §3.3、§7.3、§8.4；硬约束第 3 条）。负责人定的 10 月 25 日检查点，要看 E3a 第一个尺寸有没有清楚信号（D36-3）。所以它在关键路径上，现在开工。
- E3 的预注册在 G2 时交（D66-4）。本任务只搭框架：口径的默认值放进配置，正式数值以后由预注册冻结。
- 已有可用的接口（不要重写）：
  - 适配器：`inject_query_tokens(n, position_mode)`、`hidden_states(...)`（含查询位置）、带梯度的训练前向（`training_forward`，给 `query_embeds`）、`class_name_spans(inputs, names)`、`vision_features`、`pre_projection_features`、LoRA 的 `save_trained` / `load_trained`（含 `query_embeds`）、`set_gradient_checkpointing`；
  - 数据：`src/data/train_loaders`，要求去重清单；
  - E3b 的训练循环：`src/e3/train.py`，含累积、调度、检查点、续跑、记账，可以借鉴结构或复用工具函数；
  - 上采样：`src/executors/upsample.py`（无学习的 guided、bilinear，numpy）；E2 的可分离矩阵乘双线性：`src/e2/bilinear.py`；
  - 指标：`src/metrics`。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §3.3、§4（硬约束，尤其第 2、3 条）、§7.3、§8.4、§9、§10、§12
- `DECISIONS.md` 的 D10、D19–D21、D36、D45、D49、D50、D55、D56、D59-10、D64–D68
- `src/adapters/base.py`（上面列的接口）、`src/e3/`、`src/e2/`（探针训练、上采样、去重的读法）、`configs/e3b_train.json`

## 任务

### 1. 结构（`src/readout/`）

- **查询 token**：
  - 16×16 = 256 个，可学习的嵌入；
  - 指令在前，查询 token 追加在指令与图像之后；
  - 位置模式按家族：Qwen 用 grid2d，其他家族用 seq1d，经适配器的 `inject_query_tokens`。
- **逐层加权混合**：在全部 LLM 层的查询位置隐藏状态上做可学习的 softmax 加权（层号约定同 D19：含第 0 层，最后一层是归一化之后）。
  - 混合权重是读出的一部分，参与训练；
  - 记录训练前后的权重，便于解读。
- **共享薄解码器**：输入 [16, 16, hidden]，输出 128×128×64 的逐像素嵌入场。
  - **参数量上限 50M**，在构造时断言并报告实际数量；
  - 所有任务共享；
  - 结构在"薄"的约束内自定，写明理由（例如投影到 256 通道，再接几个上采样卷积块：16 → 32 → 64 → 128）。
- **输出头**：
  - **类别（语义分割）**：类别名的表示 = 类别名 token 在同一逐层混合下的平均隐藏状态，用 `class_name_spans` 取 token 区间，再经一个线性投影到 64 维，与像素嵌入做点积，得到开放词汇的 logits。
    - 类别名怎样放进输入（同一条提示里列出，还是单独前向），写明代价并选一个。
  - **标量（深度）、向量（法向）、二值（指代分割）**：各用一个不超过 1K 参数的线性适配，从 64 维到 1、3、1；参数量断言。
- **硬约束第 3 条，必须用测试钉住**：
  - 解码器与各头的输入只有 LLM 隐藏状态和查询 token 嵌入，不能有来自视觉编码器或像素的跳连；
  - 唯一允许的像素参与，是评测时的无学习上采样（guided，以原图为引导图）。

### 2. 训练（`scripts/train_e3a.py`、`configs/e3a_train.json`）

- **任务**：语义分割（ADE20K）、指代分割（RefCOCO 家族，D11-1）、深度（NYU，D11-2）、法向（NYU 法向，D59-6）。任务子集与混合权重放在配置里，必须支持只用指代分割与深度，即 E3b 的同一组数据（简报 §8.5 的同预算对照）。
- **损失**：在 128×128 上算，GT 下采样的方法写明：标签用最近邻，深度、法向用有效像素的面积平均。
  - 语义分割：交叉熵，忽略 0，同 E2。
  - 指代分割：BCE + Dice，同 E2。
  - 深度：SILog，同 E2。
  - 法向：余弦，同 E2。
  - 各任务损失的加权写明。
- **参数组**：
  - LoRA：r 32，作用于 LLM 全部线性层，视觉塔冻结，同 E3b；
  - 查询嵌入；
  - 逐层混合权重；
  - 解码器与各头。

  学习率分组写明，默认值参照 E3b 与 E2。
- **对照**（同一个脚本，按 `--variant` 切换）：
  - `full`：完整方法。
  - `c1`：冻结 LLM，没有 LoRA，只训练查询嵌入、混合权重、解码器与各头。
  - `c2`：解码器直接读视觉编码器的输出，绕过 LLM。输入取 `vision_features`，即投影之后、进入 LLM 之前的图像 token 网格；用无学习的插值缩放到 16×16 后接同一个解码器。没有 LoRA，也没有查询 token。
- **其余同 E3b**：bf16、梯度检查点、步数与批大小可配、warmup cosine、按步续跑、检查点（LoRA + 查询嵌入 + 混合权重 + 解码器分别存）、记账（每步的前向、反传耗时与显存峰值）、`--stop-after-steps`。
  - 数据去重要求同 E3b：`required_eval_sets`。语义分割与法向另加 `ade20k/val`、`nyu_normals/test`。
- **种子**：每个运行一个种子，3 个种子由命令行给出。

### 3. 评测（`scripts/eval_e3a.py`，在 B200 上运行，D55、D59-10）

- **加载**：检查点合并 LoRA 之后，先做合并核对，复用 `src/e3/merge_check` 的 merge 一项；c1、c2 没有 LoRA，跳过核对。
- **稠密指标**：
  - 指代分割：RefCOCO val（E1 的评测总体），cIoU、gIoU、Boundary IoU，按 person 与非 person 分列；
  - 深度：NYU 654，Eigen crop，AbsRel、δ1；
  - 语义分割：ADE20K val 2,000，mIoU；
  - 法向：NYU 法向 test 654，平均角误差、11.25° 以内的比例。
  - 上采样：默认 guided（无学习），另报 bilinear（简报 §8.4）。
  - 都带 bootstrap CI；输出逐样本统计，供配对比较（完整方法对 C1、C2）。
- **泄漏检查**（简报 §8.4，必须做）：
  - batch 内把隐藏状态打乱到别的图像上，指标应当崩溃；
  - 去掉指令后，指代分割应当崩溃。

  两项都作为评测的一部分输出，记在 metrics 里。
- **T 文本评测**：训练后的模型走 `run_e1.py --checkpoint`。E3a 的检查点要能被它加载：只用合并后的 LoRA，不需要查询 token。核对这一点并加测试。
- **输出**：`experiments/E3/<编号>/eval/e3a/<变体>/<模型>/seed<k>/<任务>/`，内容是 predictions（或逐样本统计）、metrics、compute、config、README。支持续跑。

### 4. 吞吐估算

- `train_e3a.py --stop-after-steps 20` 的记账就够用。
- 另外在总结里按 RUN-0026 的 E3b 实测（A800：0.8B 网格 3.6 秒一步、9B 13.8 秒一步）推算 E3a 的每步耗时，写明假设。

### 5. 测试（CPU 小模型，`--tiny`）

- 三家族的 full、c1、c2 训练与评测都跑通。
- 过拟合检查：每个任务几条样本，训练若干步后，在训练样本上指标接近满分。
- 参数量断言：解码器 ≤ 50M，各头 ≤ 1K。
- **没有像素通路**：
  - 构造时检查解码器与各头的输入来源；
  - 另外用梯度检查：解码器输出对图像像素张量（视觉编码器之前）的梯度，必须只经过 LLM 路径。c2 例外，它读的是视觉编码器输出，本来就是对照。
- 泄漏检查在过拟合检查点上确实崩溃。
- 续跑逐字节相同（CPU）；C1 下 LLM 的参数前后逐位不变。

## 允许改动的文件

- `src/readout/`（新）、`scripts/train_e3a.py`、`scripts/eval_e3a.py`、`configs/e3a_train.json`
- 只读复用 `src/e3/`、`src/e2/`、`src/data/` 的函数。要改它们，先在总结里说明理由，并只做不改变已有输出的小改动。
- 对 `src/adapters/`：只有在现有接口确实不够时才加参数，不改默认行为，测试钉住，总结里逐条写明。
- 对应的测试与 fixture

**不许碰**：`configs/e1_prompts.json`、`configs/e1_posthoc.json`、`configs/e2_probes.json`、`configs/e3b_train.json`、`configs/e5_prompts.json`、`src/e1/`、`src/e5/`、`src/synth/`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `train_e3a.py --tiny` 与 `eval_e3a.py --tiny`：三家族 × full、c1、c2
- 总结里写明：
  - 解码器结构与实际参数量，及理由；
  - 类别名进入输入的方式与代价；
  - 损失的加权与 GT 下采样；
  - 学习率分组；
  - 像素通路的检查方法；
  - 泄漏检查的实现；
  - E3a 每步耗时的推算（0.8B、2B、4B、9B）；
  - 所有偏离与留给预注册定的选择。
