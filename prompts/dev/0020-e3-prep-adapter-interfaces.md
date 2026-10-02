[DEV-TASK 0020] E3 开工前的适配器接口（D21-2）：训练前向、唯一的受限 logits、分块交叉熵、新增 token 的保存与核对式恢复、LoRA 的保存/加载/内存合并

## 背景

- G0 独立审查（D21-2）列了一批"E3 开工前修"的项目，现在还没有代码。
- 最要紧的是：训练后的模型重新挂适配器、再加 grid token，会把训练过的行**静默重置为均值**。
- E3 的检查点规则（D34-3、D36-7）：
  - 只存 LoRA、解码器、新 token 行与查询 token，**不存合并后的权重**；
  - 评测时在内存里合并（简报 §4："LoRA 只用于训练，评测前必须合并"）。
- 本任务把这些做成三家族都实现的公共接口，并配齐一致性测试。
- 不训练任何东西，不定 E3 的超参。E3a 的解码器、E3b 的训练循环另派任务。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- 简报 §4（硬约束，尤其第 3、4 条与 LoRA 合并）、§5.2–5.3、§8.4、§8.5、§12（新增 token、合并 LoRA、大词表、查询 token 位置）
- `DECISIONS.md` 的 D10（新增 token 预算）、D15–D19（适配器）、D21（第 2 条是本任务的清单）、D34、D36-7、D37
- `docs/env_report.md` §7：LoRA 的 `language_model_linear_only` 正则、梯度检查点的做法
- 已合入的 `src/adapters/`（base、batching、三个家族）、`tests/test_adapter_*.py`

## 任务

分两部分，**先做完第一部分并测好，再做第二部分**。中途撞上用量上限时，在总结里写清做到哪一步。

### 第一部分：训练前向与唯一的受限 logits

1. **`training_forward(inputs_list, target_ids_list, *, enable_thinking, return_hidden=None)`**（base 定义，三家族实现）
   - **共用**：提示、助手头、思考前缀、填充与位置编码，都与 `generate_batch` 共用，即"教师强制版的 generate"。拼接方式与生成时逐位相同；有测试核对拼接后的 input_ids 与 generate 时的前缀加生成部分相同。
   - **返回**：
     - 目标位置的最后一层隐藏状态（lm_head 之前、final norm 之后）；
     - 每条样本目标 token 的对齐方式：位置 t 的隐藏状态预测第 t+1 个 token。
     - `return_hidden` 给出层号集合与位置（例如查询 token 的区间、图像 token 的区间）时，另外返回这些层在这些位置的隐藏状态，**带梯度**。E3a 的逐层加权混合要用。
   - **梯度**：
     - 支持梯度检查点（`use_reentrant=False`）；
     - 训练模式与评测模式由调用方切换，提供 `train_mode()`、`eval_mode()`；
     - 有测试核对 eval 模式下 dropout 关闭、两次前向逐位相同。
2. **`restricted_logits(hidden, allowed_ids)`**：**唯一**的受限 logits 函数，训练损失与受限解码都调它。
   - 只对 allowed 的行做 lm_head 矩阵乘。
   - 含 Gemma 的 `final_logit_softcapping`。各家族 lm_head 是否与 embedding 绑定，照实处理。
   - logits 取 fp32。
3. **受限解码**：`generate_restricted(inputs_list, allowed_ids, *, max_new_tokens, stop_ids, enable_thinking)`，greedy。
   - 每一步的 logits 都用 `restricted_logits` 算，不经过 HF 的全词表 logits 加处理器。
   - 实现方式自定：手写带 KV cache 的循环，或者从 HF 取最后一层隐藏状态再调本函数。前提是 Qwen 的 M-RoPE、GDN 缓存、Gemma 与 InternVL 的位置处理，都与 `generate_batch` 一致。
   - 有测试核对：同一输入下，现有的 `generate_batch(allowed_token_ids=…)` 与本函数生成的 token 序列在 CPU fp32 上相同。
4. **一致性测试**（D21-2；简报 §12"解码时限制 logits 的实现，在训练和推理中完全一致"）：
   - 受限解码得到一个序列，再用 `training_forward` 加 `restricted_logits` 教师强制同一个序列，每个位置的 argmax 等于解码出的 token。
   - CPU fp32 上逐个相同。
   - GPU bf16 的判据写成一致率（D21-2：Qwen 预填走 fla chunk、解码走 recurrent，不逐位相同），在总结里给出建议的服务器核对方法。
5. **分块交叉熵** `chunked_lm_loss(hidden, target_ids, *, chunk_size)`：全词表 LM 损失，简报 §5.3（词表 248,320）。
   - 结果与不分块的交叉熵在 fp32 上相同（容差写明），含 Gemma 的 softcapping。
   - 受限的网格损失用 `restricted_logits` 加普通交叉熵。

### 第二部分：新增 token 与 LoRA 的保存、恢复、合并

6. **新增 token 的表与核对式恢复**：
   - `add_grid_tokens` 保持现有行为：首次加入时初始化。
   - 新增 `token_table()`，返回：
     - 名字 → id；
     - 每一行在输入 embedding 与 lm_head 里的 sha256，行按 bf16 原始字节；
     - 查询 token 的登记；
     - 家族与模型 revision。
   - 新增 `restore_tokens(table, rows)`：**只核对、不初始化**。
     - 重新登记同样的名字，id 必须与表相同；
     - 把保存的行写回去，写回后哈希与表相同；
     - 不同就报错。
   - **查询占位符**：Qwen 与 InternVL 的查询占位符固定取预留区的**末行**，grid token 的 id 因此与调用顺序无关。D17 说占位符占一个预留行，改了要核对 300 的预算仍然成立。
7. **LoRA**：
   - `lora_target_modules()`：每个家族给出只含 LLM 线性层的目标模块，视觉塔默认冻结。依据是 env_report §7 的 `language_model_linear_only`，Gemma、InternVL 照各自的模块名补齐。有测试核对目标里没有视觉塔或投影层的模块，并且覆盖了全部 LLM 线性层：GDN 的投影也算。
   - 写明并测试顺序约束：LoRA 必须在 `add_grid_tokens`、`inject_query_tokens` **之后**注入。顺序反了就报错。
   - **`save_trained(dir, *, extra)`**：只存四样东西，**不存合并后的权重**（D34），再加一份 manifest（base 的 repo 与 revision、各文件 sha256）：
     - LoRA adapter（peft 格式）；
     - token 表与新增行；
     - 查询 token 的嵌入；
     - `extra` 里调用方给的张量，例如 E3a 的解码器与逐层混合权重。
   - **`load_trained(dir, *, merge=True)`**：
     - 加载 base，按固定 revision；
     - `restore_tokens`；
     - 挂 LoRA，`merge=True` 时在内存里合并（`merge_and_unload`）；
     - 核对 manifest。
   - **合并一致性测试**：CPU 小模型上，给 LoRA 随机的非零权重，合并前后 `generate_batch` 的输出相同，`training_forward` 的 logits 在容差内相同。GPU 的判据写成一致率，与第 4 条相同。
8. **查询 token 的嵌入必须显式传入**：输入里有查询 token 而调用时没给 `query_embeds`，报错。现状是静默用占位符的行（D21-2）。
9. **类别名的 token 区间**（E3a 开放词汇头）：`class_name_spans(inputs, names)` 返回每个类别名在提示里的 token 区间。类别名按 E3a 的用法拼进提示。拼法写明，测试覆盖多 token 的类别名。

## 允许改动的文件

- `src/adapters/`（base、batching、三个家族、factory）：新增上述接口；现有接口的行为不变，只有第 8 条按设计改成报错。
- 新建 `tests/test_adapter_training_*.py`、`tests/test_adapter_tokens_lora*.py`；现有适配器测试只在第 8 条需要时改。
- `requirements.txt` 已由管理员加了 `peft==0.21.1`、`accelerate==1.15.0`，visint 里已装好，不必再装。

**不许碰**：`src/e1/`、`src/e5/`、`src/health/`、`src/data/`、`src/synth/`、`configs/`。E1 与健康检查的冻结协议不能受影响：E1 提示配置的 sha256、健康检查 v3 的输出逐位不变。

## 环境与验收

- conda 环境 `visint`，不装别的包，不下载权重，不联网。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - 现有的冒烟与 `--tiny` 彩排照旧通过：`scripts/smoke_*.py --tiny`、`run_e1.py --tiny`、`health_check.py --tiny`。
- 总结里写明：
  - 每个家族的"第一个目标位置"与对齐方式；
  - `restricted_logits` 怎么处理 softcapping 与绑定权重；
  - 受限解码的实现方式；
  - 查询占位符改到末行之后的 id 布局；
  - 各家族的 LoRA 目标模块表与参数量（rank 32 时）；
  - 检查点目录的文件清单与体积估算（0.8B、9B，rank 32）；
  - 建议的服务器 GPU 核对，作为下一个 RUN 任务的依据：一致率怎么测，n 取多少。
