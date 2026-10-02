[DEV-TASK 0021] 训练接口的 GPU 核对脚本（受限 logits 一致率、LoRA 合并一致率，兼作评测前的合并核对仪器）+ 新增 token 行的训练开关

## 背景

- DEV-0020（D42）给适配器加了两样东西：
  - 训练前向、唯一的 `restricted_logits` 与受限解码；
  - LoRA 的保存、加载与内存合并。
- CPU fp32 上逐个相同已经由测试钉住。GPU bf16 上不逐位相同：Qwen 预填走 fla chunk、解码走 recurrent。判据要写成一致率（D21-2），需要在服务器上实测。
- 简报 §12："合并 LoRA。评测前必须合并，并在开发子集上验证合并前后的输出一致。"以后每个训练出来的模型评测前都要做这一步。所以这次写的脚本不是一次性探针，而是常备仪器。
- DEV-0020 的遗留：peft 会把 base 的参数全部设为不求梯度，网格训练却要训练新增 token 的那几行。需要一个适配器层面的开关。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D21（第 2 条）、D34、D36-1（数值噪声底的口径）、D38、D42
- `src/adapters/training.py`（`restricted_consistency`）、`src/adapters/checkpoint.py`（`merge_consistency`、`save_trained`、`load_trained`）、DEV-0020 的测试
- `src/e1/`：只读，用它的开发子集与 grid16 提示

## 任务

### 1. `scripts/check_training_interfaces.py`

```bash
python scripts/check_training_interfaces.py --model <键> --checks restricted merge \
  --n 200 --data-root $PROJ/data --out <json> [--checkpoint <检查点目录>] [--tiny]
```

- **输入**：
  - E1 的开发图（RefCOCO 开发子集）加 E1 grid16 的提示原文。经 `src/e1` 的现有函数取得，只读，不改 E1 的配置与代码。
  - 两种思考开关各跑一遍；`max_new_tokens` 128；批配置照 E1 冻结的那套（`batch_protocol`）。
- **allowed 集合**：E1 数字网格格式实际用到的 token，由 tokenizer 查出来写进输出：
  - 数字 0 与 1、换行，以及本家族的停止 token；
  - 不用刚加的 grid token：它们是均值初始化，logits 全部相同，只会得到平局（DEV-0020 的提醒）。
- **`restricted`**：调 `training.restricted_consistency`，报告：
  - 受限解码与 `generate_batch(allowed_token_ids)` 的 token 一致率；
  - 第一个分歧位置的分布；
  - 教师强制（`training_forward` + `restricted_logits`）的 argmax 与解码 token 的一致率；
  - 每个不一致位置上前两名的 logit 差：中位数、最大值，以及差值 < 0.125 的比例。
- **`merge`**：
  - 不给 `--checkpoint` 时：注入 LoRA（rank 32，用 `lora_target_modules()`），按固定种子把 A、B 设成小尺度的随机值，跑 `checkpoint.merge_consistency`。尺度要让合并前后的 logits 与 base 明显不同，写明数值。
  - 给 `--checkpoint` 时：用 `load_trained(merge=False)` 与 `load_trained(merge=True)` 在同一批开发提示上比较。这是以后评测前的标准核对。
  - 报告：exact_rows、token_agreement、max_abs_logit、不一致位置的 logit 差。
- **判据**：写进输出的 `criteria` 与 `pass`，作为"提议值"，由管理员在 RUN 实测后冻结：
  - 一致率 ≥ 0.995；
  - 不一致位置的前两名 logit 差全部 < 0.125；
  - `--tiny`（CPU fp32）上必须逐个相同。
- **输出 JSON**：判据、各项读数、allowed 集合的 token 表、批配置、模型 revision、代码 commit。
- **`--tiny`**：小型随机模型 + CPU + `tests/fixtures/e1` 的小数据，三个家族都要能跑。

### 2. 新增 token 行的训练开关

- 新增 `adapter.trainable_token_rows(names_or_ids)`：
  - 在 LoRA 注入之后调用；
  - 只让输入 embedding（以及不共享时的 lm_head）里这些行求梯度，其余行保持冻结。
  - 实现方式自定，例如梯度掩码钩子或 peft 的 `trainable_token_indices`，要在总结里说明为什么选它。
  - 共享权重的家族（Qwen 小模型、Gemma），embedding 与 lm_head 是同一个张量，只能处理一次。
- `save_trained` 保存训练后的行，`restore_tokens` 能逐位恢复。
- 测试：
  - 一步优化器更新后，只有这些行变了，其余行逐位不变，视觉塔与 LLM base 权重也不变；
  - 保存再加载，行的哈希相同；
  - 合并 LoRA 后，这些行仍是训练后的值。

## 允许改动的文件

- 新建 `scripts/check_training_interfaces.py`、`tests/test_check_training_interfaces.py`、`tests/test_adapter_trainable_rows.py`
- `src/adapters/training.py`、`src/adapters/checkpoint.py`：只为第 2 条与读数补充
- 对应的 fixture

**不许碰**：`src/e1/`、`src/e5/`、`src/health/`、`src/data/`、`src/synth/`、`configs/`、三个家族文件。三个家族文件只有在第 2 条确实需要家族钩子时才改，并在总结里说明。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `check_training_interfaces.py --tiny` 三个家族各跑一遍，两项检查都 `pass`
- 总结里写明：
  - 输出 JSON 的字段；
  - 随机 LoRA 的尺度与依据；
  - 预计在 B200 上的用时，0.8B 与 9B 各一次；
  - 建议的服务器运行命令。
