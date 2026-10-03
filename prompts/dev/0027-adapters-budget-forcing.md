[DEV-TASK 0027] 适配器：预算强制收尾的思考生成 + 投影前视觉特征 + 原生框格式化 + 公开的停止 token 接口；E1 思考子集改口径（D59）

## 背景

- 项目负责人第三批答复（D59，原文在 `docs/lead_answers/2026-10-03.md`）：
  - **预算强制收尾**（D59-1、D59-5）：E1 的思考子集与 E5 的思维链对照都改用它。
    - 思考写到 4,096 token 还没收尾时，替模型补上本家族的思考结束标记，再给 1,024 token 作答；
    - 自然收尾的条数照报；
    - E1 去掉 8192 敏感性检查。
  - **E2 要做投影之前的参考线**（D59-8）：三个适配器加 `pre_projection_features`，每个家族取的位置见 D52 与 DEV-0023 的总结。
  - **E1 验收 v3 的 δ**（D59-3）：多边形与检测两组改为 0.03。
- **DEV-0025 的遗留**（D56-5）：
  - C3 对照要用各家族原生的框格式，需要 `format_boxes`，它是 `parse_boxes` 的逆；
  - 回合结束 token 要有公开接口，现在 E3 读的是私有的 `_stop_ids()`。
- **注意**：RUN-0027（E1 主评测）正在服务器上用 `configs/e1_prompts.json` 的 sha256 `5de1aec7…` 跑，用的是固定 commit 的工作树。本任务改了这个配置后 sha256 会变；主评测、free 与 extra 的行为必须逐位不变，只有思考子集与验收的 δ 变。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/lead_answers/2026-10-03.md`、`DECISIONS.md` 的 D19、D21、D38、D42、D52、D54、D56、D59
- `src/adapters/`（base、batching、training、checkpoint、三个家族）、`src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`
- `docs/model_notes.md`：各家族的思考格式

## 任务

### 1. 预算强制收尾（`src/adapters/`，三家族）

**接口**，签名固定；另一个并行任务（DEV-0028，E5）按这个签名接入：

```python
adapter.generate_batch_budgeted(
    inputs_list, *, thinking_budget: int, answer_budget: int,
    allowed_token_ids: Sequence[int] | None = None,   # 只作用于作答段；None = 全词表
) -> list[GenerationResult]
```

**语义**（逐行）：
1. 开思考，greedy。思考段在自然收尾（本家族的思考结束标记）或写满 `thinking_budget` 个 token 时结束。
2. 写满还没收尾的行，由适配器补上本家族的思考结束标记，以及它后面紧跟的固定分隔。补的 token 不计入模型生成，单独记。
3. 作答段在收尾之后最多 `answer_budget` 个 token，到停止 token 结束。自然收尾的行也是同样的作答上限，不会因为思考短而多拿预算。
4. `GenerationResult` 加字段：
   - `forced_close: bool`、`n_thinking_tokens`、`n_answer_tokens`、`n_inserted_tokens`；
   - `answer_text`（收尾之后的部分）；
   - `raw_text` 保留全文，补上的标记要能辨认。

**实现**：方式自定（两段生成、续用 KV cache 或重新预填），但要满足四点：
- 批量与逐条在 CPU fp32 上结果相同；
- 左填充与位置编码和 `generate_batch` 一致；
- 记账如实：重新预填的 token 也计入；
- 现有的 `generate` / `generate_batch` 行为不变。

**测试**（CPU 小模型，三家族）：
- 构造"自然收尾"与"写满被强制"两种行，混在同一批里；
- 补的标记与本家族的格式一致（对照 chat template）；
- `allowed_token_ids` 只作用于作答段；
- 批量与逐条相同；
- 记账字段正确。

### 2. `pre_projection_features(inputs) -> (Tensor[n, D_v], (h, w))`（三家族，D59-8）

- 只接受单图，行主序，去掉填充，张量在适配器的设备上，可以在 no_grad 下调用。
- 各家族取的位置：
  - **Qwen**：`model.visual` 在 merger 之前的输出，即 patch 网格。原始顺序是 2×2 合并窗口序，要重排成行主序。
  - **Gemma**：视觉塔在 pooler 与 `embed_vision` 之前的输出 (3h, 3w)，去掉填充 patch。
  - **InternVL**：vision_tower 的最后隐藏状态，去掉 CLS，32×32，在 pixel shuffle 与 projector 之前。
- 测试：
  - 顺序用元数据核对，参照 `docs/reviews/2026-10-02-e2-scripts/order_source.py`；
  - 与"投影之后"的关系：对它做该家族的合并或投影，应当得到 `vision_features`（能做到的家族做这条核对，做不到的写明原因）；
  - E2 的 `--reference pre_proj` 端到端跑通（`--tiny`）。

### 3. `format_boxes(boxes, *, width, height) -> str`（三家族）

- 把像素框写成本家族原生的框格式文本，与 E1 定位提示要求的输出格式相同。它是 `parse_boxes` 的逆。
- 测试：`parse_boxes(format_boxes(b)) == b`，在本家族的坐标精度内，多个尺寸与多个框。

### 4. 公开的停止 token 接口

- `adapter.end_of_turn_token_id() -> int` 与 `adapter.stop_token_ids() -> tuple[int, ...]`，替代各处对私有 `_stop_ids()` 的读取。
- `src/e3/` 改用公开接口：只改调用处，行为不变。

### 5. E1 的思考子集改口径（D59-1）

- `configs/e1_prompts.json`：
  - 思考子集改为 `thinking_budget = 4096`、`answer_budget = 1024`，用 `generate_batch_budgeted`；
  - 删掉 `think8192` 子集；
  - `acceptance` 的 δ 按组设置：多边形与检测 0.03，其余组 0.02（D59-3）；
  - 每处改动附 basis，引用 D59。
- `src/e1/`、`scripts/run_e1.py`：
  - `--subset think` 走预算强制收尾；
  - metrics 分"自然收尾"与"强制收尾"报，并列出各自的条数；
  - 删除 `think8192`；
  - 估算（`--estimate`）同步修改。
- **硬性要求**：eval、free、extra 三个子集的提示渲染、生成参数与解析，与 sha256 `5de1aec7…` 的配置逐位相同。加一条测试：用旧配置与新配置分别渲染这三个子集的全部开发样本，输入逐字节相同。
- 新的 sha256 写进总结，由管理员补进 E1 预注册 §5。

## 允许改动的文件

- `src/adapters/`（全部）、`src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`
- `src/e2/`：只为接上 `pre_proj`
- `src/e3/`：只改停止 token 的调用处
- 对应的测试与 fixture

**不许碰**：`src/e5/`、`src/synth/`（DEV-0028 在改）、`src/health/`、`src/data/`、`configs/e5_prompts.json`。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `run_e1.py --tiny --subset think`
  - `run_e2.py --tiny --reference post_proj pre_proj`
  - `train_e3b.py --tiny`
- 总结里写明：
  - 三家族补的思考结束标记的原文与 token；
  - 预算强制收尾的实现方式与记账；
  - 各家族 `pre_projection_features` 的形状；
  - `format_boxes` 的格式；
  - E1 配置的新 sha256，以及与旧配置的差异清单。
