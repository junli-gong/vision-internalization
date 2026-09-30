[DEV-TASK 0004] Gemma 4 适配器（照 DEV-0003 的 Qwen35Adapter 结构）

## 背景

G0 要求三个家族都有适配器，并在各自最小尺寸上跑通冒烟测试（简报 §8.1）。Qwen3.5 的适配器、接口补充和冒烟脚本已由 DEV-0003 合入。本任务按同样的结构和测试模式实现 Gemma 4 的适配器。

**所有 Gemma 事实以 `docs/model_notes.md` §6 与 D13 为准，不要凭记忆。**

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"跨机器差异""评测与报表"
- 简报 §4 第 3–6 条、§8.1、§12
- `DECISIONS.md` 的 D1、D4、D9、D10、D13
- `docs/model_notes.md` §6（Gemma）；对照 §5（Qwen）看差异
- 已合入的 `src/adapters/base.py`、`src/adapters/qwen35.py`、`scripts/smoke_adapter.py`、`tests/test_adapter_qwen35.py`

## 任务

1. **`src/adapters/gemma4.py`：`Gemma4Adapter(ModelAdapter)`**。行为与 Qwen35Adapter 对齐，按下面几点处理 Gemma 的差异：
   - **加载**：模型键和 revision 从 `registry.py` 取。
   - **视觉 token**：构造时给定 `max_soft_tokens`，只接受 70/140/280/560/1120，其他值直接报错。每次调用都显式传，不依赖默认值。网格 (h, w) 由 `image_position_ids` 推出；注意它的第 0 维是列（x），第 1 维是行（y）。
   - **批处理**：多图输入时，processor 要求图像按样本嵌套传入。
   - **位置**：一维位置。`inject_query_tokens` 两种模式在这里都退化为一维顺序位置。按简报 §12，二维网格位置只适用于 M-RoPE 模型：`grid2d` 要么报错，要么显式记录"对一维模型等同 seq1d"，二选一，写进偏离。
   - **思考开关**：`enable_thinking` 直接传给 `apply_chat_template`。`GenerationResult.text` 要剥离思考段；Gemma 思考段的格式在 model_notes §6.2 查，不要猜。
   - **生成**：显式 greedy（generation_config 默认是采样）。停止 token 是 `<turn|>` 和 `<eos>`，从 tokenizer 查 id。
   - **`add_grid_tokens`**（D13-1）：复用 `<unusedN>`：
     - 按 id 从小到大取；
     - 先 `add_tokens` 注册，断言注册后仍是原 id、len(tokenizer) 不变、不 resize；
     - 初始化两处：主嵌入行（与 lm_head 共享），以及 PLE 的 `embed_tokens_per_layer` 对应行，都用基础词表（不含 `<unusedN>` 和特殊 token）的精确均值；
     - 累计预算照 base.py 的检查。
   - **`vision_features`**：返回进入 LLM 之前的视觉 token，即 `embed_vision` 之后的特征。测试并在冒烟里检查它与第 0 层图像位置的关系。Gemma 在嵌入层有 `embed_scale`，而 PLE 不作用在图像位置上，所以两者可能不逐位相同。按源码写明两者的精确关系，并用测试钉住。
   - **层号约定**：遵守 base.py；`output_hidden_states` 是 36 个张量，等于层数 + 1。
   - **坐标**：写一个把 `box_2d`（0–1000，**y 在前**）换算为内部绝对像素 xyxy 的函数，用 model_notes §6.5 的实测样例写单元测试。
2. **`scripts/smoke_adapter.py`**：支持 `--model gemma4-e2b-it`，并加入对应的 `--tiny` 路径。检查项与 Qwen 相同。Gemma 特有的两项：
   - `vision_features` 与第 0 层的关系；
   - `grid2d` 按上面的决定处理。
3. **tests/**：用本地构造的小型随机 Gemma 4 做 CPU 测试。取固定 revision 的真实 config，缩到约 4 层（含 sliding 与 full，至少一层 KV 共享），缩小 PLE 维度与视觉塔。允许下载 tokenizer/processor 小文件，不下载权重。覆盖面照 `tests/test_adapter_qwen35.py`。

## 允许改动的文件

- `src/adapters/gemma4.py`
- `scripts/smoke_adapter.py`：只加 Gemma 分支
- `tests/test_adapter_gemma4.py`（可加 `tests/fixtures/` 下的小文件）
- `src/adapters/registry.py`：只有在键缺失时才改

**不许碰**：`src/adapters/base.py`（接口已定，如有必要的改动写进偏离，由管理员处理）、`src/adapters/qwen35.py`、`src/constants.py`、`requirements.txt`。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿，报告新增测试数与总耗时。
- `smoke_adapter.py --model gemma4-e2b-it --tiny` 退出码 0，把 JSON 摘要贴进总结。
- 逐条列出偏离，至少包括 `grid2d` 的处理，以及 `vision_features` 与第 0 层的关系。
