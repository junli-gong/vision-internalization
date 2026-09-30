[DEV-TASK 0005] InternVL3.5 适配器（照 DEV-0003 的 Qwen35Adapter 结构）

## 背景

G0 要求三个家族都有适配器，并在各自最小尺寸上跑通冒烟测试（简报 §8.1）。本任务按 DEV-0003 合入的 Qwen35Adapter 结构和测试模式，实现 InternVL3.5 的适配器，用 `-HF` 版本（D9）。

**所有 InternVL 事实以 `docs/model_notes.md` §7 与 D13 为准，不要凭记忆。**

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"跨机器差异""评测与报表"
- 简报 §4 第 3–6 条、§8.1、§12
- `DECISIONS.md` 的 D1、D4、D9、D10、D13
- `docs/model_notes.md` §7（InternVL）；对照 §5（Qwen）看差异
- 已合入的 `src/adapters/base.py`、`src/adapters/qwen35.py`、`scripts/smoke_adapter.py`、`tests/test_adapter_qwen35.py`

## 任务

1. **`src/adapters/internvl35.py`：`InternVL35Adapter(ModelAdapter)`**。行为与 Qwen35Adapter 对齐，按下面几点处理 InternVL 的差异：
   - **加载**：`InternVLForConditionalGeneration` 加 `AutoProcessor`，revision 从 `registry.py` 取。
   - **视觉 token**（D13-5）：
     - 构造时给定 `crop_to_patches` 和 `max_patches`，每次调用都显式传给处理器。**处理器默认值（True）与 preprocessor_config（False）相反，严禁依赖默认。**
     - 每个 tile 为 448×448、256 个 token；多 tile 时另有一张缩略图。
     - `ImageTokenInfo` 的 (h, w) 要写明含义：单 tile 时为 16×16。多 tile 时 token 不构成一张整图网格，用 docstring 说明，并在查询/探针相关的方法里拒绝多 tile。
   - **位置**：一维位置，`inject_query_tokens` 的处理同 DEV-0004：`grid2d` 要么报错，要么显式等同 seq1d，二选一，写进偏离。
   - **思考开关**（D13-2）：
     - 模板没有开关。`enable_thinking=True` = 加 card 的 R1 system prompt，原文逐字存为常量，并注明 card 链接与 revision；False = 不加 system prompt。
     - 渲染结果与 model_notes §7.2 表中两行逐字相同，写测试。
     - `GenerationResult.text` 要剥离思考段，格式按 R1 prompt 要求的写法实测确认，不要猜。
   - **生成**：显式 greedy。停止 token 是 `<|im_end|>`，从 tokenizer 查。
   - **`add_grid_tokens`**（D13-1）：复用 257 个预留行（id 151679–151935），规则同 Qwen。
     - 累计超过 257 时只增不缩地扩表；禁止缩表。
     - lm_head 不共享，嵌入行与 lm_head 行分别用各自基础行的精确均值初始化。
   - **`vision_features`**：返回 `multi_modal_projector` 的输出。等价校验：与第 0 层图像位置的隐藏状态逐位相同，写测试并放进冒烟。
   - **坐标**：写一个解析 `<box>[[x1,y1,x2,y2]]</box>`（0–1000，x 在前）并换算为内部绝对像素 xyxy 的函数，用 model_notes §7.5 的实测样例写单元测试。多个框、格式损坏时的行为写明（不猜）。
2. **`scripts/smoke_adapter.py`**：支持 `--model internvl35-2b-hf`，并加入 `--tiny` 路径。检查项同 Qwen，外加 `vision_features` 与第 0 层逐位相同。
3. **tests/**：用本地构造的小型随机 InternVL（缩小的 InternViT 加 Qwen3 LLM，取固定 revision 的真实 config 再缩小）做 CPU 测试。允许下载 tokenizer/processor 小文件，不下载权重。覆盖面照 Qwen 的测试。

## 允许改动的文件

- `src/adapters/internvl35.py`
- `scripts/smoke_adapter.py`：只加 InternVL 分支
- `tests/test_adapter_internvl35.py`（可加 `tests/fixtures/` 下的小文件）
- `src/adapters/registry.py`：只有在键缺失时才改

**不许碰**：`src/adapters/base.py`（接口已定，如有必要的改动写进偏离，由管理员处理）、`src/adapters/qwen35.py`、`src/adapters/gemma4.py`、`src/constants.py`、`requirements.txt`。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿，报告新增测试数与总耗时。
- `smoke_adapter.py --model internvl35-2b-hf --tiny` 退出码 0，把 JSON 摘要贴进总结。
- 逐条列出偏离，至少包括 `grid2d` 的处理、多 tile 下 (h, w) 的定义，以及思考段的剥离格式。
