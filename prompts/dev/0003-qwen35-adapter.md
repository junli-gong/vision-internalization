[DEV-TASK 0003] Qwen3.5 适配器 + G0 冒烟脚本

## 背景

简报 §8.1 规定了模型适配器接口，DEV-0001 已在 `src/adapters/base.py` 写好抽象基类。本任务实现 Qwen3.5 家族的适配器，以及在服务器上跑的冒烟脚本（G0 门要看它的结果）。Qwen3.5 的真实 schema 已由 RUN-0001 实测，全部在 `docs/model_notes.md`。**以文档里的实测事实为准，不要凭记忆写 Qwen 的格式。**

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含末尾"作为子 agent"一节）
- `.claude/skills/exp-triad-flow/references/lessons.md` 的"跨机器差异""评测与报表"两节
- 简报：§4（第 3、4、5、6 条）、§5.2 的"新增 token"、§8.1、§12
- `DECISIONS.md`：D1、D4、D9
- `docs/model_notes.md`：§3.1，以及 §5.1–5.7 全部
- `src/adapters/base.py`、`src/constants.py`

## 任务

1. **`requirements.txt`**：加 `transformers==5.17.0`（与服务器一致），装进 conda env `visint`。本任务只允许加这一项；确实还缺别的包，写进偏离，不要自己加。

2. **`src/adapters/registry.py`**：模型键 → (家族, 仓库, revision) 的唯一登记处。登记 D9 里的 8 个模型：
   - Qwen3.5 0.8B/2B/4B/9B
   - gemma-4 E2B/E4B-it
   - InternVL3.5 2B/4B-HF

   revision 抄 `docs/model_notes.md` §2。加载时一律带 revision。

3. **`src/adapters/qwen35.py`：`Qwen35Adapter(ModelAdapter)`**

   **加载与设备**
   - 用 `AutoModelForImageTextToText` 和 `AutoProcessor` 加载，`attn_implementation="sdpa"`，eval 模式。
   - 设备在构造时给定，dtype 在 `load` 时给定。进入任何算子之前用入口断言检查输入与模型的 device/dtype，不一致时抛自有异常并报出两个值（lessons：跨机器差异）。

   **固定分辨率**
   - 构造时给定 `image_size=(H, W)`，H、W 都必须是 32 的倍数。
   - `build_inputs` 先用执行器把图缩放到 (H, W)，再调处理器，并传 `min_pixels=max_pixels=H*W`，保证处理器不再缩放。
   - 视觉网格 = (H/32, W/32)，写进 `ImageTokenInfo`。所有对比都用同一个设置（简报 §12）。

   **`build_inputs(images, instruction, order)`**
   - 支持多张图。user 消息的内容顺序：
     - `instruction_first` → [指令, 图1, 图2, …]
     - `image_first` → [图1, …, 指令]
   - 只渲染到 user 轮结束，即 `<|im_end|>\n` 为止，**不含** assistant 头。assistant 头由 `generate` 按思考开关追加。
   - 如果已调用 `inject_query_tokens`，查询 token 放在 user 内容的最后一项之后、`<|im_end|>` 之前。
   - 精确填好每张图的 token 区间与网格、指令区间、查询区间。区间用 token 下标，测试逐项钉住。

   **`generate(inputs, *, enable_thinking, allowed_token_ids=None, max_new_tokens, **gen_kwargs)`**
   - 按 `enable_thinking` 追加 assistant 头的 token。要写一条测试：追加后的完整 ids 与 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=X)` 的结果逐个相同，两种开关都测。
   - 开关直接作为 `apply_chat_template` 的关键字参数传（D9-4），不用 `chat_template_kwargs`。
   - 默认 greedy。停止 token 取 tokenizer 里的 `<|im_end|>` 和 `<|endoftext|>`，从 tokenizer 查 id，不写死数字。
   - `allowed_token_ids` 非空时，用 logits 处理器把其他 id 置为 -inf。
   - 返回 `GenerationResult`：原始 token ids、解码文本、生成 token 数、prompt token 数、视觉 token 数、墙钟时间。
   - 本任务只做单条样本；批处理和 vLLM 另开任务。

   **`hidden_states(inputs, layers, *, query_embeds=None)`**
   - 前向时 `output_hidden_states=True`，并传 `logits_to_keep=1`，避免算整序列 logits。
   - 层号就是 HF `hidden_states` 元组的下标：0 是嵌入输出，i 是第 i 层之后。有效范围 [0, num_layers()]，越界报错。
   - 返回图像 token（多图按顺序拼接）和查询 token 位置上的向量。
   - `query_embeds`（形状 `[n_query, hidden]`）非空时，用 `inputs_embeds` 把查询位置的嵌入替换成它（E3a 的可学习查询 token 从这里进）。
   - 这个关键字参数需要同步加到 `base.py` 的抽象签名里，默认 `None`。**这是对接口的扩展，列进偏离。**

   **`inject_query_tokens(n, position_mode)`**：登记 n 个查询位。占位 token 用 `add_tokens` 新增的一个专用 token，嵌入在前向时会被替换，所以不计入 E3b 的 300 个 token 预算，但要在总结里写明。两种位置模式：
   - `grid2d`：n 必须是完全平方数，网格 q×q。查询块像一张图那样编 M-RoPE：设块前最后一个 token 的位置是 p，取 t0 = p + 1，则查询 (r, c) 的 (T, H, W) = (t0, t0 + r, t0 + c)；块之后的 token 从 t0 + q 接着编号。这与 model_notes §5.4 实测的图像规则一致。
   - `seq1d`：查询 token 按文本方式顺序编号，T = H = W。
   - 两种模式都要自己构造完整的 `position_ids`，形状 (4, B, L)：第 0 行是 0..L-1 的文本位置，第 1–3 行是 M-RoPE 位置，与 model_notes §5.4 记录的 generate 内部做法一致。然后显式传给模型。

   **`add_grid_tokens(names)`**
   - 先调 `check_grid_token_names`，再 `add_tokens(..., special_tokens=True)`。
   - 断言新 id 全部小于嵌入行数：复用预留行，**禁止调用 `resize_token_embeddings`**（D9-3）。
   - 断言每个名字分词后是单个 token。
   - 新行初始化为基础词表行的均值（D9-3）。lm_head 不共享时（4B/9B），对应 lm_head 行也用基础行均值；共享时（0.8B/2B）不要重复写。
   - 返回 名字 → id。

   **`num_layers()` / `hidden_size()`**：取 `text_config`。

   **接口补充**（来自 DEV-0001 的独立审查，D10。同步改 `base.py` 的抽象定义与 docstring，全部列进偏离）：
   - **层号约定**：写进 `base.py` 的 docstring。第 0 层 = 视觉特征并入序列之后的嵌入输出；第 i 层 = 第 i 个 decoder 层的输出；有效范围 [0, num_layers()]。各家族必须一致，否则 E2 的逐层曲线会错位。
   - **新增抽象方法 `vision_features(inputs)`**：返回进入 LLM 之前、经过投影/合并之后的视觉 token，形状 [n_image_tokens, hidden]，多图按顺序拼接。E2 的参考线和 E3a 的 C2 对照要用它。
     - Qwen 实现要满足一条免费的等价校验：与 `hidden_states(inputs, [0])` 在图像位置上逐位相同。测试和冒烟都要查。
   - **token 预算跨调用累计**：`check_grid_token_names(names, n_existing=0)` 按累计总数检查（≤ `MAX_NEW_GRID_TOKENS`）。适配器自己记已加的总数，分两次各加 200 个必须报错。
   - **`GenerationResult` 加 `raw_text` 字段**：完整解码结果，含思考段与特殊 token。`text` 改为剥离思考段后的最终回答：Qwen 取最后一个 `</think>` 之后的内容。开了思考却没生成 `</think>` 时，`text` 为空串，并在结果里标记思考未闭合。
     - 这样解析器只看最终答案，思考段里的草稿网格不会被误判为多个块。

   **原则**：模型相关的事实（特殊 token 字符串、预留行是否存在）运行时从 tokenizer、config 读取并断言，不写死 id 和层号（简报 §4 第 6 条）。

4. **`scripts/smoke_adapter.py`**：给服务器跑的冒烟脚本，输出一个 JSON，任何一项失败就以非零码退出。
   - 参数：`--model <注册键>`、`--device`、`--dtype`、`--image <路径>`、`--out <json 路径>`。
   - 另加 `--tiny`：用本地构造的小型随机 Qwen3.5 在 CPU 上彩排同一流程。
   - 检查项：
     1. 两种顺序都能 build 与 generate，思考开关开和关各一次，`max_new_tokens` 取小值。
     2. 经适配器 API 复现 RUN-0001 §5.7 的仪器：图在前时，两条等长指令下，各层图像 token 隐藏状态逐位相同；指令在前时，第 0 层相同、之后各层不同。
     3. `add_grid_tokens(["<g0>", "<g1>"])` 后，用 `allowed_token_ids` 限定生成，产出的 id 全在允许集合内。
     4. `inject_query_tokens(16, …)` 两种模式都能前向，形状正确。查询块之前各位置的隐藏状态在两种模式下逐位相同（因果性）。grid2d 的位置满足上面的规则。
     5. 不带查询 token 时，传入自己构造的 `position_ids` 与让模型自己算，隐藏状态逐位相同。这一项验证我们的位置构造没有偏。
     6. `vision_features` 与第 0 层图像位置的隐藏状态逐位相同。
     7. 思考开启时，`text` 不含思考段，`raw_text` 含思考段。

5. **tests/**：CPU 测试。
   - 用本地构造的小型随机 Qwen3.5：取固定 revision 的真实 config，缩到 4 层（3 层 linear_attention 加 1 层 full_attention）、hidden 64，视觉塔缩到 1 层。
   - 允许下载 tokenizer 和 processor 的小文件（不含权重），固定 revision。
   - 覆盖：
     - 区间精确；两种顺序；assistant 头与模板逐个相同。
     - 图在前逐位相同 / 指令在前不同。
     - grid token：不缩行、单个 token、均值初始化，tied 与 untied 两种 config 都测。
     - allowed ids；查询位置两种模式的数值，以及因果前缀逐位相同；自构 position_ids 与模型自算逐位相同。
     - `enable_thinking`、`max_new_tokens` 漏传时报 TypeError。
     - device 入口断言：用 meta 设备做可观测代理，参照 lessons。
   - fixture 的图 H≠W，两条指令内容不同但 token 长度相同。

## 允许改动的文件

- `src/adapters/qwen35.py`、`src/adapters/registry.py`
- `src/adapters/base.py`：只改这几处：`query_embeds` 关键字参数；层号约定的 docstring；`vision_features` 抽象方法；`check_grid_token_names` 的 `n_existing` 参数；`GenerationResult` 的 `raw_text` 字段。
- `scripts/smoke_adapter.py`、`tests/test_adapter_qwen35.py`（可以另加 `tests/fixtures/` 下的小文件）
- `requirements.txt`

**不许碰 `src/constants.py`**：同时在跑的 DEV-0002 会改它。

## 约束

- 模型相关逻辑只放在 `src/adapters/`；不写死层号。
- 不调 `resize_token_embeddings`，不用 `chat_template_kwargs`。
- 不引入任何学习型视觉模块；缩放只用白名单里的执行器（PIL/numpy）。
- 不下载任何模型权重。测试不能静默跳过：缺文件就失败并说明原因。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿。原有 113 条保持通过，报告新增测试数与总耗时。
- `conda run -n visint python scripts/smoke_adapter.py --tiny --out /tmp/smoke_tiny.json` 退出码 0，把 JSON 摘要贴进总结。
- 逐条列出偏离，至少包括 `query_embeds` 的接口扩展，以及查询占位 token 的选择。
