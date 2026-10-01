[DEV-TASK 0011] E1 的适配器准备：批量生成、图像口径（Qwen 像素预算、InternVL 原生动态 tile）、停止原因、统一构造入口

## 背景

- G0 已通过（D28）。E1 要在 8 个模型上做大规模零样本生成。
- 独立审查（D21）指出两件 E1 开跑前必须修的事：
  - 适配器只有 batch=1 的生成，单是定位就要 70–115 GPU 小时；
  - 图像几何口径未定。
- D28-3 冻结了 E1 与 E3 的图像口径，D28-5 定了批处理的验收标准。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"跨机器差异"一节
- 简报 §4 第 5、6、11 条，§6.3，§8.1，§8.2，§12
- `DECISIONS.md`：D13（视觉 token 与思考开关）、D15、D17、D19、D21（审查的发现）、D22（解析层）、D28（尤其第 3、5、10 条）
- `docs/model_notes.md`：
  - Qwen 处理器的像素上下界与 smart resize，约第 267–285 行；
  - InternVL 的 tile 规则，约第 959–990 行。
- 已合入的 `src/adapters/`、`src/parsing/core.py`、`src/accounting.py`、`scripts/smoke_adapter.py`、`scripts/health_check.py`（看 `make_adapter`）

## 任务

### 1. 图像口径（D28-3，冻结）

**Qwen：新增"像素预算"模式**
- 构造参数，例如 `image_policy=PixelBudget(n_tokens)`，与现有的固定 `image_size=(H, W)` 二选一，现有模式保持不变。
- 像素预算模式下，按处理器的 smart resize 规则保持长宽比：`min_pixels = max_pixels = n_tokens·32²`。
- E1 默认取 `n_tokens=300`：640×480 正好是 20×15 = 300 个 token，不缩放。
- 约 1K token 的子集实验取 `n_tokens=1024`。
- 每张图有自己的 (h, w) 网格，写进 `ImageTokenInfo.grid_hw`。
- `build_inputs` 的结果必须与官方 processor 在同样 min/max_pixels 下逐项相等（pixel_values、image_grid_thw、input_ids）。写测试，并接进冒烟检查 1。
- `hidden_states`、`vision_features`、查询 token 在像素预算模式下要按每张图自己的网格工作：至少单张图要能用；多张图且网格不同的情形，能用或者明确报错，二选一，写明。
- 去掉所有假设"全局只有一个 `llm_grid_hw`"的地方，或者让它只在固定尺寸模式下可用，在像素预算模式下报错。

**InternVL：原生动态 tile 选项**
- 默认仍是单 tile（`crop_to_patches=False, max_patches=1`）。
- 新增原生口径 `crop_to_patches=True, max_patches=12`：这是 preprocessor_config 的 12，640×480 会切成 13 个 tile、3328 个 token。只用于子集对照，生成可用；探针类方法照旧拒绝多 tile。

**Gemma**：沿用 `max_soft_tokens=280`。

**三家族共同**：每次生成都在结果里记下模型实际看到的像素数与视觉 token 数（D28-3 要求在报告里写明）。
- 建议在 `GenerationResult` 里加字段，或者在批量接口的返回元数据里给出；
- 计算量记账同步更新。

### 2. 批量生成（D28-5）

- 新增 `generate_batch(inputs_list, *, enable_thinking, max_new_tokens, allowed_token_ids=None, **gen_kwargs) -> list[GenerationResult]`。
  - 在 base 里定义接口，三家族各自实现；
  - 必填参数的规则与 `generate` 相同，也要进 `_REQUIRED_KWONLY`。
- **左填充**，注意力 mask 正确。每一行的思考开关、助手头、system 轮与单条生成完全相同。
- **位置编码**：
  - **Qwen 的 M-RoPE（重点）**：左填充之后，每一行非填充 token 的三维位置 id，必须与该样本单独生成时逐个相等。写专门的测试：长度不同、图像网格不同（横图与竖图）、带与不带思考头。
  - Gemma 与 InternVL 的一维位置同样要核对。
- **按家族的要求**：
  - Gemma：按样本嵌套传图；
  - InternVL：每个样本的 tile 数可以不同。
- **逐行拆分**：
  - 每一行有自己的 `token_ids`，从第一个生成 token 到第一个停止 token（含），之后的填充不计；
  - 每一行有自己的 `text`、`raw_text`、`thinking_unclosed`；
  - 计算量：每行记自己的生成 token 数与视觉 token 数；墙钟记整批的值，并按行均摊，规则写明。
- **一致性**：
  - 在 CPU 小模型上（fp32），对每个家族核对 batch 与逐条 greedy 的生成 token 逐个相同，至少 8 个混合长度、混合图像尺寸的样本，生成 ≥ 16 步。不相同就查清原因。
  - 不能要求逐位相同的情形（例如某个内核在填充下改变归约顺序），写进偏离，并给出实测的不一致程度。
- **预热**：首次调用含 JIT 与内核编译。提供 `warmup()` 或在批量接口里记录"是否首批"，计时口径写明（D21）。

### 3. 停止原因（D21、D22-3）

- `GenerationResult` 加 `stopped: bool`：最后一个 token 是停止 token 为 True，被 `max_new_tokens` 截断为 False。单条与批量都要填。
- 在 `src/parsing` 的 `ParseFailure` 加 `truncated`。`parse_generation` 的判定顺序：
  1. `thinking_unclosed`；
  2. `truncated`（`stopped` 为 False）；
  3. `empty`；
  4. 交给解析函数。
- 截断一律判失败，不尝试从截断的输出里解析。这是 E1 的协议，写进 docstring。
- 旧的 `GenerationResult` 构造方式要保持兼容，字段给默认值；测试替身相应更新。

### 4. 小修

- **Qwen 的 `token_id()`**：要能查词表里的任意 token，例如 `<think>`，不能只认 attach 时缓存的那几个（RUN-0005 发现）。
- **统一构造入口**：新建 `src/adapters/factory.py`，提供 `make_adapter(key, *, device, dtype, image_policy=None, **family_kw)`，按注册表分派到三个家族。
  - `scripts/smoke_adapter.py` 与 `scripts/health_check.py` 改为调用它，删掉各自那份重复代码，行为不变；
  - 两个脚本的现有测试必须照样通过。

### 5. 批处理核对脚本

新建 `scripts/validate_batching.py`，给服务器用：
- 输入：模型键、图像口径、一个 JSONL（每行：图像路径、指令、`enable_thinking`、`max_new_tokens`）、batch 大小。
- 做法：对同一批输入，逐条生成一遍、批量生成一遍。
- 输出：
  - raw_text 逐字相同的比例，以及第一个不同的位置；
  - 每家族、每个 batch 大小的吞吐（样本/秒、token/秒）。
- 解析后答案的一致率与指标的 bootstrap CI，由 E1 评测框架在开发子集上算，不在本脚本里做。
- 写一个 `--tiny` 彩排。

## 允许改动的文件

- `src/adapters/`（含新建 `factory.py`）
- `src/parsing/core.py`：只加 `truncated` 与 `parse_generation` 的判定
- `src/accounting.py`：只加像素数等新字段
- `scripts/smoke_adapter.py`、`scripts/health_check.py`：只改为调用 `make_adapter`，冒烟检查 1 接入像素预算
- 新建 `scripts/validate_batching.py`
- `tests/` 下对应的测试

**不许碰**：`src/synth/`、`src/health/`（协议不变）、`src/metrics/`、`requirements.txt`。

## 环境与验收

- 用 conda 环境 `visint`，不装包，不下载权重（tokenizer 与 processor 的小文件可以）。
- 验收：
  - `conda run -n visint pytest tests/ -q` 全绿；
  - `bash scripts/pytest_server_parity.sh` 全绿（改了适配器，D18）；
  - 三家族 `--tiny` 冒烟，包括像素预算模式；
  - `validate_batching.py --tiny` 三家族各一条。
- 总结里逐条列出偏离，至少包括：
  - 像素预算模式下多图、不同网格的处理；
  - 批量与逐条在 CPU 上是否逐个 token 相同，不同的原因与程度；
  - 墙钟均摊规则；
  - `truncated` 对现有解析结果的影响。
