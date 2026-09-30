[DEV-TASK 0006] 输出解析器（Phase 0 组件）：通用解析层 + 三家族原生框格式 + 解析失败率记账

## 背景

简报 §8.1 把"解析器：对各模型的输出格式鲁棒，并记录解析失败率"列为 Phase 0 组件。E1 的每个 T-文本任务都要靠它把模型输出变成可评分的值。

现状：
- InternVL 已有完整的框解析：`internvl35.parse_boxes`、`parse_one_box` 和 `BoxParseFailure`（D19）。
- Gemma 只有坐标换算 `gemma4.box2d_to_xyxy`，不从文本里解析。
- Qwen 没有任何框解析。
- 多选题有 `metrics.misc.normalize_choice`，但它按设计不从自由文本里抽字母（docstring 写明"那是解析器的事"）。
- 网格有 `grid.codec.parse_grid_text`。

本任务补齐解析层。所有模型相关的格式知识放在各家族的适配器模块里（简报 §4 第 6 条）；与模型无关的部分放在新的 `src/parsing/`。

先读：
- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"评测与报表""跨机器差异"
- 简报 §7.1、§8.1、§8.2、§12
- `DECISIONS.md` 的 D10（解析器只看剥离思考段后的 `text`）、D12（normalize_choice 的边界）、D13-4（三家族原生框格式）、D19（InternVL 框解析"不猜"）
- `docs/model_notes.md` 里的真实输出：
  - Qwen：§5 零样本实测，约第 321–326 行；
  - Gemma：约第 758–767 行；
  - InternVL：约第 1057–1066 行。
- 已合入的代码：`src/adapters/base.py`、`src/adapters/internvl35.py`（parse_boxes 一节）、`src/adapters/gemma4.py`（box2d_to_xyxy）、`src/executors/geometry.py`、`src/metrics/misc.py`、`src/grid/codec.py`

## 总原则："不猜"，但对真实出现的包装要鲁棒

- 每个解析函数返回 `(值, 失败原因)`。失败原因取自一个有限枚举，用来统计解析失败率。
- 规则必须是确定的、写在 docstring 里的：
  - 对真实输出里常见的包装（```json 代码块、前后的说明文字、停止 token 已被剥离后的残留空白）要能解析；
  - 对含糊的情况（多个候选、坐标越界、框颠倒）判失败，不挑一个猜。
- 任何一个框（或点）坏了，整条判失败，不返回其余的。与 D19 一致。

## 任务

### 1. `src/parsing/`（新包，与模型无关）

- **`ParseFailure`**：统一的失败原因枚举（str 枚举）。
  - InternVL 现有 `BoxParseFailure` 的取值（`no_box`、`unbalanced_tags`、`malformed`、`out_of_range`、`inverted`、`multiple_boxes`）原样保留。`internvl35.BoxParseFailure` 保持可导入（别名即可），现有测试不改或少改。
  - 至少新增：`empty`、`thinking_unclosed`、`no_number`、`multiple_numbers`、`no_choice`、`multiple_choices`、`no_polygon`、`too_few_points`、`ambiguous`（例如多个 JSON 块）。
- **`ParseResult`**：通用的 `(value, failure)`。
- **`ParseTally`**：累计解析结果。
  - 按失败原因计数，给出 `n`、`n_ok`、`failure_rate`。
  - `to_dict()`：可直接写进 metrics.json，键顺序稳定。
  - 可合并。
  - 也能接收 `GridParseFailure` 这类其他 str 枚举。
- **`parse_generation(result, parse_fn, **kw)`**：
  - `result.thinking_unclosed` → `thinking_unclosed`；
  - `text.strip() == ""` → `empty`；
  - 否则调用 `parse_fn(result.text, **kw)`。
- **`extract_json(text)`**：
  - 有 ``` 代码块时，只看代码块。恰好一个代码块能被 `json.loads` 解析才算成功；多于一个 → `ambiguous`。
  - 没有代码块时，在全文里按括号配对找顶层的 `[...]` 或 `{...}`。恰好一个 → 用它；多于一个 → `ambiguous`；没有 → 由调用方决定失败原因。
  - 严格 JSON，不用 `ast.literal_eval` 兜底。
- **`parse_count(text)`**：
  - 认阿拉伯数字，以及英文数词 zero–twenty（不分大小写，整词匹配）。
  - 恰好一个不同的数值 → 该整数；没有 → `no_number`；两个以上不同的数值 → `multiple_numbers`。
  - 小数、负数 → `malformed`。
- **`parse_choice(text, letters)`**：多选题。
  1. 先用 `normalize_choice`；
  2. 认不出时，再认明确陈述的形式，例如 "The answer is B"、"Answer: (C)"、"option D"；
  3. 最后看全文里出现的括号字母 "(X)"：只在 `letters` 里出现恰好一个不同字母时才用它。

  回显的选项列表（例如 "(A) left (B) right"）必然给出多个字母 → `multiple_choices`。规则与例子写进 docstring 和测试。
- **`parse_named_choice(text, names)`**：用于 "left/right"、"television/dining table" 这类两选一。
  1. 规范化后（小写、去掉首尾标点、引号和冠词）整段等于某个名字 → 该名字；
  2. 否则看各名字作为整词短语出现的情况，长名字优先匹配，避免 "table" 命中 "dining table"。恰好一个出现 → 该名字；
  3. 一个都没有 → `no_choice`；多个 → `multiple_choices`。
- **`parse_polygons(text, *, width, height, axis_order="xy")`**：没有哪个家族有原生的多边形格式，E1 会用统一的提示要求 0–1000 坐标的 JSON。
  - 经 `extract_json` 取值，只接受两种形状：
    - `[[a,b], …]`：单个多边形，至少 3 个点；
    - `[[[a,b], …], …]`：多个多边形。
  - 坐标必须是有限的非负数，≤1000；否则 `out_of_range` 或 `malformed`。点少于 3 个 → `too_few_points`。
  - 换算用 `geometry.norm1000_to_abs`。`axis_order="yx"` 时先交换两个坐标。

### 2. 适配器接口：框解析

- **`base.py`**：
  - 加抽象方法 `parse_boxes(self, text, *, width, height) -> BoxParse`，解析本家族的原生框格式。
  - 加具体方法 `parse_one_box(...) -> OneBoxParse`，建在 `parse_boxes` 之上：框的个数 ≠ 1 → `multiple_boxes`，没有框 → 原样给出 `no_box`。
  - `width`、`height` 是原图的宽高：三家族都按原图归一化（model_notes 已核实）。
- **类型**：`BoxParse`、`OneBoxParse` 移到 `src/parsing/`。`BoxParse` 增加 `labels`，与 boxes 等长，每个元素是 `str | None`。`internvl35` 里原来的名字保持可导入。
- **Qwen**（`qwen35.py`）：
  - 原生格式：JSON，形如 `[{"bbox_2d": [x1,y1,x2,y2], "label": "…"}, …]`，0–1000，x 在前。
  - 单个 dict 与 dict 的列表都接受；`label` 可缺；其他键忽略。
  - 另外接受裸的 `[x1,y1,x2,y2]` 或它的列表，按本家族的轴序和量程解释。
  - 数值：有限、非负的 JSON 数，整数或小数都接受（JSON 本身允许小数；不同于 InternVL 的整数语法）。
  - 失败判定：>1000 → `out_of_range`；x2<x1 或 y2<y1 → `inverted`；dict 缺 `bbox_2d` 或值不是 4 个数 → `malformed`。
- **Gemma**（`gemma4.py`）：
  - 原生格式：JSON，形如 `[{"box_2d": [y1,x1,y2,x2], "label": "…"}, …]`，**y 在前**，0–1000。换算走现有的 `box2d_to_xyxy`。
  - 裸的 4 元列表同样按 y 在前解释。
  - 其余规则同 Qwen。
- **InternVL**（`internvl35.py`）：
  - 现有规则不变，整数语法照 D19。
  - 新增：取紧挨在 `<box>` 前面的 `<ref>…</ref>` 作为 label，没有就是 None。
- **测试**：
  - 三家族各用 model_notes 里的真实输出做测试（原文逐字写进测试）。期望值按换算公式手算，写在注释里。
  - 另外覆盖：代码块与裸 JSON、前后说明文字、多个框、坏框、越界、颠倒、轴序（Gemma 的 y 在前必须有一条测试能抓住轴序写反）。

## 允许改动的文件

- 新建：`src/parsing/`（`__init__.py` 与若干模块）、`tests/test_parsing*.py`
- `src/adapters/base.py`：只加上面两个方法和类型的导入
- `src/adapters/qwen35.py`、`src/adapters/gemma4.py`、`src/adapters/internvl35.py`：只加或调整解析相关的函数与方法，不碰加载、生成、隐藏状态与冒烟相关的代码
- 现有测试：只在接口改动迫使时修改（例如测试里的假适配器要实现新的抽象方法），逐条写进偏离

**不许碰**：
- `scripts/smoke_adapter.py`
- `src/constants.py`、`src/metrics/`、`src/grid/`、`requirements.txt`

确有必要改这些文件时，写进偏离，由管理员处理。

## 环境与验收

- 用 conda 环境 `visint`。不需要 GPU，也不下载任何权重。
- 验收命令：
  - `conda run -n visint pytest tests/ -q` 全绿，报告新增测试数与总耗时。
  - `bash scripts/pytest_server_parity.sh` 全绿（D18：改了适配器就要跑）。
  - 三条 `--tiny` 冒烟退出码都是 0：
    - `conda run -n visint python scripts/smoke_adapter.py --tiny --model qwen3.5-0.8b --out <你的 scratchpad 子目录>/tiny_qwen.json`
    - Gemma（`gemma-4-e2b-it`）、InternVL（`internvl3.5-2b-hf`）同样各跑一条。
- 总结里逐条列出偏离，至少包括：
  - `ParseFailure` 最终的取值表；
  - `extract_json` 对代码块与裸 JSON 的具体规则；
  - Qwen/Gemma 接受小数而 InternVL 不接受的理由是否成立，你的意见；
  - 对 E1 提示设计的建议：哪些输出形态最容易解析失败。
