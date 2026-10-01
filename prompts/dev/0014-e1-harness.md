[DEV-TASK 0014] E1 评测框架：提示配置、任务运行器、评分、批处理验收与吞吐估算

## 背景

- E1 是 T 文本通道的零样本基线（简报 §8.2）。协议写在 `PREREGISTRATION.md` 的 E1 草稿 v0，是本任务的规格。
- 本任务把那份草稿实现成可以在服务器上逐模型、逐任务跑的框架。
- 已合入、本任务要用的部分：
  - 适配器：批量生成、冻结的图像口径、停止原因、`make_adapter`（D29）；
  - 解析层（D22）；
  - 数据加载器与子集（D30）；
  - 指标与统计（DEV-0001/0002）。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"评测与报表"一节
- 简报 §4 第 5、7、10、11 条，§6.3，§7.1，§8.2，§11.1，§11.5，§12
- `PREREGISTRATION.md` 的 E1 草稿（全文）
- `DECISIONS.md`：D10、D12、D19、D22、D28（第 3、5、9、10、11 条）、D29、D30
- `docs/model_notes.md` 里各家族原生提示的出处：Qwen §5.5、Gemma §6.5、InternVL §7.5
- 代码：`src/adapters/`、`src/parsing/`、`src/data/`、`src/metrics/`、`src/stats.py`、`src/accounting.py`、`src/runlog.py`

## 任务

### 1. 提示配置 `configs/e1_prompts.json`（冻结后入预注册）

- 结构：任务 × 家族 → 提示模板原文、出处链接与引用原文、`max_new_tokens`、解析函数名。
- **定位与检测**：用各家族官方文档或 cookbook 的原生写法。
  - 定位用 model_notes 已记的三条原文。
  - 多类别检测去官方文档或 cookbook 查原文，引用写进配置：
    - Gemma 用 "detect {类别1}, {类别2}, …"；
    - Qwen 用 Qwen3-VL cookbook 里的检测写法；
    - InternVL 用其文档里的检测写法。
  - 查不到原生写法的，用统一写法并标明"非原生"。
- **多选与问答**（CV-Bench、EmbSpatial、BLINK）：用 benchmark 自带的提示与选项格式（简报 §4 第 5 条），出处写明。
- **统一提示**（多边形、16×16 数字网格、NYU 点对、计数）：
  - 写明要求的输出格式。多边形是 `[[x, y], …]`，坐标为 0–1000，x 在前，只给一个 JSON 块。
  - CountBench 的问句由它的 caption 确定性地生成，规则写明；参考常见的 CountBench 评测写法，并写明出处。
- 每个任务只用一个提示，不调提示。
- 开发子集只用于核对格式、截断率与解析失败，不看准确率。在开发子集上发现截断率高时，可以调 `max_new_tokens`，记录调整理由。
- 配置的 sha256 写进输出的运行记录。

### 2. 任务运行器（`src/e1/` 新包，入口 `scripts/run_e1.py`）

- 参数：`--model <键> --task <任务> --subset <eval|dev|think|think8192|extra|…> --policy <默认 E1 口径；附加对照用 qwen:1024、internvl:native> --batch-size --out <目录> [--resume] [--limit N] [--tiny]`。
- 流程：加载器 → 按提示配置组装输入 → `generate_batch` → 解析 → 评分。
  - 思考子集用 `enable_thinking=True`，预算 4096 或 8192；其余一律关思考。
- **坐标**：解析结果一律换成原图的绝对像素（解析层已做），评分在原图尺寸上进行。
- **输出**：按简报 §11.1，写进 `experiments/E1/<运行编号>/<模型>/<任务>/`：
  - `predictions.jsonl`：每条样本的 id、raw_text、text、解析结果、失败原因、是否截断、思考是否未闭合、生成 token 数、视觉 token 数与像素数；
  - `metrics.json`：标题指标与 bootstrap 95% CI、ParseTally、截断率、分档或分划分的结果；
  - `compute.json`：按 `src/accounting.py`；
  - `config.yaml`、`README.md`。
- **断点续传**：按批写入 jsonl，`--resume` 跳过已完成的 id，再算一遍得到的结果逐字节相同。
- **指标**（用已合入的 `src/metrics`，不另写）：
  - 定位：Acc@IoU0.5；
  - 多边形：cIoU 与 gIoU，GT 用 RefCOCO 的 mask；
  - 16×16 网格：网格上采样到原图后的 gIoU；
  - 计数：准确率；
  - 检测：F1@0.5 为标题，随机打破平局后平均的 mAP 作补充（N9 → (b)）；另报编造类别的比例，即预测的类别不在给定列表里；
  - 点对与多选：准确率。
- **诊断**：
  - 坐标诊断：全部坐标 ≤1 且含小数的条数（D30 沿用 D22-2）；
  - 实际看到的像素数分布（D28-3）。

### 3. 批处理验收模式（D28-5）

- `scripts/run_e1.py --acceptance`：在开发子集上，同一模型与任务，逐条（batch=1）跑一遍，批量跑一遍。
- 报告：
  - 解析后答案的一致率，必须 ≥ 98%；
  - 两遍指标之差，以及按样本配对的 bootstrap CI，指标差必须落在 CI 内；
  - raw_text 逐字相同的比例。
- 覆盖任务：每个任务都要；多边形与网格按解析后的 IoU 判一致，规则写明。
- 输出一张汇总表，供管理员写进预注册与 G1。

### 4. 吞吐与算力估算模式

- `--estimate`：在开发子集上，按每个任务的真实提示与生成上限，测每个模型、每个 batch 大小的样本/秒与 token/秒。
- 按预注册的 n 外推每个任务、每个模型的 GPU 小时，并选出能装进显存的批大小。
- 输出 `estimate.json` 与一张表，管理员据此补预注册第 9 节。

### 5. 测试

- 用 `--tiny` 小模型与 `tests/fixtures/e1/` 的 fixture，每个任务端到端跑通：
  - 输出文件齐全；
  - 断点续传逐字节一致；
  - 验收模式在 CPU 上一致率为 100%。
- 提示配置：结构、出处字段齐全、sha256 稳定。
- 每个任务的评分：在构造的小例子上与手算一致。
- 检测里编造类别的计数、截断判失败、思考未闭合单独计数。

## 允许改动的文件

- 新建：`src/e1/`、`scripts/run_e1.py`、`configs/e1_prompts.json`、`tests/test_e1_*.py`，以及必要的 fixture
- 已合入模块的小修，需写进偏离：`src/parsing/`、`src/metrics/` 里确有必要的函数补充

**不许碰**：`src/adapters/`（接口已定，缺什么写进偏离）、`src/health/` 与 `scripts/health_check.py`（另一个 agent 在改）、`src/synth/`、`src/data/`（只读用）、`requirements.txt`。

## 环境与验收

- 用 conda 环境 `visint`，不装包，不下载权重（tokenizer 与 processor 的小文件可以）。
- 验收：
  - `conda run -n visint pytest tests/ -q` 全绿；
  - `bash scripts/pytest_server_parity.sh` 全绿；
  - 每个任务一条 `--tiny` 端到端；
  - `--acceptance --tiny` 与 `--estimate --tiny` 各一条。
- 总结里逐条列出：
  - 每个家族、每个任务的提示原文与出处；查不到原生写法的有哪些；
  - CountBench 问句的生成规则；
  - 各任务的 `max_new_tokens`；
  - 你对预注册草稿需要修改之处的建议。
