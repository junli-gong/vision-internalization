[RUN-TASK 0005] 思考行为事实探针：8 个 E1 模型 × 12 条提示 × 3 种生成设置（Phase 0；E1 思考子集协议的输入）

## 为什么做

RUN-0004 里，InternVL 加了 R1 system prompt、greedy 解码，直接回答 "Pink"，没有写思考段。
E1 要在固定子集上对比开/关思考（简报 §6.3）。在那之前，先弄清几件事：
- 各家族在 greedy 和推荐采样下是否真的思考；
- 思考有多长；
- 会不会在预算内写不完。

这是 Phase 0 的事实探针，不是 E1 基线：不评分，不调提示词。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D13（生成、思考开关、停止 token）、D19（思考段判定）、D20（本任务与探针用图）
- `reports/0004-g0-smoke-all.md` §3.6

本任务只运行，外加写一个探针脚本，放在 `experiments/P0/0005_thinking_probe/probe_scripts/`。
不改仓库代码：`src/`、`scripts/`、`tests/`、`configs/` 一律不动。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `0277617`。不包含就停下报告。

## 2. 环境

照旧：
- 环境块每行一个 export；
- D6 磁盘红线：`df /` 低于 3 GB 就停；
- 内存盘任何时刻不超过 25 GB。

## 3. 探针（照做，不要改设计）

### 3.1 模型与顺序

按下面的顺序跑，每个模型跑完就从内存盘删掉。全程只保留 qwen3.5-0.8b，这样任何时刻都不会超过 25 GB（9B 是 19.31 GB）。

1. `internvl3.5-2b-hf`
2. `qwen3.5-0.8b`
3. `gemma-4-e2b-it`
4. `qwen3.5-2b`
5. `internvl3.5-4b-hf`
6. `gemma-4-e4b-it`
7. `qwen3.5-4b`
8. `qwen3.5-9b`

- 下载用 registry 里的 revision。
- 删除用 `scan_cache_dir(...).delete_revisions(...)`（D13-6）。
- 每次下载或删除前后都记进 `disk_ledger.tsv`。

### 3.2 适配器的构造

与冒烟完全相同：复用 `scripts/smoke_adapter.py` 的 `parse_args` 和 `family_suite(args)`，调用返回的 `setup(args)`，只取其中的适配器。

- 参数：`--model <键> --device cuda --dtype bf16 --image <图 A 的路径> --out <任意临时路径>`。
- 这样得到的设置是：Qwen 固定 480×640；Gemma `max_soft_tokens=280`；InternVL 单 tile。
- 不跑冒烟检查，也不调用 `add_grid_tokens` 和 `inject_query_tokens`：模型保持原样。

### 3.3 图像

- **图 A**：`$PROJ/tmp/000000039769.jpg`，sha256 应与 RUN-0004 相同。
- **图 B**：COCO val2017 的 `000000000139.jpg`。RUN-0001 用过这张图，先在 `$PROJ` 下找副本。找不到就按 URL `http://images.cocodataset.org/val2017/000000000139.jpg` 单张下载到 `$PROJ/tmp`。

两张图的 sha256 都写进 `image_sha256.txt`。

### 3.4 提示

一律用 `order="image_first"`，单张图。提示原样照抄，一个字符都不要改：

| id | 图 | 提示 |
|---|---|---|
| A1 | A | `What is the main color in this image? Answer in one word.` |
| A2 | A | `How many cats are in this image? Answer with a single number.` |
| A3 | A | `How many remote controls are in this image? Answer with a single number.` |
| A4 | A | `Locate the cat on the left in this image and output its bounding box.` |
| A5 | A | `Which cat is closer to the camera, the left one or the right one? Answer 'left' or 'right'.` |
| A6 | A | `If this image were rotated 90 degrees clockwise, which edge of the image would the remote controls be closest to? Answer top, bottom, left, or right.` |
| B1 | B | `Describe this image in detail.` |
| B2 | B | `How many chairs are in this image? Answer with a single number.` |
| B3 | B | `Which is closer to the camera, the television or the dining table? Answer with one of the two.` |
| B4 | B | `Output the bounding box of the television in this image.` |
| B5 | B | `Where is the clock relative to the television? (A) left (B) right (C) above (D) below. Answer with the option letter.` |
| B6 | B | `Imagine walking from the camera position straight to the television. Which objects would you pass on your left? Think it through, then answer.` |

### 3.5 三种生成设置

每条提示依次跑下面三种设置，都走适配器的 `generate(inputs, enable_thinking=…, max_new_tokens=…, **gen_kwargs)`：

| 设置 | enable_thinking | 解码 | max_new_tokens |
|---|---|---|---|
| `nothink_greedy` | False | greedy（适配器默认） | 512 |
| `think_greedy` | True | greedy（适配器默认） | 4096 |
| `think_sampled` | True | 采样，参数见下 | 4096 |

`think_sampled` 的参数：
- **Qwen 与 Gemma**：用固定 revision 的 `generation_config.json` 里的 `do_sample`、`temperature`、`top_p`、`top_k`，原样照抄；缺哪个就不传哪个。如果某个模型的 `generation_config` 里没有 `do_sample=True`，这个模型就跳过 `think_sampled`，并在 report 里写明。
- **InternVL**：`generation_config` 里没有采样参数，按 model card 原文用 `do_sample=True, temperature=0.6`，其余不传。
- 每条生成之前都设种子：`torch.manual_seed(s)` 和 `torch.cuda.manual_seed_all(s)`，其中 `s = 1000 + 提示序号`（A1=0 … B6=11）。
- 每个模型实际用的参数和出处写进 `gen_params.json`。
- Qwen 的 model card 如果另有思考模式的推荐值，把原文抄进 report；本次不用。

### 3.6 每条生成的记录

`records.jsonl`，每条生成一行，字段如下：

- **基本信息**：`model`、`prompt_id`、`image`、`condition`、`gen_kwargs`（实际传入的值）、`seed`、`max_new_tokens`
- **长度**：`n_prompt_tokens`、`n_visual_tokens`、`n_generated_tokens`
- **结束方式**：
  - `ended_with_stop_token`
  - `hit_max_new_tokens`：生成满 max_new_tokens，且最后一个不是停止 token
- **思考段**：标记 token 的 id 从 tokenizer 查，不要写死，查到的 id 写进 `gen_params.json`。
  - `think_open`：
    - Qwen 开思考时，思考段开头由提示预填，记 True，并记 `think_open_source="prompt"`；
    - Gemma 看输出里有没有 `<|channel>`；
    - InternVL 看输出里有没有 `<think>`。
  - `think_close`：Qwen 看 `</think>`，Gemma 看 `<channel|>`，InternVL 看 `</think>`。
  - `n_think_tokens`：从开头标记之后（Qwen 从输出第一个 token 起）到结束标记之前；没有闭合就数到输出末尾。
  - `n_answer_tokens`：结束标记之后、停止 token 之前的 token 数。没有思考段时，是全部非停止 token。
- **适配器给出的**：`thinking_unclosed`、`text`（全文）、`raw_text`（全文）
- **时间**：`wall_time_s`、`tokens_per_s`（= n_generated_tokens / wall_time_s）

### 3.7 汇总

用脚本从 `records.jsonl` 生成 `summary.md`：

- **表 1**：模型 × 设置，每格 n=12。列出：
  - 开思考条数、闭合条数、写满预算条数；
  - 思考 token 的中位数和最大值（只算开了思考的）；
  - 回答 token 的中位数；
  - tokens/s 的均值；
  - 墙钟合计。
- **表 2**：`think_greedy` 下没开思考的提示 id，按模型列出。
- **表 3**：`nothink_greedy` 下，输出里是否出现任何思考标记，按模型列出。

## 4. 时间与停止

- 预计 1–3 小时。
- 从第一条生成算起，累计超过 4 小时：跑完当前这一条就停，已完成的部分照常推回，并在 report 里写明停在哪里。
- 某个模型加载失败或 OOM：记录完整报错，跳过它，继续下一个。
- 全部失败才算"停下报告"的条件。

## 5. 产出

- `experiments/P0/0005_thinking_probe/`：
  - `records.jsonl`、`summary.md`、`gen_params.json`
  - `image_sha256.txt`、`disk_ledger.tsv`
  - `README.md`、`probe_scripts/`
- `reports/0005-thinking-probe.md`，按 report 模板：
  - 第 3 节原样贴表 1–3；
  - 每个模型附两条 `think_greedy` 的 `raw_text` 开头 300 字符，选 A4 和 B6；
  - 第 6 节写磁盘与内存盘收支；
  - 只贴数字与事实，不做解读。

## 6. 回传

分两批：
1. 前三个模型（internvl3.5-2b-hf、qwen3.5-0.8b、gemma-4-e2b-it）跑完后推一次，附当时的 `summary.md`。
2. 全部跑完后推第二批和 report。

```bash
git add <逐个路径> && git commit -m "[results] 0005: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0005-thinking-probe.md` 追加进完成清单，重新进入值守。
