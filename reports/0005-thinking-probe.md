# RUN-0005 — 思考行为事实探针：8 个 E1 模型 × 12 条提示 × 3 种生成设置

服务器 CC，b200，2026-09-30 约 09:17（接单）–10:27 UTC（值守接单）。Phase 0 事实探针：不评分，不调提示词，不是 E1 基线。只贴数字与事实，不做解读。原始数据在 `experiments/P0/0005_thinking_probe/`。

## 0. 一句话观察（只陈述，不解读）

8 个模型全部跑完，共 240 条记录。4 个 Qwen3.5 的固定 revision 都没有 `generation_config.json`，按任务书跳过 `think_sampled`，所以 Qwen 每个 24 条，Gemma 与 InternVL 每个 36 条。

生成部分从第一条生成（09:35:12）到最后一个模型结束（10:25:21）约 50 分钟，逐条 `wall_time_s` 之和为 44.4 分钟，没有触发 4 小时预算。没有模型加载失败，也没有 OOM。

内存盘峰值 21.091 GB（9B 在盘时）；根盘最低 5.562 GB。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `8933547`；`0277617` 在 HEAD 历史中。
- 本次没有修改 `src/`、`scripts/`、`tests/`、`configs/`。新写的只有 `experiments/P0/0005_thinking_probe/probe_scripts/` 下的探针、汇总与驱动脚本。
- 适配器的构造：`scripts/smoke_adapter.py` 的 `parse_args(["--model", KEY, "--device", "cuda", "--dtype", "bf16", "--image", <图 A>, "--out", <临时路径>])` → `family_suite(args)` → `setup(args)`，只取返回的适配器，不跑冒烟检查，不调用 `add_grid_tokens`/`inject_query_tokens`。
- 图像（`image_sha256.txt`）：
  - 图 A `$PROJ/tmp/000000039769.jpg`，sha256 `dea9e7ef…4563b04e`，与 RUN-0004 相同；
  - 图 B 用 RUN-0001 的副本 `$PROJ/tmp/coco/000000000139.jpg`，sha256 `ffe0f0ce…b3b8a8d`，没有重新下载。
- 本次的推送：`9f198a6`（批次 1：前三个模型，96 条，附当时的 summary）；批次 2（其余五个模型与本 report）为本 commit。

## 2. 执行记录

驱动 `probe_scripts/run_models.sh` 对每个模型依次执行：记账 → `snapshot_download(revision)` → 记账 → 探针 → 记账 → 用 `scan_cache_dir().delete_revisions()` 删除（qwen3.5-0.8b 除外）→ 记账。时间取自 `logs/driver.log`：

| 模型 | 探针开始 → 结束（UTC） | 探针 rc | 记录条数 |
|---|---|---|---|
| internvl3.5-2b-hf | 09:35:07 → 09:45:02 | 0 | 36 |
| qwen3.5-0.8b | 09:45:02 → 09:45:09 | 1 | 0（见 §5） |
| gemma-4-e2b-it | 09:45:09 → 09:49:27 | 0 | 36 |
| qwen3.5-0.8b | 09:50:54 → 09:53:48 | 0 | 24 |
| qwen3.5-2b | 09:54:22 → 09:58:05 | 0 | 24 |
| internvl3.5-4b-hf | 09:58:05 → 10:11:29 | 0 | 36 |
| gemma-4-e4b-it | 10:11:29 → 10:16:09 | 0 | 36 |
| qwen3.5-4b | 10:16:10 → 10:20:19 | 0 | 24 |
| qwen3.5-9b | 10:20:19 → 10:25:20 | 0 | 24 |

- GPU：NVIDIA B200，每个模型各自一个进程，bf16，sdpa。
- 警告：Qwen 的探针日志里有 `causal_conv1d_update` 回退到 torch 的提示；InternVL 的日志里有 tie weights 提示。两者与 RUN-0004 相同，原文见 `logs/probe_*.log`。

各模型实际用的参数（`gen_params.json`）：

| 模型 | think_sampled 的 gen_kwargs | 出处 | 思考标记 id（从 tokenizer 查） | 停止 token id | 适配器设置 | 加载 s |
|---|---|---|---|---|---|---|
| internvl3.5-2b-hf | {"do_sample": true, "temperature": 0.6} | model card §Thinking Mode 原文: "When enabling Thinking mode, we recommend setting `do_sample=True` and `temperature=0.6`"; generation_config.json 无采样参数 | {"<think>": 151667, "</think>": 151668} | {"<|im_end|>": 151645} | {"tile_policy": {"crop_to_patches": false, "max_patches": 1, "min_patches": 1, "tile_px": 448}} | 3.74 |
| qwen3.5-0.8b | 跳过 | Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 没有 generation_config.json → 按任务书跳过 think_sampled | {"<think>": 248068, "</think>": 248069} | {"<|im_end|>": 248046, "<|endoftext|>": 248044} | {"image_size": [480, 640]} | 4.99 |
| gemma-4-e2b-it | {"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 64} | google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7 generation_config.json 原样: {"bos_token_id": 2, "do_sample": true, "eos_token_id": [1, 106, 50], "pad_token_id": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "transformers_version": "5.5.0.dev0"} | {"<|channel>": 100, "<channel|>": 101, "<|think|>": 98} | {"<turn|>": 106, "<eos>": 1} | {"max_soft_tokens": 280} | 7.32 |
| qwen3.5-2b | 跳过 | Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc 没有 generation_config.json → 按任务书跳过 think_sampled | {"<think>": 248068, "</think>": 248069} | {"<|im_end|>": 248046, "<|endoftext|>": 248044} | {"image_size": [480, 640]} | 6.11 |
| internvl3.5-4b-hf | {"do_sample": true, "temperature": 0.6} | model card §Thinking Mode 原文: "When enabling Thinking mode, we recommend setting `do_sample=True` and `temperature=0.6`"; generation_config.json 无采样参数 | {"<think>": 151667, "</think>": 151668} | {"<|im_end|>": 151645} | {"tile_policy": {"crop_to_patches": false, "max_patches": 1, "min_patches": 1, "tile_px": 448}} | 4.77 |
| gemma-4-e4b-it | {"do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 64} | google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2 generation_config.json 原样: {"bos_token_id": 2, "do_sample": true, "eos_token_id": [1, 106, 50], "pad_token_id": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "transformers_version": "5.5.0.dev0"} | {"<|channel>": 100, "<channel|>": 101, "<|think|>": 98} | {"<turn|>": 106, "<eos>": 1} | {"max_soft_tokens": 280} | 7.91 |
| qwen3.5-4b | 跳过 | Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 没有 generation_config.json → 按任务书跳过 think_sampled | {"<think>": 248068, "</think>": 248069} | {"<|im_end|>": 248046, "<|endoftext|>": 248044} | {"image_size": [480, 640]} | 6.79 |
| qwen3.5-9b | 跳过 | Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a 没有 generation_config.json → 按任务书跳过 think_sampled | {"<think>": 248068, "</think>": 248069} | {"<|im_end|>": 248046, "<|endoftext|>": 248044} | {"image_size": [480, 640]} | 7.49 |

- `nothink_greedy` 与 `think_greedy` 都不传 gen_kwargs，用适配器默认的 greedy（`do_sample=False`）。
- 每条生成前设种子：`torch.manual_seed(s)`、`torch.cuda.manual_seed_all(s)`，其中 s = 1000 + 提示序号。
- **Qwen model card 另有思考模式的推荐值（原文，本次未用；RUN-0001 model_notes §5.2）**：
  - 0.8B/2B 的 card："Thinking mode for text tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`"；"Thinking mode for VL or precise coding (e.g. WebDev) tasks : `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`"。
  - 4B/9B 的 card："Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`"；"Thinking mode for precise coding tasks (e.g. WebDev): `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`"。

## 3. 结果表（原样，由 `probe_scripts/p9_summary.py` 从 records.jsonl 生成；逐条记录表见 `summary.md` 的附表）

## 表 1：模型 × 设置（每格应为 n=12）

| 模型 | 设置 | n | 开思考 | 闭合 | 写满预算 | 以停止 token 结束 | 思考 token 中位数（仅开思考） | 思考 token 最大（仅开思考） | 回答 token 中位数 | tokens/s 均值 | 墙钟合计 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| internvl3.5-2b-hf | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 9 | 58.54 | 14.0 |
| internvl3.5-2b-hf | think_greedy | 12 | 8 | 3 | 5 | 7 | 4095 | 4095 | 4.5 | 67.45 | 301.7 |
| internvl3.5-2b-hf | think_sampled | 12 | 9 | 5 | 4 | 8 | 757 | 4095 | 12 | 77.69 | 272.4 |
| qwen3.5-0.8b | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 25 | 62.57 | 12.2 |
| qwen3.5-0.8b | think_greedy | 12 | 12 | 10 | 2 | 10 | 226.5 | 4096 | 2 | 82.63 | 154.4 |
| qwen3.5-0.8b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||
| gemma-4-e2b-it | nothink_greedy | 12 | 0 | 0 | 1 | 11 | — | — | 21.5 | 43.41 | 17.2 |
| gemma-4-e2b-it | think_greedy | 12 | 11 | 11 | 0 | 12 | 303 | 1132 | 13.5 | 54.55 | 108.3 |
| gemma-4-e2b-it | think_sampled | 12 | 11 | 11 | 0 | 12 | 273 | 1070 | 13.5 | 52.99 | 103.1 |
| qwen3.5-2b | nothink_greedy | 12 | 0 | 0 | 1 | 11 | — | — | 22 | 60.36 | 18.2 |
| qwen3.5-2b | think_greedy | 12 | 12 | 9 | 3 | 9 | 311.5 | 4096 | 2 | 83.26 | 185.3 |
| qwen3.5-2b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||
| internvl3.5-4b-hf | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 1.5 | 45.84 | 16.0 |
| internvl3.5-4b-hf | think_greedy | 12 | 11 | 5 | 6 | 6 | 4095 | 4095 | 1 | 71.09 | 342.7 |
| internvl3.5-4b-hf | think_sampled | 12 | 11 | 7 | 5 | 7 | 1222 | 4095 | 2 | 70.22 | 418.7 |
| gemma-4-e4b-it | nothink_greedy | 12 | 0 | 0 | 0 | 12 | — | — | 1 | 33.74 | 19.3 |
| gemma-4-e4b-it | think_greedy | 12 | 11 | 11 | 0 | 12 | 312 | 761 | 1.5 | 42.86 | 104.9 |
| gemma-4-e4b-it | think_sampled | 12 | 10 | 10 | 0 | 12 | 423.5 | 775 | 1.5 | 41 | 118.3 |
| qwen3.5-4b | nothink_greedy | 12 | 0 | 0 | 2 | 10 | — | — | 2.5 | 44.02 | 25.0 |
| qwen3.5-4b | think_greedy | 12 | 12 | 10 | 2 | 10 | 232.5 | 4096 | 2 | 60.46 | 196.7 |
| qwen3.5-4b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||
| qwen3.5-9b | nothink_greedy | 12 | 0 | 0 | 2 | 10 | — | — | 11.5 | 45.32 | 22.1 |
| qwen3.5-9b | think_greedy | 12 | 12 | 11 | 1 | 11 | 296.5 | 4096 | 2.5 | 61.16 | 211.4 |
| qwen3.5-9b | think_sampled | 0 | 跳过：Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a 没有 generation_config.json → 按任务书跳过 think_sampled ||||||||||

## 表 2：think_greedy 下没开思考的提示 id

| 模型 | 没开思考的提示 | 个数 |
|---|---|---|
| internvl3.5-2b-hf | A1, A5, B1, B3 | 4 |
| qwen3.5-0.8b | （无） | 0 |
| gemma-4-e2b-it | B4 | 1 |
| qwen3.5-2b | （无） | 0 |
| internvl3.5-4b-hf | B1 | 1 |
| gemma-4-e4b-it | A1 | 1 |
| qwen3.5-4b | （无） | 0 |
| qwen3.5-9b | （无） | 0 |

## 表 3：nothink_greedy 下输出里是否出现任何思考标记

| 模型 | 出现思考标记的提示（open/close/其他标记计数） | 个数 / n |
|---|---|---|
| internvl3.5-2b-hf | （无） | 0 / 12 |
| qwen3.5-0.8b | （无） | 0 / 12 |
| gemma-4-e2b-it | （无） | 0 / 12 |
| qwen3.5-2b | （无） | 0 / 12 |
| internvl3.5-4b-hf | （无） | 0 / 12 |
| gemma-4-e4b-it | （无） | 0 / 12 |
| qwen3.5-4b | （无） | 0 / 12 |
| qwen3.5-9b | （无） | 0 / 12 |

**写满预算（`hit_max_new_tokens`）的提示：**
- internvl3.5-2b-hf / think_greedy：A6, B2, B4, B5, B6
- internvl3.5-2b-hf / think_sampled：A4, A6, B4, B6
- qwen3.5-0.8b / think_greedy：B2, B6
- gemma-4-e2b-it / nothink_greedy：B1
- qwen3.5-2b / nothink_greedy：A6
- qwen3.5-2b / think_greedy：A6, B2, B6
- internvl3.5-4b-hf / think_greedy：A4, A5, A6, B2, B4, B6
- internvl3.5-4b-hf / think_sampled：A1, A4, A6, B4, B6
- qwen3.5-4b / nothink_greedy：B1, B6
- qwen3.5-4b / think_greedy：B2, B6
- qwen3.5-9b / nothink_greedy：B1, B6
- qwen3.5-9b / think_greedy：B2

**开了思考但没闭合的提示：**
- internvl3.5-2b-hf / think_greedy：A6, B2, B4, B5, B6
- internvl3.5-2b-hf / think_sampled：A4, A6, B4, B6
- qwen3.5-0.8b / think_greedy：B2, B6
- qwen3.5-2b / think_greedy：A6, B2, B6
- internvl3.5-4b-hf / think_greedy：A4, A5, A6, B2, B4, B6
- internvl3.5-4b-hf / think_sampled：A4, A6, B4, B6
- qwen3.5-4b / think_greedy：B2, B6
- qwen3.5-9b / think_greedy：B2

**`think_greedy` 的 raw_text 开头 300 字符（A4 与 B6，原样，`repr` 形式）：**

- **internvl3.5-2b-hf A4**（生成 1969，开=True，闭=True，满预算=False）：

  ```
  "<think>\nOkay, let's see. The user wants me to locate the cat on the left in the image and output its bounding box. The image shows two cats lying on a pink couch with remote controls next to them.\n\nFirst, I need to identify the cat on the left. The cat on the right is the one with a more prominent s"
  ```
- **internvl3.5-2b-hf B6**（生成 4096，开=True，闭=False，满预算=True）：

  ```
  "<think>\nOkay, let's see. The user wants to know which objects you'd pass on your left when walking straight from the camera position towards the television. \n\nFirst, I need to visualize the scene. The image shows a living room and dining area connected. The camera is in the living room, looking towa"
  ```
- **qwen3.5-0.8b A4**（生成 103，开=True，闭=True，满预算=False）：

  ```
  "Got it, let's see. The cat on the left is the one with the striped pattern, lying on the pink surface. So I need to find its coordinates. The cat is from the left side, so the bounding box would be [17, 114, 492, 985]. Let me check again. Yes, that's the left cat.\n</think>\n\n[17, 114, 492, 985]<|im_e"
  ```
- **qwen3.5-0.8b B6**（生成 4096，开=True，闭=False，满预算=True）：

  ```
  "The user wants me to simulate walking from the camera position towards the television.\n\n1.  **Identify the camera position:** The camera is positioned low, looking slightly up at the floor. It's in the foreground, near a pink table.\n2.  **Identify the television:** The TV is on the left side of the "
  ```
- **gemma-4-e2b-it A4**（生成 143，开=True，闭=True，满预算=False）：

  ```
  '<|channel>thought\nHere are the bounding box detections for the requested objects:\n- The user wants the bounding box for "the cat on the left".\n\n1.  **Analyze the image:** The image contains two cats lying on a pink surface.\n2.  **Identify the cat on the left:** There is a striped tabby cat lying on '
  ```
- **gemma-4-e2b-it B6**（生成 544，开=True，闭=True，满预算=False）：

  ```
  "<|channel>thought\nHere's a thinking process to arrive at the suggested answer:\n\n1.  **Analyze the Request:** The user wants to know which objects would be on the *left* side of the path taken from the camera position *straight to the television*.\n\n2.  **Establish the Starting Point (Camera Position)"
  ```
- **qwen3.5-2b A4**（生成 134，开=True，闭=True，满预算=False）：

  ```
  "Got it, let's look at the image. The cat on the left is the one with the striped pattern, lying on the pink surface. So we need to find its coordinates. Let's check the bounding box. The left cat is from the left side of the image to where its body ends. So the coordinates would be [13, 117, 490, 98"
  ```
- **qwen3.5-2b B6**（生成 4096，开=True，闭=False，满预算=True）：

  ```
  'The user wants me to identify objects on the left side of the path if I walk straight towards the television.\n\n1.  **Analyze the scene:**\n    *   **Camera position:** The camera is in the living room area, looking towards the dining/kitchen area.\n    *   **Path:** Walking "straight to the television'
  ```
- **internvl3.5-4b-hf A4**（生成 4096，开=True，闭=False，满预算=True）：

  ```
  "<think>\nOkay, let's look at the image. There are two cats on a red couch. The one on the left is a kitten with a striped pattern. It's lying down with its head resting on the couch, eyes closed, and a remote control next to it. The kitten's body is stretched out, and it has a green collar. The other"
  ```
- **internvl3.5-4b-hf B6**（生成 4096，开=True，闭=False，满预算=True）：

  ```
  "<think>\nOkay, let's see. The user is asking about the objects you would pass on your left if you walk from the camera position straight to the television. First, I need to visualize the image. The room has a yellow wall with a TV on the left side. The TV is on a stand with some items below it, like "
  ```
- **gemma-4-e4b-it A4**（生成 90，开=True，闭=True，满预算=False）：

  ```
  '<|channel>thought\nHere are the bounding box detections for the requested objects:\n- The cat on the left is located at [115, 19, 989, 494].<channel|>```json\n[\n  {"box_2d": [115, 19, 989, 494], "label": "the cat on the left"}\n]\n```<turn|>'
  ```
- **gemma-4-e4b-it B6**（生成 597，开=True，闭=True，满预算=False）：

  ```
  "<|channel>thought\nHere's a thinking process to arrive at the answer:\n\n1.  **Analyze the Request:** The user wants to know which objects would be passed on the *left* if one walks from the camera position straight to the television.\n\n2.  **Establish the Starting Point and Direction:**\n    *   **Start"
  ```
- **qwen3.5-4b A4**（生成 115，开=True，闭=True，满预算=False）：

  ```
  "Got it, let's look at the image. The cat on the left is the one with the green collar, right? So we need to find its coordinates. Let's check the bounding box. The left cat is from the left side of the image to where it ends. So the coordinates would be [11, 111, 493, 986]. Let me confirm. Yes, that"
  ```
- **qwen3.5-4b B6**（生成 4096，开=True，闭=False，满预算=True）：

  ```
  'The user wants me to identify objects on the left side of the path from the camera to the TV.\n\n1.  **Identify the path:** The camera is positioned in the foreground, looking into a living/dining area. The TV is on the left wall. So, the path is generally straight ahead but angled slightly left towar'
  ```
- **qwen3.5-9b A4**（生成 139，开=True，闭=True，满预算=False）：

  ```
  "Got it, let's see. The cat on the left is the one with the striped pattern, lying on the pink surface. I need to find its coordinates. So the left cat's position is from the left side of the image to where it's lying. Let me check the bounding box. The left cat is [10, 121, 490, 986]. Yeah, that's t"
  ```
- **qwen3.5-9b B6**（生成 3727，开=True，闭=True，满预算=False）：

  ```
  'The user wants me to identify objects on the left side of the path from the camera to the television.\n\n1.  **Identify the starting point:** The camera position.\n2.  **Identify the destination:** The television.\n3.  **Identify the path:** "Straight to the television". This implies a line of sight fro'
  ```

## 4. 与任务书假设不符的发现

1. **4 个 Qwen3.5 都没有 `generation_config.json`**（0.8B、2B、4B、9B 的固定 revision，`logs/genconfig_check.txt`），按 §3.5 跳过 `think_sampled`；表 1 里对应的 4 格 n=0。
2. **Gemma 的思考标记计数口径**：`<|channel>` 后面紧跟 "thought\n"，这几个 token 计入 `n_think_tokens`（README 口径）。
3. **Qwen 适配器的 `token_id()` 只认 attach 时缓存的 token**，不含 `<think>`（`src/adapters/qwen35.py` 第 437 行附近只缓存了 `</think>` 等）。探针第一次在 qwen3.5-0.8b 上因此抛 `KeyError: '<think>'`（§5）。改为按任务书"从 tokenizer 查"，用 `tokenizer.convert_tokens_to_ids` 查，并校验它编码成单个 token。
4. **根盘在 09:44 左右掉了约 0.30 GB**（5.863 → 5.563 GB），而 `$PROJ` 没有变化（0.488 GB）。同一时间 `/root/.vscode-server/extensions/anthropic.claude-code-2.1.285-linux-x64/…`（240 MB）与 `/root/.vscode-server/data/CachedExtensionVSIXs/…`（112 MB）被写入，不是本项目的文件，本项目没有动它们。之后根盘回到 5.67 GB 左右。

## 5. 自己造成的错误或浪费

- qwen3.5-0.8b 第一次运行时，探针脚本用 `ad.token_id('<think>')` 查思考标记，抛 `KeyError`，退出码 1，在任何生成之前就退出了（没有写入任何记录）。原始日志保留为 `logs/probe_qwen3.5-0.8b.attempt1_keyerror.log`。
  - 修复：标记与停止 token 一律从 tokenizer 查，并做单 token 校验。
  - 重跑：qwen3.5-0.8b 单独跑一次，24 条。
  - 代价：约 7 s，外加一次重跑的模型加载。
- 这次修复发生在批次 1 的三个模型之间：InternVL-2B 与 Gemma-E2B 是修复前跑的。修复只改了"怎样查 id"，不改口径；这两个模型当时查到的 id 也记在 `gen_params.json` 里，其中 InternVL 的 `<think>`=151667、`</think>`=151668，与 RUN-0002 §7.1 表一致。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0005_thinking_probe/`：
- `records.jsonl`（240 行）、`summary.md`、`gen_params.json`
- `image_sha256.txt`、`disk_ledger.tsv`、`probe_start_epoch.txt`
- `README.md`、`logs/`、`probe_scripts/`

另有本 report。

**内存盘**：
- 峰值 21.091 GB（Qwen3.5-0.8B 加 9B，上限 25 GB）；
- 各模型跑完即按 D13-6 删除，其中 Gemma E2B/E4B 与 InternVL-4B 这些小文件条目也一并清掉了；
- 现只剩 Qwen3.5-0.8B（1.770 GB）。**InternVL3.5-2B-HF 也已删除**（任务书 §3.1 要求只保留 0.8B），下次需要时重新下载（4.7 GB，约 10 s）。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 09:35:07 | before download internvl3.5-2b-hf | 5.863 | 6.486 | 0.488 |
| 09:35:07 | after download internvl3.5-2b-hf | 5.863 | 6.486 | 0.488 |
| 09:45:02 | after probe internvl3.5-2b-hf | 5.563 | 6.486 | 0.488 |
| 09:45:02 | after delete internvl3.5-2b-hf | 5.563 | 1.772 | 0.488 |
| 09:45:02 | before download qwen3.5-0.8b | 5.563 | 1.772 | 0.488 |
| 09:45:02 | after download qwen3.5-0.8b | 5.563 | 1.772 | 0.488 |
| 09:45:09 | after probe qwen3.5-0.8b | 5.563 | 1.772 | 0.488 |
| 09:45:10 | before download gemma-4-e2b-it | 5.563 | 1.772 | 0.488 |
| 09:45:28 | after download gemma-4-e2b-it | 5.563 | 12.051 | 0.488 |
| 09:49:27 | after probe gemma-4-e2b-it | 5.562 | 12.051 | 0.489 |
| 09:49:27 | after delete gemma-4-e2b-it | 5.562 | 1.772 | 0.489 |
| 09:50:54 | before download qwen3.5-0.8b | 5.674 | 1.772 | 0.489 |
| 09:50:54 | after download qwen3.5-0.8b | 5.674 | 1.772 | 0.489 |
| 09:53:48 | after probe qwen3.5-0.8b | 5.673 | 1.772 | 0.489 |
| 09:54:22 | before download qwen3.5-2b | 5.673 | 1.772 | 0.489 |
| 09:54:32 | after download qwen3.5-2b | 5.673 | 6.331 | 0.489 |
| 09:58:05 | after probe qwen3.5-2b | 5.671 | 6.331 | 0.489 |
| 09:58:05 | after delete qwen3.5-2b | 5.671 | 1.772 | 0.489 |
| 09:58:05 | before download internvl3.5-4b-hf | 5.671 | 1.772 | 0.489 |
| 09:58:24 | after download internvl3.5-4b-hf | 5.671 | 11.256 | 0.489 |
| 10:11:29 | after probe internvl3.5-4b-hf | 5.667 | 11.256 | 0.489 |
| 10:11:29 | after delete internvl3.5-4b-hf | 5.667 | 1.773 | 0.489 |
| 10:11:29 | before download gemma-4-e4b-it | 5.667 | 1.773 | 0.489 |
| 10:11:56 | after download gemma-4-e4b-it | 5.667 | 17.799 | 0.489 |
| 10:16:09 | after probe gemma-4-e4b-it | 5.666 | 17.799 | 0.489 |
| 10:16:10 | after delete gemma-4-e4b-it | 5.666 | 1.774 | 0.489 |
| 10:16:10 | before download qwen3.5-4b | 5.666 | 1.774 | 0.489 |
| 10:16:28 | after download qwen3.5-4b | 5.666 | 11.104 | 0.489 |
| 10:20:19 | after probe qwen3.5-4b | 5.665 | 11.104 | 0.489 |
| 10:20:19 | after delete qwen3.5-4b | 5.665 | 1.774 | 0.489 |
| 10:20:19 | before download qwen3.5-9b | 5.665 | 1.774 | 0.489 |
| 10:21:16 | after download qwen3.5-9b | 5.664 | 21.091 | 0.489 |
| 10:25:20 | after probe qwen3.5-9b | 5.663 | 21.091 | 0.489 |
| 10:25:21 | after delete qwen3.5-9b | 5.663 | 1.775 | 0.489 |
| 10:26:23 | after all models | 5.662 | 1.775 | 0.489 |
| 10:27:21 | before 0005 report | 5.662 | 1.775 | 0.490 |

- 根盘最低 5.562 GB（红线 3 GB），其中约 0.30 GB 的下降来自其他用户的文件（§4 第 4 条）。
- 没有删除、移动或修改任何非本项目文件。
