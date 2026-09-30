# RUN-0004 — G0 冒烟（三家族）：服务器全量测试 + Qwen3.5-0.8B / Gemma 4 E2B-it / InternVL3.5-2B-HF 真实权重冒烟 + Qwen3.5-4B

服务器 CC，b200，2026-09-30 09:08–09:14 UTC（值守接单）。只贴数字与事实，不做解读。本任务取代 RUN-0003 §3–4。

## 0. 一句话观察（只陈述，不解读）

- 服务器全量测试：`558 passed, 1 skipped, 23 warnings in 93.55s (0:01:33)`，rc=0，与任务书预期的 558 passed + 1 skipped 一致。唯一的跳过项是 cv2 未装。
- 四份真实权重冒烟全部 `all_ok=True`：Qwen3.5-0.8B、Gemma 4 E2B-it、InternVL3.5-2B-HF（含检查 9）和 Qwen3.5-4B，冒烟脚本的退出码都是 0。
- Qwen3.5-4B 的 lm_head 与嵌入**共享**（`tied_word_embeddings=True`，与 0.8B 相同）。
- 两个 Qwen 在 GPU 上的 GDN 走 fla 内核，`switched_to_torch=False`。

## 1. 前置校验

- `git pull --ff-only` 后 HEAD 为 `0d83485`；`b4b4810` 在 HEAD 历史中（`git merge-base --is-ancestor` 成立）。
- `pip install --dry-run -r requirements.txt`（venv）：没有任何 "Would install"，不会改动已装包（`pip_dryrun_requirements.txt`）。
- 本次没有修改任何仓库代码或 config。冒烟参数按脚本 argparse：`--dtype bf16`，`--model` 用注册表的键。
- 测试图：RUN-0003 在 §3 停下，当时没有下载这张图。本次按 URL `http://images.cocodataset.org/val2017/000000039769.jpg` 单张下载到 `$PROJ/tmp/000000039769.jpg`（173,131 B），sha256 `dea9e7ef…4563b04e`，与 RUN-0001 的副本相同（`test_image_sha256.txt`）。
- 本次的推送：`806f4c0`（批次 1：§3 全量测试）；批次 2（冒烟与本 report）为本 commit。

## 2. 执行记录

| 步骤 | 命令 | 墙钟 | 结果 |
|---|---|---|---|
| 依赖 dry-run | `pip install --dry-run -r requirements.txt` | 数秒 | 不改动任何包 |
| 全量测试 | `cd $REPO && python -m pytest tests/ -q` | 95.0 s（pytest 自报 93.55 s） | 558 passed, 1 skipped, 23 warnings；rc=0 |
| 跳过项原因 | `python -m pytest tests/test_metrics_seg.py -q -rs` | 0.2 s | `SKIPPED [1] tests/test_metrics_seg.py:129: could not import 'cv2'` |
| 冒烟 0.8B | `python scripts/smoke_adapter.py --model qwen3.5-0.8b --device cuda --dtype bf16 --image $PROJ/tmp/000000039769.jpg --out experiments/P0/0004_g0_smoke/smoke_qwen3.5-0.8b.json` | 10.8 s（脚本自报 8.64 s） | all_ok=True，rc=0 |
| 冒烟 Gemma | 同上，`--model gemma-4-e2b-it` | 15.7 s（13.67 s） | all_ok=True，rc=0 |
| 冒烟 InternVL | 同上，`--model internvl3.5-2b-hf --think-max-new-tokens 2048` | 8.9 s（6.94 s） | all_ok=True（含 9_multi_tile），rc=0 |
| 删除内存盘上的 Gemma | `scan_cache_dir('/dev/shm/visint/hf-hub').delete_revisions('3e22461f65e8…').execute()` | <1 s | 预计并实际释放 10,278,849,571 B（`shm_delete_gemma.txt`） |
| 下载 Qwen3.5-4B @851bf6e806ef | `snapshot_download(revision=…)` | 19.5 s | 9.33 GB → /dev/shm |
| 冒烟 4B | 同上，`--model qwen3.5-4b` | 13.4 s（11.15 s） | all_ok=True，rc=0 |
| 删除内存盘上的 4B | `delete_revisions('851bf6e806ef…').execute()` | <1 s | 释放 9,330,099,487 B（`shm_delete_4b.txt`） |

- GPU：NVIDIA B200；冒烟全部在 cuda:0 上以 bf16 运行（检查 8 的设备审计全部通过）。
- 警告（原样）：
  - 两次 Qwen 冒烟：`` `causal_conv1d_update` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. ``（causal-conv1d 未安装，与 RUN-0001 相同）；
  - InternVL 冒烟：`The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. ...`（与 RUN-0002 相同）；
  - Gemma 冒烟：没有警告；
  - pytest：23 条 warnings，原文见 `pytest.txt`。

## 3. 结果表（原样，由 `probe_scripts/p8_smoke_summary.py` 从四份 JSON 生成）

### 3.1 各检查项的 ok（原样）

| 检查 | qwen3.5-0.8b | gemma-4-e2b-it | internvl3.5-2b-hf | qwen3.5-4b |
|---|---|---|---|---|
| 0_facts | True | True | True | True |
| 1_build_generate | True | True | True | True |
| 2_task_modulation | True | True | True | True |
| 3_grid_tokens | True | True | True | True |
| 4_query_tokens | True | True | True | True |
| 5_position_ids | True | True | True | True |
| 6_vision_features | True | True | True | True |
| 7_thinking | True | True | True | True |
| 8_device_audit | True | True | True | True |
| 9_multi_tile | — | — | True | — |
| **all_ok** | **True** | **True** | **True** | **True** |
| wall_time_s | 8.64 | 13.67 | 6.94 | 11.15 |

### 3.2 检查 0：实际内核实现与 lm_head 共享

- **qwen3.5-0.8b**（Qwen/Qwen3.5-0.8B @ 2fc06364715b，cuda:0，torch.bfloat16，attn=sdpa）：lm_head 与嵌入共享 = **True**；嵌入 248320 行 / lm_head 248320 行 / len(tokenizer) 248077
  - kernels（模块级绑定）：`{"torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule", "causal_conv1d_fn": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn", "causal_conv1d_update": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update"}`
  - 适配器前向里实际用到的 GDN：`torch_chunk_gated_delta_rule` → `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`（is_torch_reference=False, switched_to_torch=False）；`torch_recurrent_gated_delta_rule` → `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule`（is_torch_reference=False, switched_to_torch=False）
  - 其他：`{"llm_grid_hw": [15, 20], "stop_token_ids": {"<\|im_end\|>": 248046, "<\|endoftext\|>": 248044}}`
- **gemma-4-e2b-it**（google/gemma-4-E2B-it @ 3e22461f65e8，cuda:0，torch.bfloat16，attn=sdpa）：lm_head 与嵌入共享 = **True**；嵌入 262144 行 / lm_head 262144 行 / len(tokenizer) 262144
  - 其他：`{"max_soft_tokens": 280, "image_grid_hw": [14, 19], "n_image_tokens": 266, "kv_shared_layers": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34], "sliding_window": 512, "n_unused_pool": 6227, "n_base_rows_for_mean": 255893, "stop_token_ids": {"<turn\|>": 106, "<eos>": 1}}`
- **internvl3.5-2b-hf**（OpenGVLab/InternVL3_5-2B-HF @ 3f301ffcf3dc，cuda:0，torch.bfloat16，attn={'top': 'sdpa', 'text': 'sdpa', 'vision': 'sdpa'}）：lm_head 与嵌入共享 = **False**（config 的 tie_word_embeddings = {"top": true, "text": false}）；嵌入 151936 行 / lm_head 151936 行 / len(tokenizer) 151679
  - 其他：`{"tile_policy": {"crop_to_patches": false, "max_patches": 1, "min_patches": 1, "tile_px": 448}, "tile_grid_hw": [16, 16], "tokens_per_tile": 256, "image_size_arg_ignored": {"value": [480, 640], "note": "--image-size 只对 Qwen 有效；InternVL 由图像处理器按 tile 策略缩放"}, "stop_token_ids": {"<\|im_end\|>": 151645}, "r1_system_prompt_source": "https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF/blob/3f301ffcf3dcbb47893afae6650ea3e78d96fb6d/README.md （§Thinking Mode，R1_SYSTEM_PROMPT）"}`
- **qwen3.5-4b**（Qwen/Qwen3.5-4B @ 851bf6e806ef，cuda:0，torch.bfloat16，attn=sdpa）：lm_head 与嵌入共享 = **True**；嵌入 248320 行 / lm_head 248320 行 / len(tokenizer) 248077
  - kernels（模块级绑定）：`{"torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule", "causal_conv1d_fn": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn", "causal_conv1d_update": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update"}`
  - 适配器前向里实际用到的 GDN：`torch_chunk_gated_delta_rule` → `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`（is_torch_reference=False, switched_to_torch=False）；`torch_recurrent_gated_delta_rule` → `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule`（is_torch_reference=False, switched_to_torch=False）
  - 其他：`{"llm_grid_hw": [15, 20], "stop_token_ids": {"<\|im_end\|>": 248046, "<\|endoftext\|>": 248044}}`

### 3.3 检查 1：与官方 processor 的逐项比对

| 模型 | 顺序 | 思考 | template_equal | 官方 processor 逐项 | 视觉 token | prompt token | thinking_unclosed | text（前 40 字） |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | instruction_first | False | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 318 | False | 'The image shows two cats lounging on' |
| qwen3.5-0.8b | instruction_first | True | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 316 | True | '' |
| qwen3.5-0.8b | image_first | False | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 318 | False | 'The image features two cats lying side by' |
| qwen3.5-0.8b | image_first | True | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 316 | True | '' |
| gemma-4-e2b-it | instruction_first | False | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, pixel_values=True, image_position_ids=True | 266 | 281 | False | 'This is a picture of a **tab' |
| gemma-4-e2b-it | instruction_first | True | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, pixel_values=True, image_position_ids=True | 266 | 288 | True | '' |
| gemma-4-e2b-it | image_first | False | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, pixel_values=True, image_position_ids=True | 266 | 281 | False | 'This image features **two cats** lying' |
| gemma-4-e2b-it | image_first | True | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, pixel_values=True, image_position_ids=True | 266 | 288 | True | '' |
| internvl3.5-2b-hf | instruction_first | False | True | ok=True, keys=True, input_ids=True, attention_mask=True, pixel_values=True | 256 | 271 | False | 'The image shows two tabby cats lying' |
| internvl3.5-2b-hf | instruction_first | True | True | ok=True, keys=True, input_ids=True, attention_mask=True, pixel_values=True | 256 | 391 | True | '' |
| internvl3.5-2b-hf | image_first | False | True | ok=True, keys=True, input_ids=True, attention_mask=True, pixel_values=True | 256 | 271 | False | 'The cat in the image is a tab' |
| internvl3.5-2b-hf | image_first | True | True | ok=True, keys=True, input_ids=True, attention_mask=True, pixel_values=True | 256 | 391 | True | '' |
| qwen3.5-4b | instruction_first | False | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 318 | False | 'This image shows two tabby cats sleeping' |
| qwen3.5-4b | instruction_first | True | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 316 | True | '' |
| qwen3.5-4b | image_first | False | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 318 | False | 'The image features **two tabby cats' |
| qwen3.5-4b | image_first | True | True | ok=True, input_ids=True, attention_mask=True, mm_token_type_ids=True, image_grid_thw=True, pixel_values=True | 300 | 316 | True | '' |

### 3.4 检查 2：任务调制仪器（经适配器 API）

- **qwen3.5-0.8b image_first**：逐位相同 25/25（图像 token 300，seq 311，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0 L2:0 L3:0 L4:0 L5:0 L6:0 L7:0 L8:0 L9:0 L10:0 L11:0 L12:0 L13:0 L14:0 L15:0 L16:0 L17:0 L18:0 L19:0 L20:0 L21:0 L22:0 L23:0 L24:0
- **qwen3.5-0.8b instruction_first**：逐位相同 1/25（图像 token 300，seq 311，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0.03125 L2:0.125 L3:0.1562 L4:0.1562 L5:0.2695 L6:0.207 L7:0.1606 L8:0.2617 L9:0.2322 L10:0.2432 L11:0.2297 L12:0.3032 L13:0.29 L14:0.3213 L15:0.3662 L16:0.4893 L17:0.5923 L18:0.6719 L19:0.8516 L20:1.07 L21:1.188 L22:1.25 L23:1.594 L24:20.25
- **gemma-4-e2b-it image_first**：逐位相同 36/36（图像 token 266，seq 278，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0 L2:0 L3:0 L4:0 L5:0 L6:0 L7:0 L8:0 L9:0 L10:0 L11:0 L12:0 L13:0 L14:0 L15:0 L16:0 L17:0 L18:0 L19:0 L20:0 L21:0 L22:0 L23:0 L24:0 L25:0 L26:0 L27:0 L28:0 L29:0 L30:0 L31:0 L32:0 L33:0 L34:0 L35:0
- **gemma-4-e2b-it instruction_first**：逐位相同 1/36（图像 token 266，seq 278，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0.875 L2:0.9795 L3:0.918 L4:1.5 L5:1.5 L6:2 L7:2 L8:2.41 L9:2.5 L10:5.75 L11:4.148 L12:5.164 L13:8 L14:3.396 L15:4.5 L16:9.5 L17:11.5 L18:10.5 L19:6.5 L20:4.875 L21:8.5 L22:9.75 L23:5.25 L24:5.73 L25:7.5 L26:9.75 L27:10.75 L28:12.5 L29:15.5 L30:17.5 L31:16 L32:13.5 L33:11.75 L34:7.75 L35:52
- **internvl3.5-2b-hf image_first**：逐位相同 29/29（图像 token 256，seq 268，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0 L2:0 L3:0 L4:0 L5:0 L6:0 L7:0 L8:0 L9:0 L10:0 L11:0 L12:0 L13:0 L14:0 L15:0 L16:0 L17:0 L18:0 L19:0 L20:0 L21:0 L22:0 L23:0 L24:0 L25:0 L26:0 L27:0 L28:0
- **internvl3.5-2b-hf instruction_first**：逐位相同 1/29（图像 token 256，seq 268，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0.1875 L2:0.25 L3:0.7891 L4:0.8223 L5:1 L6:1.25 L7:4.125 L8:6 L9:4.5 L10:7.75 L11:15.5 L12:12.75 L13:13 L14:12 L15:15.44 L16:19.43 L17:36 L18:40 L19:58 L20:83 L21:90 L22:96 L23:132 L24:220 L25:252 L26:260 L27:236 L28:36.25
- **qwen3.5-4b image_first**：逐位相同 33/33（图像 token 300，seq 311，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0 L2:0 L3:0 L4:0 L5:0 L6:0 L7:0 L8:0 L9:0 L10:0 L11:0 L12:0 L13:0 L14:0 L15:0 L16:0 L17:0 L18:0 L19:0 L20:0 L21:0 L22:0 L23:0 L24:0 L25:0 L26:0 L27:0 L28:0 L29:0 L30:0 L31:0 L32:0
- **qwen3.5-4b instruction_first**：逐位相同 1/33（图像 token 300，seq 311，指令 token 长度 [4, 4]）。逐层 max|Δ|：L0:0 L1:0.25 L2:0.25 L3:0.5 L4:0.5 L5:0.4375 L6:0.5 L7:0.5625 L8:0.8594 L9:1.297 L10:1.125 L11:0.625 L12:1.016 L13:1.109 L14:1.484 L15:1.5 L16:2.094 L17:2.734 L18:2.156 L19:3.312 L20:1.969 L21:3.359 L22:4.812 L23:4.438 L24:7.109 L25:8 L26:7.488 L27:8.719 L28:10.12 L29:11.73 L30:11.41 L31:10.59 L32:17.47

### 3.5 检查 3：grid token

| 模型 | ids | 单个 token | 行数 前→后 | 均值初始化（嵌入 / lm_head） | 生成全在允许集合内 | 生成的 ids |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | {'<g0>': 248077, '<g1>': 248078} | {'<g0>': True, '<g1>': True} | 248320→248320 | True / True | True | [248077, 248077, 248077, 248077, 248077, 248077, 248077, 248077] |
| gemma-4-e2b-it | {'<g0>': 6, '<g1>': 7} | {'<g0>': True, '<g1>': True} | None→[262144, 262144, 262144, 262144] | None / None | True | [6, 6, 6, 6, 6, 6, 6, 6] |
| ↳ 其他字段 | `{"surfaces": {"<g0>": "<unused0>", "<g1>": "<unused1>"}, "expected_lowest_unused_ids": [6, 7], "rows_before(embed,lm_head,ple,len_tok)": [262144, 262144, 262144, 262144], "mean_init_embed(tied_lm_head)": true, "mean_init_ple": true}` ||||||
| internvl3.5-2b-hf | {'<g0>': 151679, '<g1>': 151680} | {'<g0>': True, '<g1>': True} | 151936→151936 | True / True | True | [151679, 151679, 151679, 151679, 151679, 151679, 151679, 151679] |
| qwen3.5-4b | {'<g0>': 248077, '<g1>': 248078} | {'<g0>': True, '<g1>': True} | 248320→248320 | True / True | True | [248077, 248077, 248077, 248077, 248077, 248077, 248077, 248077] |

### 3.6 检查 7：思考段

- **qwen3.5-0.8b**：`{"closed": true, "closed_case_exercised": true, "thinking_unclosed": false, "n_generated_tokens": 144, "max_new_tokens": 512}`；text=`'pink'`；raw_text 开头=`'The user is asking for the main color of the image.\n1.  **Identify the dominant color:** The most striking and large color in the image is the bright pink fab`；关思考对照 `{"ok": true, "text": " pink", "thinking_unclosed": false}`
- **gemma-4-e2b-it**：`{"channel_opened": true, "closed": true, "closed_case_exercised": true, "thinking_unclosed": false, "n_generated_tokens": 202, "max_new_tokens": 512}`；text=`'Pink'`；raw_text 开头=`'<\|channel>thought\n1.  **Analyze the request:** The user wants to know the main color in the provided image and the answer must be in one word.\n\n2.  **Examin`；关思考对照 `{"ok": true, "text": "Pink", "thinking_unclosed": false, "channel_opened": false}`
- **internvl3.5-2b-hf**：`{"case": "no_think", "closed": false, "closed_case_exercised": false, "thinking_unclosed": false, "system_turn_in_prompt": true, "n_generated_tokens": 2, "max_new_tokens": 2048, "format_facts": {"first_token_is_think_start": false, "n_think_start": 0, "n_think_end": 0, "ended_with_stop_token": true}}`；text=`'Pink'`；raw_text 开头=`'Pink<\|im_end\|>'`；关思考对照 `{"ok": true, "case": "no_think", "text": "pink", "thinking_unclosed": false}`
- **qwen3.5-4b**：`{"closed": true, "closed_case_exercised": true, "thinking_unclosed": false, "n_generated_tokens": 147, "max_new_tokens": 512}`；text=`'Pink'`；raw_text 开头=`'The user wants to identify the main color in the image.\n1.  **Analyze the image:** I see two cats sleeping on a couch.\n2.  **Identify dominant colors:**\n   `；关思考对照 `{"ok": true, "text": " pink", "thinking_unclosed": false}`

### 3.7 检查 9（仅 InternVL）：多 tile

`{"ok": true, "policy": {"crop_to_patches": true, "max_patches": 6, "min_patches": 1, "tile_px": 448}, "image_size_wh": [640, 480], "n_tiles": 7, "n_tiles_from_image_processor": 7, "grid_hw": [112, 16], "span_len": 1792, "layout_ok": true, "processor_compare_by_thinking": {"False": {"ok": true, "keys": true, "input_ids": true, "attention_mask": true, "pixel_values": true}, "True": {"ok": true, "keys": true, "input_ids": true, "attention_mask": true, "pixel_values": true}}, "generate_ok": true, "n_generated_tokens": 8, "probe_methods_refuse_multi_tile": {"hidden_states": true, "vision_features": true}, "note": "带查询 token 的 build_inputs 拒绝多 tile 由单元测试覆盖（这里不加 token，保持共享的模型与 tokenizer 不变）", "wall_time_s": 0.638}`；adapter_path_vs_model_path：29/29 为 True



## 4. 与任务书假设不符的发现

1. **InternVL 检查 7 走的是 "no_think" 情形**：开启思考（加 R1 system prompt，prompt 里有 system 轮，`system_turn_in_prompt=True`）后，模型直接输出 `Pink<|im_end|>`，共 2 个 token，没有 `<think>`（`format_facts`: first_token_is_think_start=False, n_think_start=0, n_think_end=0, ended_with_stop_token=True）。因此 `closed_case_exercised=False`，即这次冒烟没有覆盖"思考段闭合"这一分支。检查仍判 ok。
2. InternVL 在冒烟里的 tile 策略（检查 0/1）为 `crop_to_patches=False, max_patches=1, min_patches=1`，即单 tile、256 token；检查 9 另用 `crop_to_patches=True, max_patches=6` 测多 tile，得到 7 个 tile、1792 token。冒烟 JSON 里的注记（原样）："--image-size 只对 Qwen 有效；InternVL 由图像处理器按 tile 策略缩放"。
3. Gemma 检查 3：grid token 复用 `<unused0>`/`<unused1>`（id 6、7），嵌入、lm_head、PLE 与 len(tokenizer) 的行数都不变（262144）。按 D13-1 初始化为均值（`mean_init_embed(tied_lm_head)=true`，`mean_init_ple=true`）。
4. 内存盘缓存里另有 `OpenGVLab/InternVL3_5-4B-HF` 的 43,006 B 小文件条目：`scan_cache_dir` 在删除 Gemma 之前就已列出它，推测是全量测试下载的 config/tokenizer 一类小文件，本任务没有主动下载它。保留未删。

## 5. 自己造成的错误或浪费

`p8_smoke_summary.py` 的第一版没有识别 InternVL 检查 1 的字段名（`processor_compare`，而另两家是 `official_processor`），表里那一列渲染成空。已在生成 report 前修正；冒烟本身没有重跑。

## 6. 生成文件清单与磁盘收支

**仓库内**：`experiments/P0/0004_g0_smoke/`：
- 四份冒烟 JSON
- `pytest.txt`、`pytest_skip_reason.txt`、`pip_dryrun_requirements.txt`
- `smoke_*.log`、`smoke_summary.md`
- `shm_delete_gemma.txt`、`shm_delete_4b.txt`、`test_image_sha256.txt`、`disk_ledger.tsv`、`README.md`

另有本 report 与 `probe_scripts/p8_smoke_summary.py`（副本）。

**仓库外**：`$PROJ/logs/0004/`（原件）；`$PROJ/tmp/000000039769.jpg`（173 KB）；`$PROJ/tmp/pytest-of-root/`（pytest 临时目录）。

**内存盘**（`/dev/shm/visint/hf-hub`，按 D9-5 可随时删）：现有 Qwen3.5-0.8B 1.770 GB、InternVL3.5-2B-HF 4.715 GB、InternVL3_5-4B-HF 小文件 43 KB。Gemma 4 E2B-it 按任务书已删除，RUN 需要时再下载，实测约 18 s。4B 用完已删除。本任务期间内存盘峰值 16.765 GB（上限 25 GB）。

**磁盘收支**（`disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 09:08:36 | 0004 start | 5.877 | 16.765 | 0.486 |
| 09:10:11 | after pytest | 5.875 | 16.765 | 0.487 |
| 09:10:43 | before smoke 0.8b | 5.875 | 16.765 | 0.488 |
| 09:11:03 | before smoke gemma | 5.874 | 16.765 | 0.488 |
| 09:11:19 | before smoke internvl | 5.873 | 16.765 | 0.488 |
| 09:11:28 | after smoke internvl | 5.871 | 16.765 | 0.488 |
| 09:11:40 | before delete gemma-4-E2B-it | 5.871 | 16.765 | 0.488 |
| 09:11:41 | after delete gemma-4-E2B-it | 5.871 | 6.486 | 0.488 |
| 09:12:00 | after download Qwen3.5-4B | 5.871 | 15.817 | 0.488 |
| 09:12:25 | after smoke 4B | 5.871 | 15.817 | 0.488 |
| 09:12:26 | after delete Qwen3.5-4B | 5.871 | 6.486 | 0.488 |
| 09:14:29 | before 0004 report | 5.870 | 6.486 | 0.488 |

- 根盘最低 5.870 GB（红线 3 GB）。`$PROJ` 从 0.486 GB 变为 0.488 GB。
- 没有删除、移动或修改任何非本项目文件。删除的只有内存盘上本项目的 Gemma 4 E2B-it 与 Qwen3.5-4B 缓存，均由任务书 §4 指定。
