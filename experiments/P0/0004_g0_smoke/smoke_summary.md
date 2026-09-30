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

