# model_notes — 候选模型登记与 Qwen3.5 事实探针（RUN-0001 §4、§5）

记录者：服务器 CC，2026-09-30。只记事实，不替项目选型。§4 只下载了 config / README / 模板等小文件，没有下载权重（Qwen3.5-0.8B、9B 的权重是 §5 下载的）。原始数据：`experiments/P0/0001_env_probe/raw/p4/registry.json`（每个仓库的完整摘要）以及各仓库的 config、README、chat template 副本。

## 1. 检索方法

- `HfApi().list_models(author=..., search=...)`：`Qwen` + "Qwen3.5" 返回 29 个，`google` + "gemma-4" 返回 63 个，`OpenGVLab` + "InternVL3_5" 返回 50 个。"InternVL3.5"（带点）返回 0 个，仓库名用下划线 `InternVL3_5`。
- 对每个候选仓库调用 `HfApi().model_info(repo, files_metadata=True)`，再按该 sha 下载小文件（config、README 等），以确认 token 能访问。
- token 身份：`whoami` = `JunliGong`（user，无 org）。
- 所有候选仓库**都不是 gated**，下载小文件全部成功（下表 "our token" 列为 ok）。

## 2. 登记表

"attention layout" 与 "heads" 两列取自 config 的 `text_config`（InternVL GitHub 格式的仓库取 `llm_config`）。safetensors GB 是仓库里全部 `.safetensors` 文件的字节数之和除以 1e9。

| repo | revision (sha) | license | gated | our token | safetensors GB | architectures | LLM layers | hidden | attention layout | heads | vision (patch, merge/pool) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | apache-2.0 | False | ok | 1.75 | Qwen3_5ForConditionalGeneration | 24 | 1024 | 18 linear_attention + 6 full_attention | 8Q/2KV×256; GDN 16K/16V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | apache-2.0 | False | ok | 4.55 | Qwen3_5ForConditionalGeneration | 24 | 2048 | 18 linear_attention + 6 full_attention | 8Q/2KV×256; GDN 16K/16V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | apache-2.0 | False | ok | 9.32 | Qwen3_5ForConditionalGeneration | 32 | 2560 | 24 linear_attention + 8 full_attention | 16Q/4KV×256; GDN 16K/32V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | apache-2.0 | False | ok | 19.31 | Qwen3_5ForConditionalGeneration | 32 | 4096 | 24 linear_attention + 8 full_attention | 16Q/4KV×256; GDN 16K/32V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | apache-2.0 | False | ok | 1.75 | Qwen3_5ForConditionalGeneration | 24 | 1024 | 18 linear_attention + 6 full_attention | 8Q/2KV×256; GDN 16K/16V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-2B-Base` | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | apache-2.0 | False | ok | 4.55 | Qwen3_5ForConditionalGeneration | 24 | 2048 | 18 linear_attention + 6 full_attention | 8Q/2KV×256; GDN 16K/16V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` | apache-2.0 | False | ok | 9.32 | Qwen3_5ForConditionalGeneration | 32 | 2560 | 24 linear_attention + 8 full_attention | 16Q/4KV×256; GDN 16K/32V heads ×128 | patch 16, merge 2 |
| `Qwen/Qwen3.5-9B-Base` | `68c46c4b3498877f3ef123c856ecfde50c39f404` | apache-2.0 | False | ok | 19.31 | Qwen3_5ForConditionalGeneration | 32 | 4096 | 24 linear_attention + 8 full_attention | 16Q/4KV×256; GDN 16K/32V heads ×128 | patch 16, merge 2 |
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | apache-2.0 | False | ok | 10.25 | Gemma4ForConditionalGeneration | 35 | 1536 | 28 sliding_attention + 7 full_attention | 8Q/1KV×256; sliding_window=512 | patch 16, pool 3, default 280 tok |
| `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | apache-2.0 | False | ok | 15.99 | Gemma4ForConditionalGeneration | 42 | 2560 | 35 sliding_attention + 7 full_attention | 8Q/2KV×256; sliding_window=512 | patch 16, pool 3, default 280 tok |
| `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | apache-2.0 | False | ok | 10.25 | Gemma4ForConditionalGeneration | 35 | 1536 | 28 sliding_attention + 7 full_attention | 8Q/1KV×256; sliding_window=512 | patch 16, pool 3, default 280 tok |
| `google/gemma-4-E4B` | `411aa17b749aa952df1359d2dcea73917a544d9a` | apache-2.0 | False | ok | 15.99 | Gemma4ForConditionalGeneration | 42 | 2560 | 35 sliding_attention + 7 full_attention | 8Q/2KV×256; sliding_window=512 | patch 16, pool 3, default 280 tok |
| `google/gemma-4-E2B-it-assistant` | `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df` | apache-2.0 | False | ok | 0.16 | Gemma4AssistantForCausalLM | 4 | 256 | 3 sliding_attention + 1 full_attention | 4Q/1KV×256; sliding_window=512 | — |
| `google/gemma-4-E4B-it-assistant` | `8d0031ea8c2109e2b1e86bb9368a4539b537f80a` | apache-2.0 | False | ok | 0.16 | Gemma4AssistantForCausalLM | 4 | 256 | 3 sliding_attention + 1 full_attention | 4Q/2KV×256; sliding_window=512 | — |
| `OpenGVLab/InternVL3_5-2B` | `7d7bd7bcc35b77b391883ebd7f9686cfa0e25cc0` | apache-2.0 | False | ok | 4.70 | InternVLChatModel | 28 | 2048 | all full_attention (no layer_types in config) | 16Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-2B-HF` | `3f301ffcf3dcbb47893afae6650ea3e78d96fb6d` | apache-2.0 | False | ok | 4.70 | InternVLForConditionalGeneration | 28 | 2048 | 28 full_attention | 16Q/8KV×128 | patch [14, 14], img [448, 448], downsample 0.5 |
| `OpenGVLab/InternVL3_5-2B-Instruct` | `a1e50c526faea805730148e7b1f2af18abe9e4b2` | apache-2.0 | False | ok | 4.70 | InternVLChatModel | 28 | 2048 | all full_attention (no layer_types in config) | 16Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-2B-MPO` | `bf5ffd82a66a10df7de9fe520239a837c8808798` | apache-2.0 | False | ok | 4.70 | InternVLChatModel | 28 | 2048 | all full_attention (no layer_types in config) | 16Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-2B-Pretrained` | `ce387f38cd21603625cdeb6c37f34a2bb89a771a` | apache-2.0 | False | ok | 4.70 | InternVLChatModel | 28 | 2048 | all full_attention (no layer_types in config) | 16Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-2B-Flash` | `43902d0577f0bb49386d60a0c31da4eb6e293462` | apache-2.0 | False | ok | 4.98 | InternVLChatModel | 28 | 2048 | all full_attention (no layer_types in config) | 16Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-4B` | `481f6e32467eab4e922ccd7fd6cf420441a62331` | apache-2.0 | False | ok | 9.47 | InternVLChatModel | 36 | 2560 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-4B-HF` | `6bd4487402110ef9889ba50eb7aefeb302526fed` | apache-2.0 | False | ok | 9.47 | InternVLForConditionalGeneration | 36 | 2560 | 36 full_attention | 32Q/8KV×128 | patch [14, 14], img [448, 448], downsample 0.5 |
| `OpenGVLab/InternVL3_5-4B-Instruct` | `a3fd3158be4027880d91ccee80ea8089559e0896` | apache-2.0 | False | ok | 9.47 | InternVLChatModel | 36 | 2560 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-4B-MPO` | `d8c8081570cabbd86984bd849617391f10e962e2` | apache-2.0 | False | ok | 9.47 | InternVLChatModel | 36 | 2560 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-4B-Pretrained` | `33f47a9c33c12798b6e583267108f2f782b9d0bd` | apache-2.0 | False | ok | 9.47 | InternVLChatModel | 36 | 2560 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-4B-Flash` | `bba25bb1015b0ff9e97e2562203217f3ea63bd77` | apache-2.0 | False | ok | 9.81 | InternVLChatModel | 36 | 2560 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-8B` | `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79` | apache-2.0 | False | ok | 17.06 | InternVLChatModel | 36 | 4096 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-8B-HF` | `741a7d03020411e666c6109218ab71e08151ef86` | apache-2.0 | False | ok | 17.06 | InternVLForConditionalGeneration | 36 | 4096 | 36 full_attention | 32Q/8KV×128 | patch [14, 14], img [448, 448], downsample 0.5 |
| `OpenGVLab/InternVL3_5-8B-Instruct` | `27506812aa9914804018996329e895977ee2d0c8` | apache-2.0 | False | ok | 17.06 | InternVLChatModel | 36 | 4096 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-8B-MPO` | `bb33f08275aa566f3442c1f5002512ffb66d7952` | apache-2.0 | False | ok | 17.06 | InternVLChatModel | 36 | 4096 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-8B-Pretrained` | `e21d3bab4089ed923c08072a1eefc8b5faf41cd8` | apache-2.0 | False | ok | 17.06 | InternVLChatModel | 36 | 4096 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |
| `OpenGVLab/InternVL3_5-8B-Flash` | `62246ac8390ab12eb2640385fc308289e8d07bf2` | apache-2.0 | False | ok | 17.63 | InternVLChatModel | 36 | 4096 | all full_attention (no layer_types in config) | 32Q/8KV×128 | patch 14, img 448, downsample 0.5 |

## 3. 各家族的补充事实

### 3.1 Qwen3.5

- **后训练版**（仓库名无后缀）：`Qwen/Qwen3.5-0.8B`、`Qwen/Qwen3.5-2B`、`Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-9B`，四个都存在。每个都有同名的 `-Base` 版本；model card 元数据里后训练版的 `base_model` 字段指向对应的 `-Base`。
- `-Base` 仓库**没有** `chat_template.jinja`。四个后训练版有 chat template，但**默认思考开关不同**：
  - 0.8B 与 2B 的模板：只有 `enable_thinking` 显式为 true 才开思考；未传参时渲染成空思考块 `<think>\n\n</think>\n\n`（即关闭）。model card 原文："**Qwen3.5-0.8B operates in non-thinking mode by default**."
  - 4B 与 9B 的模板：只有 `enable_thinking` 显式为 false 才关思考；未传参时渲染成 `<think>\n`（即开启）。model card 原文："Qwen3.5 models operate in thinking mode by default"。
  - 模板 diff 在 `raw/p4/` 的各仓库目录下可复现（0.8B 对 4B/9B 有差异，0.8B 对 2B 相同）。
- 仓库里没有 `generation_config.json`，因此 transformers 用 config 生成默认值（见 §5.2）。
- 9B 的 safetensors 含 15 个 `mtp.*` 张量（多 token 预测头）。按 `safetensors` 元数据共 9,653,104,368 参数，transformers 实际加载 9,409,813,744 参数，`mtp.*` 没有加载，且加载时无相关警告。
- model card 里的服务框架要求（原文）："SGLang from the main branch of the open-source repository is required for Qwen3.5"、"vLLM from the main branch of the open-source repository is required for Qwen3.5"、"The latest `transformers` is required for Qwen3.5"（示例装的是 transformers main）。实测 transformers 5.17.0 release 能加载，见 env_report §3。
- 许可证：全部 apache-2.0。

### 3.2 Gemma 4

- 指令版：`google/gemma-4-E2B-it`、`google/gemma-4-E4B-it`。另有预训练版 `google/gemma-4-E2B` / `E4B`（与 -it 同体积），以及 `-it-assistant`：model card 说明它是 "Multi-Token Prediction (MTP) drafters"，用于投机解码，4 层、hidden 256、约 78M 参数、无视觉塔。
- 参数与体积：E2B-it 共 5,123,178,051 参数，safetensors 10.25 GB；E4B-it 共 7,996,156,490 参数，15.99 GB。两者 config 都含 `audio_config`，`hidden_size_per_layer_input`=256。
- 注意力：E2B 35 层（28 sliding + 7 full，sliding_window=512，`num_kv_shared_layers`=20）；E4B 42 层（35 sliding + 7 full，`num_kv_shared_layers`=18）；全局层 `global_head_dim`=512。
- 视觉：patch 16，`pooling_kernel_size`=3；processor 默认 `max_soft_tokens`=280（`image_seq_length`=280）。model card 列出的可选视觉 token 预算："**70**, **140**, **280**, **560**, and **1120**"。
- 思考开关：chat template 有 `enable_thinking` 变量（`default(false)`）。model card 原文："Thinking is enabled by including the `<|think|>` token at the start of the system prompt"、"For all models except for the E2B and E4B variants, if thinking is disabled, the model will still generate the tags but with an empty thought block"。E2B 与 E4B 的模板逐字相同。
- 推荐采样（model card 原文）："`temperature=1.0`, `top_p=0.95`, `top_k=64`"。`generation_config.json` 为 `do_sample=true, temperature=1.0, top_k=64, top_p=0.95, eos_token_id=[1,106,50]`。
- model card 原文："place: Image content **before** the text in your prompt."
- 许可证：apache-2.0（card 元数据）。

### 3.3 InternVL3.5

- 每个尺寸在 OpenGVLab 下有 **6 个变体**。训练阶段依据 model card 表格：
  | 后缀 | 格式 | 训练阶段（card 原文） | architectures |
  |---|---|---|---|
  | （无后缀） | GitHub 格式，需 `trust_remote_code` | CPT + SFT + CascadeRL | InternVLChatModel |
  | `-HF` | HF 格式（card："aligned with the official Transformers standard"） | 与无后缀相同（最终版） | InternVLForConditionalGeneration |
  | `-Instruct` | GitHub 格式 | CPT + SFT | InternVLChatModel |
  | `-MPO` | GitHub 格式 | CPT + SFT + MPO | InternVLChatModel |
  | `-Pretrained` | GitHub 格式 | CPT | InternVLChatModel |
  | `-Flash` | GitHub 格式 | 另加 ViCO 阶段（card："reduces the token cost required to represent an image patch"） | InternVLChatModel |
- HF 格式只有最终版有（`-HF`）；没有找到 `-Instruct-HF`、`-MPO-HF` 等。
- `-Pretrained` 的 card 元数据 `base_model`：2B = `InternViT-300M-448px-V2_5` + `Qwen/Qwen3-1.7B`；4B = 同一 ViT + `Qwen/Qwen3-4B`；8B = 同一 ViT + `Qwen/Qwen3-8B`。
- **元数据异常（只记录）**：`OpenGVLab/InternVL3_5-2B-HF` 的 card 元数据 `base_model` 写的是 `OpenGVLab/InternVL3_5-4B-MPO`；4B-HF、8B-HF 写的是各自尺寸的 -MPO。
- 结构：LLM 为 qwen3（全注意力；2B 为 28 层 / hidden 2048 / 16Q-8KV，4B 为 36 / 2560 / 32Q-8KV，8B 为 36 / 4096 / 32Q-8KV），词表 151936；视觉为 InternViT-300M（patch 14、448 px、`downsample_ratio`=0.5，每个 448 tile 对应 `image_seq_length`=256 token）。HF processor 是 `GotOcr2ImageProcessorFast`，`min_patches`=1，`max_patches`=12，`crop_to_patches`=false（默认值）。
- 思考开关：用 system prompt 开启。card 原文："To enable thinking mode, please set the system prompt to our Thinking System Prompt. When enabling Thinking mode, we recommend setting `do_sample=True` and `temperature=0.6`"（`R1_SYSTEM_PROMPT`）。-HF 的 `generation_config.json` 没有采样参数（`_from_model_config`）。
- card 原文："Please use transformers>=4.52.1 to ensure the model works normally."
- 许可证：全部 apache-2.0（card 元数据）。

## 4. 找不到或有歧义的地方（不替项目选）

1. InternVL3.5 "2B/4B/8B" 各有 6 个变体。只有最终版（无后缀与 `-HF`）两种格式都有；是否用 `-HF`，以及用哪个训练阶段，留给项目决定。
2. Gemma 4 "E2B/E4B" 各有 -it、预训练版、-it-assistant（MTP drafter）三个仓库，指令版是 `-it`。
3. Qwen3.5 小尺寸（0.8B/2B）与大尺寸（4B/9B）的默认思考开关相反（§3.1）。简报 §6.3 写的是"Qwen3.5 默认开启思考"。
4. Qwen3.5 的 `-Base` 仓库没有 chat template。


---

## 5. Qwen3.5 事实探针（RUN-0001 §5；0.8B 与 9B，权重在 `/dev/shm/visint/hf-hub`）

运行环境见 env_report。所有生成都用 greedy（`do_sample=False`），并显式设置思考开关。脚本在 `experiments/P0/0001_env_probe/probe_scripts/`，原始 JSON 在 `raw/p5/`。

### 5.1 结构

| 项 | Qwen3.5-0.8B | Qwen3.5-9B |
|---|---|---|
| 加载类 | Qwen3_5ForConditionalGeneration | Qwen3_5ForConditionalGeneration |
| processor / image processor / tokenizer | Qwen3VLProcessor / Qwen2VLImageProcessor / Qwen2Tokenizer | Qwen3VLProcessor / Qwen2VLImageProcessor / Qwen2Tokenizer |
| LLM 层数 | 24 | 32 |
| 层类型计数 | full_attention=6, linear_attention=18 | full_attention=8, linear_attention=24 |
| 全注意力层下标（0 起） | [3, 7, 11, 15, 19, 23] | [3, 7, 11, 15, 19, 23, 27, 31] |
| LLM hidden size | 1024 | 4096 |
| 全注意力：Q 头 / KV 头 / head_dim / 输出门 | 8 / 2 / 256 / True | 16 / 4 / 256 / True |
| Gated DeltaNet：K 头 / V 头 / k_dim / v_dim / conv kernel | 16 / 16 / 128 / 128 / 4 | 16 / 32 / 128 / 128 / 4 |
| 视觉编码器 depth / hidden / heads | 12 / 768 / 12 | 27 / 1152 / 16 |
| patch_size / spatial_merge_size / temporal_patch_size | 16 / 2 / 2 | 16 / 2 / 2 |
| 每个 LLM 视觉 token 对应（缩放后图像） | 32×32 px | 32×32 px |
| deepstack_visual_indexes | [] | [] |
| 参数量：总 / 视觉 / 语言模型（加载后） | 852,985,920 / 100,592,896 / 752,393,024 | 9,409,813,744 / 456,010,480 / 7,936,684,544 |
| config vocab_size / 嵌入行数 / lm_head 行数 | 248320 / 248320 / 248320 | 248320 / 248320 / 248320 |
| len(tokenizer) / 基础词表（不含 added） | 248077 / 248044 | 248077 / 248044 |
| 嵌入行数 − len(tokenizer)（超出 tokenizer 的行） | 243 | 243 |
| tie_word_embeddings / lm_head 与嵌入同一存储 | True / True | False / False |
| 嵌入：超出 tokenizer 的 243 行 | 范数 0.4214–0.4215（均值 0.4215），全零行 0，与首行最大差 0.0004883 | 范数 0.1941–0.2045（均值 0.1991），全零行 0，与首行最大差 0.02087 |
| 嵌入：基础词表随机 2 万行 | 范数 0.3765–0.9469（均值 0.6269），全零行 0，与首行最大差 0.2175 | 范数 0.0144–1.0485（均值 0.8524），全零行 0，与首行最大差 0.404 |
| lm_head：超出 tokenizer 的 243 行 | （与嵌入共享） | 范数 0.4144–0.4204（均值 0.4175），全零行 0，与首行最大差 0.01706 |
| lm_head：基础词表随机 2 万行 | （与嵌入共享） | 范数 0.3793–1.4064（均值 0.9832），全零行 0，与首行最大差 0.2682 |

- 层类型为周期排列（`full_attention_interval`=4）：每 4 层中前 3 层是 Gated DeltaNet（`linear_attention`），第 4 层是门控全注意力。
- 位置编码参数（config）：`mrope_section`=[11, 11, 10]，`mrope_interleaved`=true，`partial_rotary_factor`=0.25，`rope_theta`=1e7。

**tokenizer 里的 added tokens**（33 个，id 248044–248076；"出现在 chat template 里"指 token 字符串是否出现在 `chat_template.jinja` 中；base vocab 里没有 `<|reserved...|>` 或 `<unused..>` 这类占位 token）：

| id | token | special | 出现在 chat template 里 | 0.8B 嵌入范数 | 9B 嵌入范数 | 9B lm_head 范数 |
|---|---|---|---|---|---|---|
| 248044 | `<\|endoftext\|>` | True | False | 0.8618 | 0.9128 | 0.8361 |
| 248045 | `<\|im_start\|>` | True | True | 0.5278 | 0.7847 | 0.9987 |
| 248046 | `<\|im_end\|>` | True | True | 0.6244 | 0.7836 | 0.8365 |
| 248047 | `<\|object_ref_start\|>` | True | False | 0.6287 | 0.8694 | 0.8685 |
| 248048 | `<\|object_ref_end\|>` | True | False | 0.5676 | 0.8512 | 0.7791 |
| 248049 | `<\|box_start\|>` | True | False | 0.4302 | 1.0276 | 0.5760 |
| 248050 | `<\|box_end\|>` | True | False | 0.5208 | 0.8936 | 1.5449 |
| 248051 | `<\|quad_start\|>` | True | False | 0.5091 | 0.8697 | 0.9915 |
| 248052 | `<\|quad_end\|>` | True | False | 0.5282 | 0.8682 | 1.1075 |
| 248053 | `<\|vision_start\|>` | True | True | 0.3466 | 0.8758 | 0.4338 |
| 248054 | `<\|vision_end\|>` | True | True | 0.3601 | 0.7706 | 0.4229 |
| 248055 | `<\|vision_pad\|>` | True | False | 0.4208 | 0.1996 | 0.4153 |
| 248056 | `<\|image_pad\|>` | True | True | 0.4204 | 0.2032 | 0.4265 |
| 248057 | `<\|video_pad\|>` | True | True | 0.4201 | 0.1981 | 0.4176 |
| 248058 | `<tool_call>` | False | True | 0.4965 | 0.8686 | 0.8327 |
| 248059 | `</tool_call>` | False | True | 0.5220 | 0.8613 | 0.9900 |
| 248060 | `<\|fim_prefix\|>` | False | False | 0.4796 | 0.8492 | 1.2012 |
| 248061 | `<\|fim_middle\|>` | False | False | 0.5888 | 0.7879 | 0.9107 |
| 248062 | `<\|fim_suffix\|>` | False | False | 0.5635 | 0.9226 | 1.0588 |
| 248063 | `<\|fim_pad\|>` | False | False | 0.4203 | 0.2012 | 0.4160 |
| 248064 | `<\|repo_name\|>` | False | False | 0.5623 | 0.8635 | 1.2787 |
| 248065 | `<\|file_sep\|>` | False | False | 0.5819 | 0.8762 | 0.7518 |
| 248066 | `<tool_response>` | False | True | 0.4725 | 0.8412 | 1.2059 |
| 248067 | `</tool_response>` | False | True | 0.5033 | 0.8616 | 0.7544 |
| 248068 | `<think>` | False | True | 0.5486 | 0.7957 | 0.9017 |
| 248069 | `</think>` | False | True | 0.5706 | 0.7887 | 1.1445 |
| 248070 | `<\|audio_start\|>` | True | False | 0.4215 | 0.2000 | 0.4181 |
| 248071 | `<\|audio_end\|>` | True | False | 0.4215 | 0.2029 | 0.4193 |
| 248072 | `<tts_pad>` | True | False | 0.4215 | 0.2003 | 0.4182 |
| 248073 | `<tts_text_bos>` | True | False | 0.4214 | 0.1987 | 0.4160 |
| 248074 | `<tts_text_eod>` | True | False | 0.4215 | 0.1958 | 0.4173 |
| 248075 | `<tts_text_bos_single>` | True | False | 0.4215 | 0.1993 | 0.4186 |
| 248076 | `<\|audio_pad\|>` | True | False | 0.4215 | 0.1962 | 0.4179 |

- 事实：在 0.8B 上，`<|audio_*|>`、`<tts_*>` 这 7 个 token 的嵌入范数为 0.4214–0.4215，与超出 tokenizer 的 243 行（0.4214–0.4215）处在同一区间；9B 上，这 7 个 token 以及 `<|vision_pad|>`、`<|image_pad|>`、`<|video_pad|>`、`<|fim_pad|>` 的嵌入范数为 0.196–0.203，与那 243 行（0.194–0.204）处在同一区间。只陈述，不据此判断哪些行"训练过"。

**新增 token `<g0>`：**

- **qwen3.5-0.8b**：`add_tokens(['<g0>'])` 返回 1，`add_tokens(['<g1>'], special_tokens=True)` 返回 1；id(`<g0>`)=248077，id(`<g1>`)=248078，len(tokenizer) 248077→248079；新 id < 嵌入行数：True；id 248077 这一行原有嵌入的范数 0.4215。`resize_token_embeddings(len(tok))` 后行数：248320→248079（lm_head 248079）。
- **qwen3.5-9b**：`add_tokens(['<g0>'])` 返回 1，`add_tokens(['<g1>'], special_tokens=True)` 返回 1；id(`<g0>`)=248077，id(`<g1>`)=248078，len(tokenizer) 248077→248079；新 id < 嵌入行数：True；id 248077 这一行原有嵌入的范数 0.1953。`resize_token_embeddings(len(tok))` 后行数：248320→248079（lm_head 248079）。

分词结果（0.8B 与 9B 一致）：

| 输入 | ids | tokens |
|---|---|---|
| `'<g0>'` | [248077] | ['<g0>'] |
| `'a<g0>b'` | [64, 248077, 65] | ['a', '<g0>', 'b'] |
| `' <g0>'` | [220, 248077] | ['Ġ', '<g0>'] |
| `'<g0><g0><g0>'` | [248077, 248077, 248077] | ['<g0>', '<g0>', '<g0>'] |
| `'<g0>\n<g1>'` | [248077, 198, 248078] | ['<g0>', 'Ċ', '<g1>'] |
| `'x <g1> y'` | [87, 220, 248078, 374] | ['x', 'Ġ', '<g1>', 'Ġy'] |
| `'<g1><g0>'` | [248078, 248077] | ['<g1>', '<g0>'] |
| `'12<g0>34'` | [16, 17, 248077, 18, 19] | ['1', '2', '<g0>', '3', '4'] |

- 新 id 从 248077 起，落在已有的 243 个嵌入行之内（248077–248319），不 resize 就能直接索引到现有行。
- **`model.resize_token_embeddings(len(tokenizer))` 会把嵌入从 248320 行缩到 len(tokenizer)**（实测 248320→248079）。

### 5.2 聊天模板与思考开关

- 开关传法：直接作为 `apply_chat_template(..., enable_thinking=False)` 的关键字参数（processor 与 tokenizer 都支持）。transformers 用 Jinja 自省模板变量，把同名 kwarg 传进模板。
- `chat_template_kwargs={"enable_thinking": False}` 是 vLLM/SGLang OpenAI API 的字段（见 model card）。**传给 transformers 的 `apply_chat_template` 不生效**：在 9B（默认开启思考）上，processor 版照样渲染 `<think>\n`（开启），并打印 `Kwargs passed to processor.__call__ have to be in processor_kwargs dict`；tokenizer 版不报警也不生效。0.8B 默认就是关闭，所以两种写法的输出看起来相同。

**qwen3.5-0.8b**

| 调用方式 | 生成提示结尾（原样） | 备注 |
|---|---|---|
| processor · default(no kwarg) · text | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |
| tokenizer · default(no kwarg) · text(str content) | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |
| processor · enable_thinking=False · text | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |
| tokenizer · enable_thinking=False · text(str content) | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |
| processor · enable_thinking=True · text | `'<\|im_start\|>assistant\n<think>\n'` |  |
| tokenizer · enable_thinking=True · text(str content) | `'<\|im_start\|>assistant\n<think>\n'` |  |
| processor · chat_template_kwargs={enable_thinking:False} · text | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` | 日志：Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| tokenizer · chat_template_kwargs={enable_thinking:False} · text(str content) | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |

**qwen3.5-9b**

| 调用方式 | 生成提示结尾（原样） | 备注 |
|---|---|---|
| processor · default(no kwarg) · text | `'<\|im_start\|>assistant\n<think>\n'` |  |
| tokenizer · default(no kwarg) · text(str content) | `'<\|im_start\|>assistant\n<think>\n'` |  |
| processor · enable_thinking=False · text | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |
| tokenizer · enable_thinking=False · text(str content) | `'<\|im_start\|>assistant\n<think>\n\n</think>\n\n'` |  |
| processor · enable_thinking=True · text | `'<\|im_start\|>assistant\n<think>\n'` |  |
| tokenizer · enable_thinking=True · text(str content) | `'<\|im_start\|>assistant\n<think>\n'` |  |
| processor · chat_template_kwargs={enable_thinking:False} · text | `'<\|im_start\|>assistant\n<think>\n'` | 日志：Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| tokenizer · chat_template_kwargs={enable_thinking:False} · text(str content) | `'<\|im_start\|>assistant\n<think>\n'` |  |


**关闭思考（`enable_thinking=False`）时，带图消息渲染出的完整文本（原样，0.8B 与 9B 相同：True）：**

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>
<|im_start|>assistant
<think>

</think>


```

- `generation_config`：两个仓库都没有 `generation_config.json`（`GenerationConfig.from_pretrained` 报 OSError），transformers 从 config 生成，非 None 字段只有：
  - 0.8B：`{"use_cache": true, "eos_token_id": 248044, "_from_model_config": true, "transformers_version": "5.17.0"}`
  - 9B：`{"use_cache": true, "eos_token_id": 248044, "_from_model_config": true, "transformers_version": "5.17.0"}`
  
  即默认 greedy，`eos_token_id`=248044（`<|endoftext|>`），不含 `<|im_end|>`=248046。§5.5 的实测里，模型先输出 `<|im_end|>`、`\n`，再输出 `<|endoftext|>` 才停。
- model card 推荐采样参数（原文）：
  - 0.8B（[card](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md)；2B 的 card 文字相同）：
    > - Non-thinking mode for text tasks: `temperature=1.0, top_p=1.00, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0`
    > - Non-thinking mode for VL tasks: `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
    > - Thinking mode for text tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
    > - Thinking mode for VL or precise coding (e.g. WebDev) tasks : `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`

    同一 card 另有一句："In thinking mode, we have observed that when using the recommended sampling parameters, Qwen3.5-0.8B is more prone to entering thinking loops compared to other Qwen3.5 models, which may prevent it from terminating generation properly."
  - 9B（[card](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)；4B 的 card 文字相同），Quickstart 的 Tip 框：
    > - Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
    > - Thinking mode for precise coding tasks (e.g. WebDev): `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`
    > - Instruct (or non-thinking) mode for general tasks: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
    > - Instruct (or non-thinking) mode for reasoning tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`

    **同一 card 的 Best Practices 一节里，"Instruct (or non-thinking) mode for reasoning tasks" 写的是另一组：** `temperature=1.0`, `top_p=1.0`, `top_k=40`, `min_p=0.0`, `presence_penalty=2.0`, `repetition_penalty=1.0`。其余三组两处一致。

### 5.3 图像 token 化

- 处理器：`Qwen2VLImageProcessor`，默认 `size={"shortest_edge": 65536, "longest_edge": 16777216}`。这两个值是**像素总数**的上下界（256² 与 4096²），不是边长。图像先按 smart resize 缩放，使高、宽都是 32（patch 16 × merge 2）的倍数，每个 LLM 视觉 token 对应缩放后图像的 32×32 像素。
- **固定分辨率的参数名**：在 processor 调用时传 `size={"shortest_edge": A, "longest_edge": A}`，或传 `min_pixels=A, max_pixels=A`，两者结果相同（见下表）。二者只固定像素总数；宽高比不同的图，网格仍不同。例如 A=1024² 时，4:3 的图得到 28×37=1036 个 token。要得到确定的 (h, w) 网格，可以先把图缩放到 32 的倍数，并落在默认上下界之内（例如 448×448 得到 14×14）。
- 0.8B 与 9B 结果逐项相同（表中若有不同会标注）。输入为合成 RGB 图：

| 输入（W×H）与参数 | image_grid_thw（patch 网格 t,h,w） | LLM 网格 (h, w) | 视觉 token 数 | 缩放后像素 (h, w) | 每 token 像素（缩放后） | 每 token 像素（原图） |
|---|---|---|---|---|---|---|
| 448x448 default | [1, 28, 28] | (14, 14) | 196 | (448, 448) | 1024 | 1024.0 |
| 640x480 default | [1, 30, 40] | (15, 20) | 300 | (480, 640) | 1024 | 1024.0 |
| 1024x768 default | [1, 48, 64] | (24, 32) | 768 | (768, 1024) | 1024 | 1024.0 |
| 1024x1024 default | [1, 64, 64] | (32, 32) | 1024 | (1024, 1024) | 1024 | 1024.0 |
| 640x480 size=(min=max=200704) | [1, 24, 32] | (12, 16) | 192 | (384, 512) | 1024 | 1600.0 |
| 640x480 min_pixels=max_pixels=200704 | [1, 24, 32] | (12, 16) | 192 | (384, 512) | 1024 | 1600.0 |
| 640x480 size=(min=max=1048576) | [1, 56, 74] | (28, 37) | 1036 | (896, 1184) | 1024 | 296.5 |
| 640x480 min_pixels=max_pixels=1048576 | [1, 56, 74] | (28, 37) | 1036 | (896, 1184) | 1024 | 296.5 |
| 1024x768 size=(min=max=200704) | [1, 24, 32] | (12, 16) | 192 | (384, 512) | 1024 | 4096.0 |
| 1024x768 min_pixels=max_pixels=200704 | [1, 24, 32] | (12, 16) | 192 | (384, 512) | 1024 | 4096.0 |
| 1024x768 size=(min=max=1048576) | [1, 56, 74] | (28, 37) | 1036 | (896, 1184) | 1024 | 759.1 |
| 1024x768 min_pixels=max_pixels=1048576 | [1, 56, 74] | (28, 37) | 1036 | (896, 1184) | 1024 | 759.1 |
| 500x333 default | [1, 20, 32] | (10, 16) | 160 | (320, 512) | 1024 | 1040.6 |

- **两张图同时输入**（448×448 与 640×480，`[image, image, text]`）：模板渲染为 `'<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Compare.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`。token 序列（游程编码）：`'<|im_start|>' 'user' 'Ċ' '<|vision_start|>' '<|image_pad|>'x196 '<|vision_end|>' '<|vision_start|>' '<|image_pad|>'x300 '<|vision_end|>' 'Compare' '.' '<|im_end|>' 'Ċ' '<|im_start|>' 'assistant' 'Ċ' '<think>' 'ĊĊ' '</think>' 'ĊĊ'`；`mm_token_type_ids`（0 文本 / 1 图像）：`0x4 1x196 0x2 1x300 0x12`。每张图各有一对 `<|vision_start|>`…`<|vision_end|>`，中间是 N 个 `<|image_pad|>`，两张图按出现顺序直接相连。processor 输出的键：`['input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values', 'image_grid_thw']`。若传 `add_vision_id=True`，模板会在每张图前加 "Picture k: "：`'<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: <|vision_start|><|image_pad|><|vision_end|>Compare.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`。

### 5.4 位置编码（M-RoPE）

- `model.model.get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask)` 返回 `(3, B, L)` 的 (T, H, W) 位置，以及 `rope_deltas`。`generate()` 内部把它和一行 0..L-1 的文本位置拼成 `(4, B, L)`，传给文本模型：第 0 行作 `text_position_ids`，第 1–3 行进 M-RoPE。
- 规则（transformers 5.17 源码与实测一致）：文本 token 的三个维度都等于顺序索引；图像 token 的 T 取图像起点 s，H = s + 行号，W = s + 列号（LLM 网格，即 patch 网格 ÷2）；图像之后的文本从 s + max(h, w) 接着编号（不是 s + h·w）。
- 模型把 `rope_deltas` 缓存在 `model.model.rope_deltas` 上，在增量解码时使用。
- 最小例子："指令 + 图（448×320 合成图，LLM 网格 10×14）+ 文本"，模板文本 `'<|im_start|>user\nFind the cat.<|vision_start|><|image_pad|><|vision_end|>Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`：

**qwen3.5-0.8b**：`get_rope_index` 输出形状 [3, 1, 161]，rope_deltas=[[-126]]；generate 内部构造的 position_ids 形状 [4, 1, 161]（第 0 行是 0..L-1 的文本位置：True；第 1–3 行等于 get_rope_index：True）。图像 grid_thw=[[1, 20, 28]]，seq_len=161。

| 段 | token 下标 | 个数 | T 范围 | H 范围 | W 范围 | 前 5 个 (T,H,W) | 后 3 个 (T,H,W) |
|---|---|---|---|---|---|---|---|
| text | 0–7 | 8 | 0–7 | 0–7 | 0–7 | [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]] | [[5, 5, 5], [6, 6, 6], [7, 7, 7]] |
| image | 8–147 | 140 | 8–8 | 8–17 | 8–21 | [[8, 8, 8], [8, 8, 9], [8, 8, 10], [8, 8, 11], [8, 8, 12]] | [[8, 17, 19], [8, 17, 20], [8, 17, 21]] |
| text | 148–160 | 13 | 22–34 | 22–34 | 22–34 | [[22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25], [26, 26, 26]] | [[32, 32, 32], [33, 33, 33], [34, 34, 34]] |

**qwen3.5-9b**：`get_rope_index` 输出形状 [3, 1, 161]，rope_deltas=[[-126]]；generate 内部构造的 position_ids 形状 [4, 1, 161]（第 0 行是 0..L-1 的文本位置：True；第 1–3 行等于 get_rope_index：True）。图像 grid_thw=[[1, 20, 28]]，seq_len=161。

| 段 | token 下标 | 个数 | T 范围 | H 范围 | W 范围 | 前 5 个 (T,H,W) | 后 3 个 (T,H,W) |
|---|---|---|---|---|---|---|---|
| text | 0–7 | 8 | 0–7 | 0–7 | 0–7 | [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]] | [[5, 5, 5], [6, 6, 6], [7, 7, 7]] |
| image | 8–147 | 140 | 8–8 | 8–17 | 8–21 | [[8, 8, 8], [8, 8, 9], [8, 8, 10], [8, 8, 11], [8, 8, 12]] | [[8, 17, 19], [8, 17, 20], [8, 17, 21]] |
| text | 148–160 | 13 | 22–34 | 22–34 | 22–34 | [[22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25], [26, 26, 26]] | [[32, 32, 32], [33, 33, 33], [34, 34, 34]] |


### 5.5 原生坐标格式

**文档的说法：**
- Qwen3.5 的 model card（0.8B、2B、4B、9B、27B、397B-A17B）里**找不到**关于定位输出坐标格式的说明（检索了 ground / bbox / coordinate / point / 1000 / relative / absolute 等关键词）。
- Qwen3.5 的 GitHub 仓库 `QwenLM/Qwen3.5` 现在重定向到 `QwenLM/Qwen3.8`（README 自述 "the GitHub repository of the Qwen3.5 open model series"）。README 里没有坐标格式的说明。
- 官方博客 https://qwen.ai/blog?id=qwen3.5 是 JS 渲染的页面，抓取只得到标题 "Qwen"，**未能核实**。
- 前代 Qwen3-VL 的官方 cookbook（[2d_grounding.ipynb](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)，与 Qwen3.5 不是同一个模型）原文："Coordinate System: Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000. (You don't need to calculate the resized_w)"。

**零样本实测**（每张图一次，只用一个固定提示，没有调提示词）：提示 `Locate {obj} in the image. Report bbox coordinates in JSON format.`（措辞沿用上面的 Qwen3-VL cookbook）。greedy，`enable_thinking=False`，`max_new_tokens=256`。原始输出（含特殊 token）：

- **qwen3.5-0.8b** `000000039769.jpg`（640×480，模型输入缩放为 h×w=480×640），目标 "the cat on the left"，生成 47 token：`'```json\n[\n\t{"bbox_2d": [14, 114, 492, 988], "label": "cat on the left"}\n]\n```<|im_end|>\n<|endoftext|>'`
- **qwen3.5-0.8b** `000000000139.jpg`（640×426，模型输入缩放为 h×w=416×640），目标 "the television"，生成 45 token：`'```json\n[\n\t{"bbox_2d": [10, 391, 242, 610], "label": "television"}\n]\n```<|im_end|>\n<|endoftext|>'`
- **qwen3.5-9b** `000000039769.jpg`（640×480，模型输入缩放为 h×w=480×640），目标 "the cat on the left"，生成 47 token：`'```json\n[\n\t{"bbox_2d": [11, 119, 494, 989], "label": "cat on the left"}\n]\n```<|im_end|>\n<|endoftext|>'`
- **qwen3.5-9b** `000000000139.jpg`（640×426，模型输入缩放为 h×w=416×640），目标 "the television"，生成 45 token：`'```json\n[\n\t{"bbox_2d": [10, 388, 241, 624], "label": "television"}\n]\n```<|im_end|>\n<|endoftext|>'`

COCO val2017 GT（取自官方 `instances_val2017.json`，经 HTTP Range 从 `annotations_trainval2017.zip` 中单独抽取），以及按 ×1000/(W,H) 换算后的值，只列与目标同类的物体：

| 图 | 物体 | GT xyxy（像素） | GT xyxy ×1000/(W,H) |
|---|---|---|---|
| 000000039769.jpg (640×480) | cat | [17.5, 54.4, 318.8, 469.4] | [27, 113, 498, 978] |
| 000000039769.jpg (640×480) | cat | [347.5, 25.6, 640.0, 369.4] | [543, 53, 1000, 770] |
| 000000000139.jpg (640×426) | tv | [7.0, 167.8, 156.3, 262.6] | [11, 394, 244, 617] |
| 000000000139.jpg (640×426) | tv | [557.2, 209.2, 638.6, 287.9] | [871, 491, 998, 676] |

### 5.6 Gated DeltaNet 数值

**5.6.1 内核对比**。同一组随机输入分别跑 6 种实现：
- `fla_chunk`：`fla.ops.gated_delta_rule.chunk_gated_delta_rule`（Triton）；
- `fla_fused_rec`：`fused_recurrent_gated_delta_rule`（Triton）；
- `fla_naive_rec`、`fla_naive_chunk`：fla 的 torch 参考实现（内部转成 fp32）；
- `hf_torch_chunk`、`hf_torch_rec`：transformers `modeling_qwen3_5` 自带的 torch 实现（`torch_chunk_gated_delta_rule.__wrapped__` 与 `torch_recurrent_gated_delta_rule.__wrapped__`）；
- 另加 `fp64_ref`：本地写的 float64 逐 token 递推，作为基准。

"prenorm_l2" 表示 q、k 在外部先做 L2 归一化，各实现的内核内归一化都关闭；"inkernel_l2" 表示 `use_qk_l2norm_in_kernel=True`（与模型调用方式相同，只有 fla 的两个 Triton 内核和 transformers 的两个实现有这个参数）。g = logsigmoid(randn)，beta = sigmoid(rand)。

形状 B=2, T=300（chunk=64，共 5 个 chunk，最后一个不满），H=16, K=V=128；fla 0.5.2。

**float32|prenorm_l2**（输出 dtype：fla_chunk=float32, fla_fused_rec=float32, fla_naive_rec=float32, fla_naive_chunk=float32, hf_torch_chunk=float32, hf_torch_rec=float32）

| 对比 | o max\|Δ\| | o mean\|Δ\| | o ‖Δ‖/‖ref‖ | o max rel | 末状态 S max\|Δ\| |
|---|---|---|---|---|---|
| fla_chunk vs fla_fused_rec | 1.135e-04 | 5.834e-06 | 1.472e-03 | 9.601e+00 | 6.937e-04 |
| fla_chunk vs fla_naive_rec | 1.135e-04 | 5.834e-06 | 1.472e-03 | 9.601e+00 | 6.937e-04 |
| fla_chunk vs fla_naive_chunk | 1.135e-04 | 5.834e-06 | 1.472e-03 | 9.661e+00 | 6.937e-04 |
| fla_chunk vs hf_torch_chunk | 1.135e-04 | 5.834e-06 | 1.472e-03 | 9.661e+00 | 6.937e-04 |
| fla_chunk vs hf_torch_rec | 1.135e-04 | 5.834e-06 | 1.472e-03 | 9.601e+00 | 6.937e-04 |
| fla_fused_rec vs fla_naive_rec | 1.118e-08 | 6.351e-10 | 1.651e-07 | 1.510e-03 | 5.960e-08 |
| fla_fused_rec vs fla_naive_chunk | 1.043e-07 | 2.041e-09 | 6.344e-07 | 1.404e-02 | 8.941e-07 |
| fla_fused_rec vs hf_torch_chunk | 1.043e-07 | 2.053e-09 | 6.364e-07 | 1.404e-02 | 8.941e-07 |
| fla_fused_rec vs hf_torch_rec | 1.118e-08 | 5.783e-10 | 1.516e-07 | 2.724e-03 | 5.960e-08 |
| fla_naive_rec vs fla_naive_chunk | 1.043e-07 | 2.043e-09 | 6.349e-07 | 1.404e-02 | 8.941e-07 |
| fla_naive_rec vs hf_torch_chunk | 1.043e-07 | 2.054e-09 | 6.368e-07 | 1.404e-02 | 8.941e-07 |
| fla_naive_rec vs hf_torch_rec | 1.118e-08 | 5.132e-10 | 1.371e-07 | 1.566e-03 | 0.000e+00 |
| fla_naive_chunk vs hf_torch_chunk | 7.451e-09 | 2.002e-10 | 7.094e-08 | 3.757e-04 | 1.192e-07 |
| fla_naive_chunk vs hf_torch_rec | 1.043e-07 | 2.014e-09 | 6.309e-07 | 1.344e-02 | 8.941e-07 |
| hf_torch_chunk vs hf_torch_rec | 1.043e-07 | 2.026e-09 | 6.328e-07 | 1.344e-02 | 8.941e-07 |
| fla_chunk vs fp64_ref | 1.135e-04 | 5.834e-06 | 1.472e-03 | 9.609e+00 | 6.937e-04 |
| fla_fused_rec vs fp64_ref | 1.047e-08 | 5.076e-10 | 1.290e-07 | 1.874e-03 | 5.894e-08 |
| fla_naive_rec vs fp64_ref | 1.388e-08 | 5.073e-10 | 1.291e-07 | 1.107e-03 | 5.894e-08 |
| fla_naive_chunk vs fp64_ref | 1.038e-07 | 1.963e-09 | 6.229e-07 | 1.403e-02 | 8.816e-07 |
| hf_torch_chunk vs fp64_ref | 1.038e-07 | 1.975e-09 | 6.249e-07 | 1.403e-02 | 8.861e-07 |
| hf_torch_rec vs fp64_ref | 8.530e-09 | 4.256e-10 | 1.067e-07 | 1.238e-03 | 5.894e-08 |

**float32|inkernel_l2**（输出 dtype：fla_chunk=float32, fla_fused_rec=float32, hf_torch_chunk=float32, hf_torch_rec=float32）

| 对比 | o max\|Δ\| | o mean\|Δ\| | o ‖Δ‖/‖ref‖ | o max rel | 末状态 S max\|Δ\| |
|---|---|---|---|---|---|
| fla_chunk vs fla_fused_rec | 1.135e-04 | 5.833e-06 | 1.472e-03 | 9.616e+00 | 6.937e-04 |
| fla_chunk vs hf_torch_chunk | 1.135e-04 | 5.833e-06 | 1.472e-03 | 9.680e+00 | 6.937e-04 |
| fla_chunk vs hf_torch_rec | 1.135e-04 | 5.833e-06 | 1.472e-03 | 9.609e+00 | 6.937e-04 |
| fla_fused_rec vs hf_torch_chunk | 1.043e-07 | 2.096e-09 | 6.436e-07 | 1.363e-02 | 9.239e-07 |
| fla_fused_rec vs hf_torch_rec | 1.490e-08 | 6.722e-10 | 1.767e-07 | 1.845e-03 | 1.192e-07 |
| hf_torch_chunk vs hf_torch_rec | 1.043e-07 | 2.028e-09 | 6.332e-07 | 1.181e-02 | 8.941e-07 |

**bfloat16|prenorm_l2**（输出 dtype：fla_chunk=bfloat16, fla_fused_rec=bfloat16, fla_naive_rec=float32, fla_naive_chunk=float32, hf_torch_chunk=bfloat16, hf_torch_rec=bfloat16）

| 对比 | o max\|Δ\| | o mean\|Δ\| | o ‖Δ‖/‖ref‖ | o max rel | 末状态 S max\|Δ\| |
|---|---|---|---|---|---|
| fla_chunk vs fla_fused_rec | 4.883e-04 | 1.012e-05 | 3.707e-03 | 1.593e+01 | 2.799e-03 |
| fla_chunk vs fla_naive_rec | 4.857e-04 | 1.163e-05 | 3.338e-03 | 1.595e+01 | 2.799e-03 |
| fla_chunk vs fla_naive_chunk | 4.857e-04 | 1.163e-05 | 3.338e-03 | 1.599e+01 | 2.799e-03 |
| fla_chunk vs hf_torch_chunk | 4.883e-04 | 1.012e-05 | 3.707e-03 | 1.600e+01 | 2.799e-03 |
| fla_chunk vs hf_torch_rec | 4.883e-04 | 1.012e-05 | 3.707e-03 | 1.593e+01 | 2.799e-03 |
| fla_fused_rec vs fla_naive_rec | 1.219e-04 | 5.678e-06 | 1.658e-03 | 3.889e-03 | 5.960e-08 |
| fla_fused_rec vs fla_naive_chunk | 1.220e-04 | 5.678e-06 | 1.658e-03 | 8.160e-03 | 8.941e-07 |
| fla_fused_rec vs hf_torch_chunk | 6.104e-05 | 1.885e-09 | 4.172e-05 | 7.750e-03 | 8.643e-07 |
| fla_fused_rec vs hf_torch_rec | 6.104e-05 | 7.953e-10 | 2.847e-05 | 7.572e-03 | 5.960e-08 |
| fla_naive_rec vs fla_naive_chunk | 9.313e-08 | 2.042e-09 | 6.328e-07 | 6.853e-03 | 8.941e-07 |
| fla_naive_rec vs hf_torch_chunk | 1.219e-04 | 5.678e-06 | 1.658e-03 | 5.890e-03 | 8.643e-07 |
| fla_naive_rec vs hf_torch_rec | 1.219e-04 | 5.678e-06 | 1.658e-03 | 3.903e-03 | 0.000e+00 |
| fla_naive_chunk vs hf_torch_chunk | 1.220e-04 | 5.678e-06 | 1.658e-03 | 3.904e-03 | 8.941e-08 |
| fla_naive_chunk vs hf_torch_rec | 1.220e-04 | 5.678e-06 | 1.658e-03 | 8.094e-03 | 8.941e-07 |
| hf_torch_chunk vs hf_torch_rec | 6.104e-05 | 1.924e-09 | 4.301e-05 | 7.750e-03 | 8.643e-07 |
| fla_chunk vs fp64_ref | 4.857e-04 | 1.163e-05 | 3.338e-03 | 1.595e+01 | 2.799e-03 |
| fla_fused_rec vs fp64_ref | 1.219e-04 | 5.678e-06 | 1.658e-03 | 3.889e-03 | 5.698e-08 |
| fla_naive_rec vs fp64_ref | 1.173e-08 | 5.079e-10 | 1.293e-07 | 1.634e-03 | 6.992e-08 |
| fla_naive_chunk vs fp64_ref | 9.398e-08 | 1.960e-09 | 6.211e-07 | 6.960e-03 | 8.884e-07 |
| hf_torch_chunk vs fp64_ref | 1.219e-04 | 5.678e-06 | 1.658e-03 | 5.875e-03 | 8.586e-07 |
| hf_torch_rec vs fp64_ref | 1.219e-04 | 5.678e-06 | 1.658e-03 | 3.889e-03 | 6.992e-08 |

**bfloat16|inkernel_l2**（输出 dtype：fla_chunk=bfloat16, fla_fused_rec=bfloat16, hf_torch_chunk=bfloat16, hf_torch_rec=bfloat16）

| 对比 | o max\|Δ\| | o mean\|Δ\| | o ‖Δ‖/‖ref‖ | o max rel | 末状态 S max\|Δ\| |
|---|---|---|---|---|---|
| fla_chunk vs fla_fused_rec | 4.883e-04 | 1.428e-05 | 4.372e-03 | 3.050e+01 | 3.926e-03 |
| fla_chunk vs hf_torch_chunk | 4.883e-04 | 1.428e-05 | 4.372e-03 | 3.047e+01 | 3.926e-03 |
| fla_chunk vs hf_torch_rec | 4.883e-04 | 1.428e-05 | 4.372e-03 | 3.047e+01 | 3.926e-03 |
| fla_fused_rec vs hf_torch_chunk | 1.221e-04 | 1.955e-09 | 4.092e-05 | 7.797e-03 | 8.941e-07 |
| fla_fused_rec vs hf_torch_rec | 1.221e-04 | 5.573e-10 | 2.647e-05 | 7.797e-03 | 8.941e-08 |
| hf_torch_chunk vs hf_torch_rec | 1.221e-04 | 2.041e-09 | 4.542e-05 | 7.805e-03 | 8.941e-07 |


**5.6.2 整模型**。Qwen3.5-0.8B，同一输入：COCO 000000039769（640×480，300 个视觉 token），加一句指令，seq_len 321。对比 fla 路径（默认）与 torch 回退路径（把模块级的 `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` 换成其 `__wrapped__` torch 原函数）的 logits。另起一个进程，在 import transformers 之前屏蔽 `fla` 的 import，让 transformers 自己走回退；其 logits 的 sha256 与进程内的 torch 参考逐位相同（最后一列）。

| dtype | logits max\|Δ\| | mean\|Δ\| | ‖Δ‖/‖ref‖ | 逐位置 argmax 一致 | 末位置 argmax 一致 | 末位置 top5（fla） | 末位置 top5（torch） | 同路径重复两次逐位相同（fla / torch） | 进程内 torch 参考 与 屏蔽 fla 的独立进程 sha256 相同 |
|---|---|---|---|---|---|---|---|---|---|
| bfloat16 | 1.281 | 0.04718 | 0.01686 | 309/321 | True | [11280, 8771, 32, 357, 3733] | [11280, 8771, 32, 357, 3733] | True / True | True |
| float32 | 0.1007 | 0.002631 | 0.001042 | 321/321 | True | [11280, 32, 8771, 357, 3733] | [11280, 32, 8771, 357, 3733] | True / True | True |

**5.6.3 transformers 实际走的路径**（transformers 5.17 在 import 时解析，`use_kernel_func_from_hub_with_fallback`：优先 HF kernels（需另装 `kernels`，未装），其次原包，最后 torch）：

| 函数 | 装了 fla（默认） | 屏蔽 fla 的 import |
|---|---|---|
| `torch_chunk_gated_delta_rule` | `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` | `transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule` |
| `torch_recurrent_gated_delta_rule` | `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule` | `transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule` |
| `causal_conv1d_fn` | `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn` | `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn` |
| `causal_conv1d_update` | `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update` | `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update` |

- 全注意力层：`attn_implementation` = `sdpa`（未装 flash-attn）。`Qwen3_5RMSNormGated` 等 `use_kernel_forward_from_hub` 装饰的层，因为没装 `kernels` 包，都走 torch。
- 运行时的警告（原样）。默认路径：``causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel.`。屏蔽 fla 后多出：`` `chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed. ... ``。
- 结论只陈述：卷积走 torch（`F.conv1d`，未装 causal-conv1d）；GDN 的 prefill 走 fla chunk 内核，逐 token 解码走 fla fused_recurrent 内核。

### 5.7 隐藏状态与任务调制（D1 仪器）

取一张图（COCO 000000039769）和两条 token 长度相同的指令，batch=1，dtype 与内核相同（默认路径：fla + sdpa），取图像 token 位置上 `output_hidden_states` 的每一层（第 0 层是嵌入输出，即图像特征并入序列之后）。

**qwen3.5-0.8b bf16**：指令 A = `Describe the cat.`，B = `Describe the sofa.`（token 长度 [4, 4]），attn=sdpa，图像 token 300 个，seq_len 318。

| 层 | 图在前：torch.equal | 图在前：max\|Δ\| | 指令在前：torch.equal | 指令在前：max\|Δ\| | 指令在前：cos（展平） | 指令在前：逐 token cos 最小 | 指令在前：变化的 token 数 |
|---|---|---|---|---|---|---|---|
| 0 | True | 0 | True | 0 | 1.000000 | 1.000000 | 0 |
| 1 | True | 0 | False | 0.03125 | 0.999989 | 0.999793 | 300 |
| 2 | True | 0 | False | 0.125 | 0.999950 | 0.999304 | 300 |
| 3 | True | 0 | False | 0.1562 | 0.999837 | 0.995823 | 300 |
| 4 | True | 0 | False | 0.1562 | 0.999653 | 0.918870 | 300 |
| 5 | True | 0 | False | 0.2695 | 0.999290 | 0.914848 | 300 |
| 6 | True | 0 | False | 0.207 | 0.998588 | 0.850324 | 300 |
| 7 | True | 0 | False | 0.1606 | 0.998367 | 0.860290 | 300 |
| 8 | True | 0 | False | 0.2617 | 0.996630 | 0.832538 | 300 |
| 9 | True | 0 | False | 0.2322 | 0.995711 | 0.820862 | 300 |
| 10 | True | 0 | False | 0.2432 | 0.993662 | 0.699117 | 300 |
| 11 | True | 0 | False | 0.2297 | 0.994222 | 0.728582 | 300 |
| 12 | True | 0 | False | 0.3032 | 0.991927 | 0.598239 | 300 |
| 13 | True | 0 | False | 0.29 | 0.990681 | 0.589617 | 300 |
| 14 | True | 0 | False | 0.3213 | 0.989908 | 0.578554 | 300 |
| 15 | True | 0 | False | 0.3662 | 0.989909 | 0.587303 | 300 |
| 16 | True | 0 | False | 0.4893 | 0.989588 | 0.531738 | 300 |
| 17 | True | 0 | False | 0.5923 | 0.989148 | 0.490026 | 300 |
| 18 | True | 0 | False | 0.6719 | 0.989066 | 0.501896 | 300 |
| 19 | True | 0 | False | 0.8516 | 0.987770 | 0.469217 | 300 |
| 20 | True | 0 | False | 1.07 | 0.986223 | 0.437157 | 300 |
| 21 | True | 0 | False | 1.188 | 0.986438 | 0.450028 | 300 |
| 22 | True | 0 | False | 1.25 | 0.986299 | 0.447278 | 300 |
| 23 | True | 0 | False | 1.594 | 0.984930 | 0.419426 | 300 |
| 24 | True | 0 | False | 20.25 | 0.979475 | 0.293624 | 300 |

图在前：25/25 层逐位相同；指令在前：1/25 层逐位相同。

**qwen3.5-9b bf16**：指令 A = `Describe the cat.`，B = `Describe the sofa.`（token 长度 [4, 4]），attn=sdpa，图像 token 300 个，seq_len 318。

| 层 | 图在前：torch.equal | 图在前：max\|Δ\| | 指令在前：torch.equal | 指令在前：max\|Δ\| | 指令在前：cos（展平） | 指令在前：逐 token cos 最小 | 指令在前：变化的 token 数 |
|---|---|---|---|---|---|---|---|
| 0 | True | 0 | True | 0 | 1.000000 | 1.000000 | 0 |
| 1 | True | 0 | False | 0.875 | 0.999977 | 0.999582 | 300 |
| 2 | True | 0 | False | 0.875 | 0.999938 | 0.998703 | 300 |
| 3 | True | 0 | False | 0.875 | 0.999865 | 0.997585 | 300 |
| 4 | True | 0 | False | 1 | 0.999803 | 0.992653 | 300 |
| 5 | True | 0 | False | 1.375 | 0.999519 | 0.984464 | 300 |
| 6 | True | 0 | False | 1.625 | 0.999206 | 0.972567 | 300 |
| 7 | True | 0 | False | 2.625 | 0.998335 | 0.968545 | 300 |
| 8 | True | 0 | False | 2.75 | 0.996508 | 0.898640 | 300 |
| 9 | True | 0 | False | 3.125 | 0.994891 | 0.813431 | 300 |
| 10 | True | 0 | False | 3.125 | 0.993880 | 0.792269 | 300 |
| 11 | True | 0 | False | 5 | 0.992838 | 0.838759 | 300 |
| 12 | True | 0 | False | 2.004 | 0.990285 | 0.802610 | 300 |
| 13 | True | 0 | False | 2.453 | 0.988673 | 0.777287 | 300 |
| 14 | True | 0 | False | 2.789 | 0.987202 | 0.770877 | 300 |
| 15 | True | 0 | False | 9 | 0.984846 | 0.777531 | 300 |
| 16 | True | 0 | False | 4.75 | 0.980958 | 0.715149 | 300 |
| 17 | True | 0 | False | 4.25 | 0.976210 | 0.666013 | 300 |
| 18 | True | 0 | False | 7.75 | 0.972650 | 0.623768 | 300 |
| 19 | True | 0 | False | 23.38 | 0.963077 | 0.349821 | 300 |
| 20 | True | 0 | False | 13.12 | 0.950495 | 0.295070 | 300 |
| 21 | True | 0 | False | 19 | 0.942178 | 0.188477 | 300 |
| 22 | True | 0 | False | 16.56 | 0.935682 | 0.172591 | 300 |
| 23 | True | 0 | False | 37.38 | 0.918142 | 0.044842 | 300 |
| 24 | True | 0 | False | 31.88 | 0.918725 | 0.082134 | 300 |
| 25 | True | 0 | False | 42.25 | 0.915448 | 0.118854 | 300 |
| 26 | True | 0 | False | 50.25 | 0.911739 | 0.109275 | 300 |
| 27 | True | 0 | False | 38.25 | 0.904483 | 0.057684 | 300 |
| 28 | True | 0 | False | 62.75 | 0.912922 | 0.175195 | 300 |
| 29 | True | 0 | False | 64.25 | 0.908536 | 0.163184 | 300 |
| 30 | True | 0 | False | 60.5 | 0.912985 | 0.203838 | 300 |
| 31 | True | 0 | False | 61.25 | 0.899839 | 0.103097 | 300 |
| 32 | True | 0 | False | 14.47 | 0.884071 | 0.046460 | 300 |

图在前：33/33 层逐位相同；指令在前：1/33 层逐位相同。

**qwen3.5-0.8b fp32**：指令 A = `Describe the cat.`，B = `Describe the sofa.`（token 长度 [4, 4]），attn=sdpa，图像 token 300 个，seq_len 318。

| 层 | 图在前：torch.equal | 图在前：max\|Δ\| | 指令在前：torch.equal | 指令在前：max\|Δ\| | 指令在前：cos（展平） | 指令在前：逐 token cos 最小 | 指令在前：变化的 token 数 |
|---|---|---|---|---|---|---|---|
| 0 | True | 0 | True | 0 | 1.000000 | 1.000000 | 0 |
| 1 | True | 0 | False | 0.02456 | 0.999992 | 0.999798 | 300 |
| 2 | True | 0 | False | 0.1258 | 0.999963 | 0.999343 | 300 |
| 3 | True | 0 | False | 0.1657 | 0.999855 | 0.995924 | 300 |
| 4 | True | 0 | False | 0.1452 | 0.999661 | 0.912916 | 300 |
| 5 | True | 0 | False | 0.2493 | 0.999286 | 0.893904 | 300 |
| 6 | True | 0 | False | 0.1933 | 0.998542 | 0.819886 | 300 |
| 7 | True | 0 | False | 0.1683 | 0.998320 | 0.836631 | 300 |
| 8 | True | 0 | False | 0.2417 | 0.996558 | 0.815974 | 300 |
| 9 | True | 0 | False | 0.2341 | 0.995635 | 0.811759 | 300 |
| 10 | True | 0 | False | 0.2435 | 0.993555 | 0.707440 | 300 |
| 11 | True | 0 | False | 0.2394 | 0.994137 | 0.731326 | 300 |
| 12 | True | 0 | False | 0.3099 | 0.991937 | 0.600104 | 300 |
| 13 | True | 0 | False | 0.2989 | 0.990829 | 0.584252 | 300 |
| 14 | True | 0 | False | 0.3422 | 0.990083 | 0.576285 | 300 |
| 15 | True | 0 | False | 0.4012 | 0.990138 | 0.575903 | 300 |
| 16 | True | 0 | False | 0.5188 | 0.989711 | 0.521399 | 300 |
| 17 | True | 0 | False | 0.6038 | 0.989262 | 0.463810 | 300 |
| 18 | True | 0 | False | 0.676 | 0.989147 | 0.484789 | 300 |
| 19 | True | 0 | False | 0.8803 | 0.987736 | 0.451974 | 300 |
| 20 | True | 0 | False | 1.062 | 0.985972 | 0.419220 | 300 |
| 21 | True | 0 | False | 1.153 | 0.986169 | 0.436857 | 300 |
| 22 | True | 0 | False | 1.334 | 0.986035 | 0.428453 | 300 |
| 23 | True | 0 | False | 1.354 | 0.984622 | 0.400627 | 300 |
| 24 | True | 0 | False | 20.04 | 0.979022 | 0.276087 | 300 |

图在前：25/25 层逐位相同；指令在前：1/25 层逐位相同。

