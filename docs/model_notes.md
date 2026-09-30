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
