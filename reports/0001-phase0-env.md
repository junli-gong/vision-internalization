# RUN-0001 — Phase 0：环境核查、Qwen3.5 事实探针、模型登记、数据源体积与 schema（磁盘受限版）

服务器 CC，b200，2026-09-30 05:10–06:36 UTC。只贴数字与事实，不做解读。事实汇总在 `docs/env_report.md`、`docs/model_notes.md`、`docs/data_sources.md`；原始输出在 `experiments/P0/0001_env_probe/raw/`。

## 0. 一句话观察（只陈述，不解读）

transformers 5.17.0 release 能加载 Qwen3.5（不需要 main）。§5.7 图在前时，图像 token 的隐藏状态在两个模型的每一层都逐位相同（0.8B 25/25、9B 33/33 层，bf16；0.8B fp32 也是 25/25）；指令在前时只有第 0 层相同。两个 Qwen3.5 模型都输出 `bbox_2d`，640×480 图上的 y2 为 988 与 989。CV-Bench 含 NYUv2 官方 test 的 36 张图；BLINK 的 test 答案为 "hidden"，其评测服务器已关闭。根盘从 6.387 GB 降到 5.927 GB；内存盘 `/dev/shm/visint` 占用 21.09 GB，为 0.8B 与 9B 的权重，按任务书保留。

## 1. 前置校验

- `git pull`：1fdeb1d..587b9a5 fast-forward。`git log --oneline | head -3` 包含 `587b9a5 [mgr] RUN-0001 and DEV-0001 issued; RUN-0001 predictions logged` 和 `1fdeb1d [mgr] bootstrap`，符合任务书 §1。
- 测试：任务书写明本次没有测试可跑，未运行 pytest。本次没有修改任何仓库内的代码或 config。
- 仓库级 git：`core.sshCommand` 使用 `$PROJ/.ssh/id_ed25519_github`，身份 `b200-server-cc`，未改任何全局文件。
- 本次的推送：`45ce09b`（批次 1：§3 + §4），`27b00a5`（批次 2：§5 + experiments），批次 3 为本 report 与 §6 所在的 commit。

## 2. 执行记录

所有脚本都先 source `$PROJ/scripts/env.sh`（HF_HOME、HF_HUB_CACHE=/dev/shm/visint/hf-hub、HF_XET_CACHE=/dev/shm/visint/xet、TMPDIR=$PROJ/tmp、PIP_NO_CACHE_DIR=1 等）。

| 步骤 | 命令或脚本 | 墙钟 | 结果 |
|---|---|---|---|
| venv | `$ROOT/env/bin/python -m venv --system-site-packages $PROJ/venv` | 1.0 s | ok |
| pip 批次 1 | `pip install transformers accelerate peft huggingface_hub`（先 dry-run 确认不装 torch） | 5.6 s | transformers 5.17.0 等 |
| pip 批次 2 | `pip install flash-linear-attention shapely pycocotools` | 1.2 s | fla 0.5.2 等 |
| §4 登记 | `p4_model_registry.py`（32 个仓库，只下小文件） | 约 1 min | 全部可访问，都不是 gated |
| vLLM dry-run | `p3_dryrun.sh vllm`（临时目录 /dev/shm） | 2 m 40 s | 成功，137 个包 |
| SGLang dry-run | `p3_dryrun.sh sgl` | 1 m 09 s | **失败**：缺 cargo |
| SGLang dry-run（重跑） | `p3_dryrun_sgl_norust.sh`（`SGLANG_BUILD_RUST_EXTS=none`） | 1 m 24 s | 成功，186 个包 |
| 下载 0.8B | `snapshot_download('Qwen/Qwen3.5-0.8B', revision=2fc0636471)` | 6.8 s | 1.77 GB → /dev/shm |
| 下载 9B | `p5_dl_9b.sh`（revision c202236235） | 34.2 s | 19.33 GB → /dev/shm |
| §5.1–5.4 | `p5_facts.py`，0.8B / 9B | 12.1 s / 14.9 s | ok |
| §5.5 | `p5_5_box_zeroshot.py`，0.8B / 9B | 10.8 s / 25.6 s | ok |
| §5.6.1 | `p5_6_gdn_kernels.py` | 52.2 s | ok |
| §5.6.2–3 | `p5_6_model_paths.py inproc` / `blockfla` | 14.3 s / 10.2 s | ok |
| §5.7 | `p5_7_taskmod.py`：0.8B bf16、9B bf16、0.8B fp32 | 各约 15–30 s | ok |
| §5.8 | `p5_8_mem.py`：0.8B infer / lora，9B infer / lora | 33.0 / 51.9 / 33.1 / 57.0 s | ok，无 OOM |
| §6 | 三个并行的只读探针（HEAD、HF API、datasets-server、zip Range） | 约 44 min（三组并行，05:48–06:33；耗时最长的是 A 组，其中对 30,956 张图逐张 HEAD 用了 499 s） | 见 §3.6 |

GPU：NVIDIA B200（capability (10, 0)，183359 MiB）。§5.8 期间 nvidia-smi 每 500 ms 采样一次：util 最高 98–100%，各窗口均值 17.9–33.4%（窗口包含模型加载与 JIT）；nvidia-smi 显存最高 62264 MiB（9B LoRA）；功耗最高 882 W。torch 峰值显存见 §3.5。开工时（05:11）与 §5.8 开始采样时（05:35:41），nvidia-smi 显示 0 MiB 被占用、没有进程；其余时刻没有持续监控。

警告（原样，均为环境类）：
- `` `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`. ``
- `Flash Attention is not installed. Please install it via pip install flash-attn --no-build-isolation`
- `` `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel. ``
- `Specified kernel cache directory could not be created! This disables kernel caching. Specified directory is .../tmp/xdg-cache/torch/kernels`：前几个 GPU 探针出现过；原因是 env.sh 里设置的 `XDG_CACHE_HOME` 目录还不存在，建好目录后不再出现。
- `Kwargs passed to processor.__call__ have to be in processor_kwargs dict, not in **kwargs`：出现在向 processor 的 `apply_chat_template` 传 `chat_template_kwargs=` 时（§4 第 3 条）。

## 3. 结果表（原样）

### 3.1 环境

| 项 | 值 |
|---|---|
| `torch.cuda.get_device_capability()` | (10, 0) |
| torch / CUDA / cuDNN | 2.13.0+cu130 / 13.0 / 92000 |
| 驱动 | 580.159.03 |
| transformers | 5.17.0（release，含 `qwen3_5`） |
| fla | flash-linear-attention 0.5.2 / fla-core 0.5.2 |
| peft / accelerate | 0.21.1 / 1.15.0 |

vLLM nightly / SGLang main 的 dry-run：

| | vLLM 0.30.1rc1.dev392+ge006d761a | SGLang 0.5.21.dev835+geb9c9ee99 |
|---|---|---|
| 包数 | 137 | 186 |
| 是否另装 torch | 否（声明 `torch==2.13.0`） | 否（声明 `torch==2.13.0`） |
| 会降级的已装包 | numpy 2.5.2→2.3.5 | transformers 5.17.0→5.12.1、numpy→2.3.5、pillow 12.3.0→11.3.0、PyYAML 6.0.3→6.0.1 |
| 下载体积（HEAD 求和） | 1.163 GB | 1.383 GB + git clone |

### 3.2 模型登记（§4）

32 个仓库全部可访问，都不是 gated，许可证都是 apache-2.0。完整表见 `docs/model_notes.md` §2。

| 模型 | revision | safetensors | LLM 层数 / hidden | 注意力布局 |
|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B | 2fc06364715b | 1.75 GB | 24 / 1024 | 18 GDN + 6 full |
| Qwen/Qwen3.5-2B | 15852e8c1636 | 4.55 GB | 24 / 2048 | 18 GDN + 6 full |
| Qwen/Qwen3.5-4B | 851bf6e806ef | 9.32 GB | 32 / 2560 | 24 GDN + 8 full |
| Qwen/Qwen3.5-9B | c20223623576 | 19.31 GB | 32 / 4096 | 24 GDN + 8 full |
| google/gemma-4-E2B-it | 3e22461f65 | 10.25 GB | 35 / 1536 | 28 sliding(512) + 7 full |
| google/gemma-4-E4B-it | ee0ef60236 | 15.99 GB | 42 / 2560 | 35 sliding(512) + 7 full |
| OpenGVLab/InternVL3_5-2B-HF | 3f301ffcf3 | 4.70 GB | 28 / 2048 | 全注意力（qwen3） |
| OpenGVLab/InternVL3_5-4B-HF | 6bd4487402 | 9.47 GB | 36 / 2560 | 全注意力 |
| OpenGVLab/InternVL3_5-8B-HF | 741a7d0302 | 17.06 GB | 36 / 4096 | 全注意力 |

InternVL3.5 每个尺寸有 6 个变体（无后缀、-HF、-Instruct、-MPO、-Pretrained、-Flash），HF 格式只有最终版有。Gemma 4 另有预训练版和 `-it-assistant`（MTP drafter）。

### 3.3 Qwen3.5 结构与 tokenizer（§5.1–5.4；0.8B / 9B）

| 项 | 0.8B | 9B |
|---|---|---|
| 层 | 24（全注意力在第 3, 7, 11, 15, 19, 23 层） | 32（全注意力在第 3, 7, …, 31 层） |
| 全注意力 Q/KV/head_dim | 8 / 2 / 256 | 16 / 4 / 256 |
| GDN K 头 / V 头 / dim | 16 / 16 / 128 | 16 / 32 / 128 |
| 视觉 patch / merge | 16 / 2（每 token 对应 32×32 px） | 16 / 2 |
| 嵌入行数 / len(tokenizer) | 248320 / 248077（多出 243 行） | 248320 / 248077（多出 243 行） |
| tie embeddings | True | False |
| `<g0>` 的 id / 是否单个 token | 248077 / 是 | 248077 / 是 |
| `resize_token_embeddings(len(tok))` | 248320→248079（缩小） | 248320→248079（缩小） |
| 默认思考（模板） | 关 | 开 |
| 448×448 / 640×480 / 1024×768 的视觉 token | 196 (14×14) / 300 (15×20) / 768 (24×32) | 相同 |
| position_ids | get_rope_index (3,B,L)；generate (4,B,L) | 相同 |

### 3.4 §5.5、§5.6、§5.7 关键数字

§5.5 原始输出。提示：`Locate {obj} in the image. Report bbox coordinates in JSON format.`，greedy，思考关。

| 模型 | 图（W×H） | 目标 | 输出 bbox_2d | COCO GT xyxy 像素 | GT ×1000/(W,H) |
|---|---|---|---|---|---|
| 0.8B | 000000039769（640×480） | the cat on the left | [14, 114, 492, 988] | [17.5, 54.4, 318.8, 469.4] | [27, 113, 498, 978] |
| 9B | 000000039769（640×480） | the cat on the left | [11, 119, 494, 989] | 同上 | 同上 |
| 0.8B | 000000000139（640×426） | the television | [10, 391, 242, 610] | [7.0, 167.8, 156.3, 262.6]（左侧 tv） | [11, 394, 244, 617] |
| 9B | 000000000139（640×426） | the television | [10, 388, 241, 624] | 同上 | 同上 |

§5.6 GDN（B=2, T=300, H=16, K=V=128）：

| 对比 | fp32 o max\|Δ\| | fp32 ‖Δ‖/‖ref‖ | bf16 o max\|Δ\| | bf16 ‖Δ‖/‖ref‖ |
|---|---|---|---|---|
| fla_chunk vs fp64_ref（prenorm） | 1.135e-04 | 1.472e-03 | 4.857e-04 | 3.338e-03 |
| fla_fused_rec vs fp64_ref | 1.047e-08 | 1.290e-07 | 1.219e-04 | 1.658e-03 |
| hf_torch_chunk vs fp64_ref | 1.038e-07 | 6.249e-07 | 1.219e-04 | 1.658e-03 |
| hf_torch_rec vs fp64_ref | 8.530e-09 | 1.067e-07 | 1.219e-04 | 1.658e-03 |
| fla_chunk vs hf_torch_chunk（inkernel_l2） | 1.135e-04 | 1.472e-03 | 4.883e-04 | 4.372e-03 |

0.8B 整模型，fla 路径与 torch 回退的 logits 对比（seq 321）：bf16 max|Δ| 1.281，‖Δ‖/‖ref‖ 0.01686，argmax 一致 309/321，末位置 top5 相同；fp32 max|Δ| 0.1007，‖Δ‖/‖ref‖ 0.001042，argmax 一致 321/321。同一路径重复运行逐位相同。进程内的 torch 参考与屏蔽 fla 的独立进程 sha256 相同。

§5.7（逐层表见 model_notes §5.7）：

| 运行 | 图在前：逐位相同的层 | 指令在前：逐位相同的层 | 指令在前：最后一层 cos（展平） / 逐 token cos 最小 |
|---|---|---|---|
| 0.8B bf16 | 25/25 | 1/25（仅第 0 层） | 0.979475 / 0.293624 |
| 9B bf16 | 33/33 | 1/33（仅第 0 层） | 0.884071 / 0.046460 |
| 0.8B fp32 | 25/25 | 1/25（仅第 0 层） | 0.979022 / 0.276087 |

### 3.5 §5.8 显存与吞吐（torch 峰值 allocated / reserved，GiB）

| 测量 | 0.8B | 9B |
|---|---|---|
| 仅加载权重 | 1.589 / 1.605 | 17.527 / 17.543 |
| 推理前向 bs1，1024 视觉 token，logits_to_keep=1 | 1.794 / 1.812 | 17.795 / 17.822 |
| 推理前向 bs1，full logits | 2.152 / 2.547 | 18.096 / 18.619 |
| 推理前向 bs16，logits_to_keep=1 | 4.055 / 4.236 | 20.983 / 21.676 |
| 推理前向 bs16，full logits | 9.777 / 11.963 | 25.811 / 29.402 |
| 在线探针前向 bs8，全部 hidden，logits_to_keep=1 | 3.365 / 3.512 | 22.848 / 23.213 |
| 在线探针前向 bs8，全部 hidden，full logits | 6.093 / 7.375 | 23.743 / 27.076 |
| LoRA r=32 all-linear（含视觉塔），梯度检查点，bs8，前向+反向 | 30.664 / 31.686 | 50.196 / 52.699 |
| LoRA r=32 仅 LLM 线性层，梯度检查点，bs8，前向+反向 | 29.814 / 30.844 | 47.825 / 50.143 |
| HF generate bs16，max_new_tokens=64，思考关：样本/s（3 次） | 16.886 / 16.807 / 16.934 | 9.773 / 9.771 / 9.789 |
| HF generate 峰值 | 4.055 / 4.439 | 21.829 / 22.137 |

LoRA 的 labels 为整条序列（seq 1060，含 1024 个 image_pad），用 HF 默认 CE，不分块。可训练参数：0.8B 为 26.69M（all-linear）/ 21.65M（LLM-only），9B 为 102.53M / 86.56M。

### 3.6 数据源（§6）

完整总表见 `docs/data_sources.md` §1，下面是摘录：

| 数据集 | 许可（摘要） | 划分 / 条数 | 体积（HEAD 或 HF API） | 我们需要的子集 |
|---|---|---|---|---|
| RefCOCO / + / g | 数据许可未声明（refer 仓库代码为 Apache-2.0）；RefCOCOg-google 为 CC BY 4.0 | refs 50,000 / 49,856 / 49,822 | 标注 zip 54.0 / 45.6 / 56.7 MB（Wayback） | train2014 图像去重并集 30,956 张，5,252,007,403 B；只取评测划分为 6,549 张，1,105,415,725 B |
| COCO val2017 | 标注 CC BY 4.0；图像遵守 Flickr 条款（69.5% 为 NC） | 5,000 张 | 815,585,330 B | 500 张约 81.5 MB |
| ODinW13 | 混合（含 1 个 undefined） | train/valid/test = 24,489 / 4,938 / 4,575 张 | 2,249,062,175 B | — |
| CountBench | 未声明 | 540（不分划分） | 只有 URL | 前 50 个 URL 中非 200 占 16%；全部 540 个中占 22.8% |
| NYUv2 | 主页无许可文字 | 795 / 654 | labeled .mat 2,972,037,809 B | BTS 24,231 对（Drive 显示 "5.9G"）；HF aradhye 6,857,931,462 B |
| BLINK Relative_Depth | Apache-2.0 | val 124 / test 124（答案 hidden） | 9,203,975 B | 全部 |
| CV-Bench | apache-2.0（Omni3D 标注为 CC BY-NC 4.0） | 只有 test，2,638 | 404,808,364 B | 全部 |
| EmbSpatial-Bench | cc-by-4.0（Phineas476） | 只有 test，3,640 | 253,512,493 B（json） | 全部 |
| ADE20K SceneParse150 | 图像限非商业研究；标注 BSD-3 | 20,210 / 2,000 | 967,382,037 B | 整个 zip |
| Hypersim（可选） | CC BY-SA 3.0 | 59,543 / 7,386 / 7,690 | 1,914,288,906,139 B | 外推约 69–432 GB，取决于要哪些文件类型 |

## 4. 与任务书或简报假设不符的发现

1. **任务书 §3 的 export 写法有误**：`export ROOT=/home/... PROJ=$ROOT/vision-internalization` 在 bash 里先展开 `$ROOT`（此时为空）再执行 export，实测 `bash -c 'export ROOT=/x PROJ=$ROOT/y; echo $PROJ'` 输出 `/y`。照抄会得到 `PROJ=/vision-internalization`、`HF_HOME=/.hf`、`TMPDIR=/vision-internalization/tmp`，都是根盘上项目外的路径。本次改用每行一个 export 的 `env.sh`；原样照抄的命令没有执行过。CLAUDE.md 与 D3 没有这个问题。
2. **简报 §6.3 与 §12 写的是"Qwen3.5 默认开启思考"**。实测：0.8B 与 2B 的 chat template 默认关闭（card 原文："Qwen3.5-0.8B operates in non-thinking mode by default"），4B 与 9B 默认开启。
3. **`chat_template_kwargs={"enable_thinking": False}` 对 transformers 的 `apply_chat_template` 不生效**：在 9B 上渲染出的仍是思考开启。processor 版会打印 warning，tokenizer 版不报警也不生效。要用 `enable_thinking=False` 直接作关键字参数。`chat_template_kwargs` 是 vLLM/SGLang OpenAI API 的字段。
4. **Qwen3.5 仓库没有 `generation_config.json`**：transformers 默认 `eos_token_id`=248044（`<|endoftext|>`），不含 `<|im_end|>`（248046）。实测 generate 会先输出 `<|im_end|>\n`，再输出 `<|endoftext|>` 才停。
5. **词表**：嵌入行数 248320 = len(tokenizer) 248077 + 243 行。这 243 行的范数彼此接近（0.8B：0.4214–0.4215；9B 嵌入：0.194–0.204）。新增 token 的 id 从 248077 起，落在这 243 行里。**`resize_token_embeddings(len(tokenizer))` 会把矩阵缩小**（248320→248079）。与硬约束 4 相关的算术（只陈述）：248077 + 300 = 248377 > 248320，新增 300 个 token 要 resize 出 57 行；新增 ≤243 个不需要 resize。
6. **SGLang main 从源码安装需要 Rust/cargo**，或者设置 `SGLANG_BUILD_RUST_EXTS=none`；并且声明 `transformers==5.12.1`，会把现有的 5.17.0 降级。vLLM nightly 会把 numpy 从 2.5.2 降到 2.3.5。
7. **Qwen3.5-9B checkpoint 含 `mtp.*` 共 15 个张量**（约 2.4 亿参数）。transformers 不加载它们，也不报警（加载后 9,409,813,744 参数；safetensors 元数据为 9,653,104,368）。
8. **model card 内部不一致**：9B（以及 4B）的 card 里，"Instruct (or non-thinking) mode for reasoning tasks" 的推荐采样参数在 Tip 框与 Best Practices 一节不同：前者为 `top_p=0.95, top_k=20, presence_penalty=1.5`，后者为 `top_p=1.0, top_k=40, presence_penalty=2.0`。
9. **坐标格式没有 Qwen3.5 自己的文档**：Qwen3.5 的各 model card 与 GitHub README 里没有坐标格式说明；博客页面是 JS 渲染，未能抓取。GitHub `QwenLM/Qwen3.5` 已重定向到 `QwenLM/Qwen3.8`。只有前代 Qwen3-VL 的 cookbook 写明 "relative coordinates ranging from 0 to 1000"。
10. **flash-attn 与 causal-conv1d 未安装**：全注意力走 sdpa，GDN 的短卷积走 torch `F.conv1d`（会打印 warning）。fla 只接管 chunk 与 recurrent 两个 gated-delta-rule 函数。
11. **与简报 §5.3 的数字对照**（只做算术）：
    - 9B 为 32 层，其中 8 层全注意力，4 个 KV 头，head_dim 256：与 config 一致。
    - KV cache：8 层 × 2 × 4 × 256 × 2 B = 32768 B/token。
    - GDN 状态：24 层 × 32 头 × 128 × 128 × 4 B（fp32）= 50.3 MB/序列，不含 conv state。
    - 9B safetensors 19.33 GB（含 mtp）。
    - 全部 32 层 × 1024 token × 4096 × 2 B = 256 MiB/图。实测 `output_hidden_states` 返回 33 个张量（含嵌入输出）。
12. **§5.5、§5.6.2、§5.7、§5.8 用的两张图属于 COCO val2017**：000000039769 与 000000000139。简报 §7.1 的开放词汇检测要从 COCO val2017 抽 500 张子集。这两张图只用于零样本探针与数值探针，没有据此调提示词或参数。是否把它们排除出子集，由管理员决定。
13. model card 元数据 `OpenGVLab/InternVL3_5-2B-HF` 的 `base_model` 写的是 `OpenGVLab/InternVL3_5-4B-MPO`。

14. **RefCOCO 官方主机失效**：`bvisionweb1.cs.unc.edu` 今天 DNS 解析失败，refer README 自己写着 "the webserver is broken"。可用的只有 2022-04-13 的 Wayback 快照。Google Refexp 的原始 zip 返回 403。ODinW 官方 `download.py` 的链接返回 404，要用 HF 的 `/resolve/` 直链。
15. **CV-Bench 含 NYUv2 官方 test 的 36 张图**（89 行），同时包含 ADE20K val 633 张、COCO val2017 805 张（data_sources §2 第 1 条）。
16. **BLINK 的 test 答案全部隐藏**，官方唯一的 test 评测渠道 EvalAI challenge 2287 已于 2026-01-31 结束。
17. **"约 2.4 万帧 NYUv2 训练集"对应多个互不相同的集合**：BTS 清单 24,231 对；BTS sync 目录（按 aradhye 镜像）36,396 对；FastDepth/sayakpaul 47,584 帧；DenseDepth 自称 "50K"。BTS 清单里有 1,383 帧来自 raw 目录 `bookstore_0001d…j`，而 `bookstore_0001` 是官方 test 划分里的 scene。
18. **Eigen crop 的两种写法相差 1 像素**：toolbox 为 MATLAB `mask(45:471, 41:601)`，即 427×561；BTS 为 Python `[45:471, 41:601]`，即 426×560。
19. **与硬约束相关的疑问（按任务书"遇到与简报硬约束相关的疑问，停下写进 report"记录，只列事实，由管理员汇总裁定）**：
    - **硬约束 9"不使用许可不明的数据"**：以下来源的数据许可未声明或未能核实：
      - RefCOCO / RefCOCO+ 标注；
      - NYUv2（官方主页没有许可文字）；
      - CountBench（只有 LAION 图像 URL，没有许可声明）；
      - ODinW13 中的 thermalDogsAndPeople（"License: undefined"）；
      - BLINK 相对深度的图像来源 DIW；
      - 若干 HF 镜像（没有 card license，或几处写法互相矛盾）。

      另有限制性条款：ADE20K 图像仅限非商业研究与教育；Omni3D（CV-Bench 的 3D 部分）标注为 CC BY-NC 4.0；COCO val2017 图像逐图许可中 69.5% 为 NC 类。
    - **硬约束 7"防泄漏、按图像 ID 去重"**：
      - RefCOCOg(umd) train 与 RefCOCO(unc) 评测划分有 1,860 张共同图像；RefCOCO(unc) train 与 RefCOCOg(umd) 评测划分有 1,891 张共同图像。
      - CV-Bench（留出评测）含 NYUv2 test 36 张、ADE20K val 633 张、COCO val2017 805 张。
      - 本次 §5 探针用了 COCO val2017 的 2 张图（第 12 条）；这 2 张不在 CV-Bench 里。
    - **BLINK 的 test 已无法评测**，目前只能用 val 的 124 条（简报 §7.1 写的是"BLINK 相对深度"，没有指定划分）。

## 5. 自己造成的错误或浪费

1. 第一版 `diskcheck.sh` 用 `bc` 处理科学计数法，出现 syntax error，首条账目显示 "0.00GB"。ledger 的字节数记录正确；改用 awk 后正常。代价：约 1 分钟。
2. 用 HF datasets-server 的 `filter` 接口取 COCO GT 超时（索引未就绪），后台白等约 2 分钟，改用官方 zip 的 Range 抽取（只取 7.3 MB）。
3. `p5_6_gdn_kernels.py` 第一版把 fla 的 `naive_recurrent` 喂 fp64 当基准，但它内部会转成 fp32。运行前发现，改为本地的 float64 递推；错误版本没有产出结果。
4. env.sh 设置了 `XDG_CACHE_HOME=$PROJ/tmp/xdg-cache`，但没建目录，导致前几个 GPU 探针的 torch kernel cache 被禁用（只影响速度，不影响数值）。之后已建好目录。
5. §5.8 的第一次调用包含 Triton JIT 编译（LoRA 的第一次前向加反向约 40 s），表中已标注。没有为吞吐另做 JIT 预热以外的处理。
6. **§6 子代理偏离了我写给它们的规则（如实记录）**：
   - C 组的网络读取约 80 MB，超过我给它的 55 MB 预算；它落盘 23.7 MB，已删除。原因是它的 ODinW 脚本每取一个成员都要重读一遍 zip 中央目录（这部分共 37.5 MB）。§6 的总量仍在 200 MB 以内。
   - B 组在仓库里执行过一次只读的 `git status --short`，违反了"不做任何 git 操作"。仓库没有被改动。
   - A 组往 `/dev/shm/visint/p6A_prev_bytes.json` 写过一个 2.8 KB 的临时文件，超出允许的路径，写完立即删除。
   - B 组有一次 h5py 进程在退出时 core dump；`ulimit -c` 为 0，没有生成 core 文件。
   - 以上偏离都没有造成磁盘或共享环境的改动。

## 6. 生成文件清单与磁盘收支

**仓库内**（三次 push）：
- `docs/env_report.md`、`docs/model_notes.md`、`docs/data_sources.md`
- `experiments/P0/0001_env_probe/`：README.md、`probe_scripts/`（51 个脚本）、`raw/`（原始 JSON、log、csv；单个文件都小于 5 MB）
- `reports/0001-phase0-env.md`（本文件）

**仓库外**：
- `$PROJ/venv`（约 0.2 GB）
- `$PROJ/scripts`（探针脚本原件）、`$PROJ/logs/0001`（原始输出原件）、`$PROJ/tmp`（Triton 缓存、2 张 COCO 图 336 KB）
- `/dev/shm/visint/hf-hub`：Qwen3.5-0.8B（1.77 GB）与 Qwen3.5-9B（19.33 GB），共 21.09 GB，**按任务书保留，供后续 RUN 复用**（注意：重启后会丢失）
- `/dev/shm/visint/xet`（xet 日志，KB 级）

**磁盘收支**（`raw/disk_ledger.tsv`；单位 GB，1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 05:11:46 | start | 6.387 | 0.000 | 0.000 |
| 05:11:54 | start(after mkdir) | 6.379 | 0.000 | 0.000 |
| 05:12:08 | after venv create | 6.365 | 0.000 | 0.012 |
| 05:12:27 | after pip batch1 (transformers/accelerate/peft/hf_hub) | 6.184 | 0.000 | 0.177 |
| 05:12:44 | after pip batch2 (fla/shapely/pycocotools) | 6.155 | 0.000 | 0.202 |
| 05:15:09 | after p4 registry (small files only) | 6.151 | 0.000 | 0.205 |
| 05:16:44 | before 0.8B download | 6.150 | 0.321 | 0.205 |
| 05:16:51 | after 0.8B download | 6.150 | 2.091 | 0.205 |
| 05:19:17 | vllm dryrun done, tmp cleaned; sglang dryrun started | 6.149 | 1.770 | 0.207 |
| 05:25:15 | after p5.6 | 6.081 | 1.770 | 0.271 |
| 05:25:39 | before 9B download | 6.081 | 1.770 | 0.271 |
| 05:28:45 | after 9B download | 6.072 | 21.087 | 0.280 |
| 05:39:16 | before batch1 writeup | 5.911 | 21.087 | 0.432 |
| 05:46:39 | after batch2 push | 5.901 | 21.087 | 0.439 |
| 05:48:55 | p6_C start | 5.901 | 21.087 | 0.439 |
| 05:49:03 | p6_A start | 5.900 | 21.087 | 0.439 |
| 05:49:10 | p6_B start | 5.900 | 21.087 | 0.439 |
| 05:50:40 | p6_B before h5 range read | 5.896 | 21.087 | 0.444 |
| 06:02:56 | p6_C before odinw ann fetch | 5.925 | 21.087 | 0.476 |
| 06:09:48 | p6_C end | 5.923 | 21.087 | 0.477 |
| 06:16:44 | p6_B end | 5.926 | 21.087 | 0.474 |
| 06:22:59 | p6_A mid | 5.925 | 21.087 | 0.474 |
| 06:28:08 | p6_A before facts | 5.930 | 21.087 | 0.469 |
| 06:31:55 | p6_A end | 5.929 | 21.087 | 0.469 |
| 06:32:49 | p6 all agents done | 5.929 | 21.087 | 0.469 |
| 06:35:08 | after tmp cleanup | 5.927 | 21.087 | 0.471 |
| 06:36:19 | before batch3 commit | 5.927 | 21.087 | 0.471 |

- 根盘：本项目新增占用 0.471 GB（上限 3 GB）；整机根盘最低可用 5.896 GB（红线 3 GB），整个过程中没有触发红线。
- 内存盘：本项目峰值 21.09 GB（上限 25 GB）。pip dry-run 的临时目录（vLLM 约 0.31 GB）放在内存盘，用完已删。
- §6 下载的标注样本文件：合计写入磁盘约 57 MB，网络读取约 188 MB（均在"单个小于 50 MB、总量小于 200 MB"以内）：
  - 服务器 CC：`instances_val2017.json` 19,987,840 B（网络取回 7,340,760 B），Qwen3-VL cookbook 8,691,442 B；
  - A 组：落盘 168,736 B，网络 58.3 MB；
  - B 组：临时文件 4.64 MB，网络 30.2 MB；
  - C 组：落盘 23.68 MB，网络约 80 MB。

  全部临时文件都已删除（`raw/own_tmp_deleted.txt`，以及 `raw/p6/{A,B,C}/downloads.tsv` 与删除清单）。删除前记录了 instances_val2017.json 的 sha256：`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
- 没有删除、移动或修改任何非本项目文件，也没有写 `/root/.cache`、`~/.gitconfig`、`~/.ssh/config`。
