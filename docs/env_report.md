# env_report — b200 环境核查（RUN-0001 §3，§5.6.3，§5.8）

记录者：服务器 CC，2026-09-30。只记事实，不解读。原始输出在 `experiments/P0/0001_env_probe/raw/`。

## 1. 硬件、驱动、CUDA、torch

| 项 | 值 |
|---|---|
| GPU | NVIDIA B200 ×1，183359 MiB（torch 可见 178.35 GiB），power limit 1000 W，compute_cap 10.0 |
| `torch.cuda.get_device_capability()` | **(10, 0)** |
| 驱动 | 580.159.03（nvidia-smi 报 CUDA Version 13.0） |
| torch | 2.13.0+cu130（`torch.version.cuda`=13.0，cuDNN 92000，NCCL 2.29.7） |
| torch arch list | sm_75, sm_80, sm_86, sm_90, **sm_100**, sm_120 |
| nvcc（PATH 上） | 13.3 (V13.3.73) |
| triton | 3.7.1（base env） |
| Python | 3.12.13；Ubuntu 22.04.5，kernel 5.15.0-179 |
| CPU / RAM | 30 核；180 GB RAM，无 swap |

nvidia-smi 头部（2026-09-30 05:11 UTC，空闲时）：

```
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
|   0  NVIDIA B200                    On  |   00000000:05:00.0 Off |                    0 |
| N/A   34C    P0            247W / 1000W |       0MiB / 183359MiB |      0%      Default |
```

## 2. venv

- `$ROOT/env/bin/python -m venv --system-site-packages $PROJ/venv`，1.0 s。`$ROOT/env` 未改动。
- venv 继承 base env 的 torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / triton 3.7.1 / numpy 2.5.2 / scipy / pillow / einops / pytest / PyYAML，未重装。
- 环境变量统一由 `$PROJ/scripts/env.sh` 设置（副本在 `experiments/P0/0001_env_probe/probe_scripts/env.sh`）。**注意**：任务书 §3 的一行写法
  `export ROOT=... PROJ=$ROOT/vision-internalization` 在 bash 里会先展开 `$ROOT`（此时为空）再赋值，得到 `PROJ=/vision-internalization`、`HF_HOME=/.hf`（根盘上项目外的路径）。已实测确认（见 report §4）。env.sh 改为每行一个 export。
- 额外设置（env.sh 内）：`HF_XET_CACHE=/dev/shm/visint/xet`、`HF_XET_CHUNK_CACHE_SIZE_BYTES=0`（hf-xet 默认把 chunk cache 与日志写到 `$HF_HOME/xet`，即根盘）；`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`XDG_CACHE_HOME`、`HF_DATASETS_CACHE` 都指到 `$PROJ/tmp/` 下。

## 3. 安装的包（全部装在 venv，`PIP_NO_CACHE_DIR=1`）

| 批次 | 命令 | 耗时 | 装了什么 | `df /` 可用（前→后） |
|---|---|---|---|---|
| 1 | `pip install transformers accelerate peft huggingface_hub` | 5.6 s | transformers 5.17.0, accelerate 1.15.0, peft 0.21.1, huggingface_hub 1.33.0, safetensors 0.8.0, tokenizers 0.23.2, hf-xet 1.6.0, psutil 7.2.2, regex, httpx 等 23 个 | 6.365 → 6.184 GB |
| 2 | `pip install flash-linear-attention shapely pycocotools` | 1.2 s | flash-linear-attention 0.5.2, fla-core 0.5.2, shapely 2.1.2, pycocotools 2.0.11 | 6.184 → 6.155 GB |

两批 dry-run 都确认**不会**装 torch（`logs/pip_dryrun_batch{1,2}.json`）。einops、pyyaml、pytest、scipy、pillow 在 base env 里已有，跳过。

- **transformers**：最新 release **5.17.0 已含 `models/qwen3_5`**（`Qwen3_5ForConditionalGeneration`，经 `AutoModelForImageTextToText` 加载），同时含 `gemma4`、`internvl`。未装 main。
- **qwen-vl-utils**：Qwen3.5 各尺寸 model card（0.8B/9B/27B/397B-A17B）的示例都没有引用它，未安装。transformers 路径下 `AutoProcessor`（`Qwen3VLProcessor` + `Qwen2VLImageProcessor`）自己完成图像预处理。
- 未安装：flash-attn、causal-conv1d、kernels、vllm、sglang、datasets。

最终版本（`logs/env_versions.txt`）：

```
transformers==5.17.0  accelerate==1.15.0  peft==0.21.1  flash-linear-attention==0.5.2  fla-core==0.5.2
huggingface_hub==1.33.0  hf-xet==1.6.0  safetensors==0.8.0  tokenizers==0.23.2  shapely==2.1.2  pycocotools==2.0.11
(base env) torch==2.13.0+cu130 torchvision==0.28.0+cu130 triton==3.7.1 numpy==2.5.2 scipy==1.18.0 pillow==12.3.0
           einops==0.8.2 PyYAML==6.0.3 pytest==9.1.1
```

运行时出现的警告（原样）：
- `` `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`. ``（加载 Qwen3.5 时，多次）
- `Flash Attention is not installed. Please install it via pip install flash-attn --no-build-isolation`（加载时）
- `` `causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed. This is correct but much slower; install `causal_conv1d` for the optimized kernel. ``（首次前向）

## 4. vLLM / SGLang：只做 dry-run，未安装

pip dry-run 的临时目录放在 `/dev/shm`（resolver 会下载整个 vllm wheel 320 MB 读元数据），完成后已删。下载体积用 pip `--report` 里每个 URL 的 HTTP HEAD `Content-Length` 求和，未实际下载。

| | vLLM nightly | SGLang main |
|---|---|---|
| 命令 | `pip install --dry-run vllm --pre --extra-index-url https://wheels.vllm.ai/nightly` | `pip install --dry-run 'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python'` |
| 解析到的版本 | vllm 0.30.1rc1.dev392+ge006d761a | sglang 0.5.21.dev835+geb9c9ee99（commit eb9c9ee99d47） |
| 结果 | 成功，2 m 40 s | 第一次失败：`cargo is required to discover the Rust extension modules ... or set SGLANG_BUILD_RUST_EXTS=none`（机器上没有 Rust 工具链）。加 `SGLANG_BUILD_RUST_EXTS=none` 重跑，成功，1 m 24 s |
| 要装的包数 | 137 | 186 |
| 是否另装 torch | **否**：vllm 声明 `torch==2.13.0`，与现有一致 | **否**：sglang、sglang-kernel、sgl-deep-ep 都声明 `torch==2.13.0` |
| 会改动的已装包 | numpy 2.5.2→2.3.5；新装 torchaudio 2.11.0（其元数据未声明 torch 依赖）、torchcodec 0.16.0 | **transformers 5.17.0→5.12.1**（sglang 声明 `transformers==5.12.1`）；numpy→2.3.5；pillow 12.3.0→11.3.0；PyYAML 6.0.3→6.0.1；新装 torchaudio 2.11.0、flash-attn-4 4.0.0b19、kernels 0.14.1、datasets 5.0.1 等 |
| 下载总体积（wheel 压缩后） | **1.163 GB**（最大：vllm 320.6 MB、tokenspeed-triton 99.9 MB、cutlass-dsl-libs-cu12/cu13 各约 88 MB） | **1.383 GB** + sglang 源码 git clone（未计入）；最大：sglang-kernel 373.1 MB |
| 安装后体积 | 未测（需实际安装，wheel 解压后会大于下载体积） | 同左 |

完整清单：`raw/pip_dryrun_vllm_nightly.json`、`raw/pip_dryrun_sglang_main_norust.json` 与对应的 `*_sizes.txt`。

## 5. 磁盘与内存盘

| 时刻 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|
| 开始（05:11） | 6.387 GB | 0 | 0.4 MB |
| venv + 两批 pip 后 | 6.155 GB | 0 | 0.20 GB |
| 两个模型权重下载后 | 6.072 GB | 21.09 GB | 0.28 GB |
| 批次 1 写报告前（05:40） | 5.911 GB | 21.09 GB | 0.43 GB |

`/dev/shm` 是 91 GB tmpfs，写报告前已用 20 GB（全部为本项目）。逐步收支见 report §6 与 `raw/disk_ledger.tsv`。
