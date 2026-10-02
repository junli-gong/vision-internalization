# RUN-0015 — LISA-7B-v1 外部对照：B200 上独立 venv 冒烟（D36-5）

服务器 CC，b200，2026-10-02 00:12–00:25 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **最终结果：** 冒烟成功了，但用了一次修法（见 §4）。按 LISA `chat.py` 的流程加载 bf16 的 LISA-7B-v1，对 `000000039769.jpg` 和提示 "Can you segment the cat on the left in this image?" 输出了 1 个 mask：
  - 51,993 像素，占全图的 0.1692；
  - 外接框 [12, 56, 315, 473]（xyxy，图像 640×480）；
  - 回答文本 "Sure, the segmentation result is [SEG] ."；
  - 目检：叠加图上红色区域覆盖左边那只橘猫。
- **未加修法时失败。** 第 1 次运行在生成的第一步触发 CUDA device-side assert：embedding 的下标越界。
  - 根因：本 venv 没装 `tokenizers`。transformers 4.31 要求 tokenizers<0.14，而 Python 3.12 没有它的二进制包，本机也没有 Rust 无法编译。
  - 没装 `tokenizers` 时，transformers 4.31 自带的 `AddedToken` 替身没有 `__str__`。tokenizer_config 里的 bos/eos 因此变成新词 32004/32005，BOS 超出 embed_tokens 的 32004 行。
  - 修法：在探针脚本里，运行期给这个替身补上 `__str__ = content`，与 `tokenizers.AddedToken` 的行为一致。之后分词器长度 32004，bos/eos/unk/pad 依次为 1/2/0/0，冒烟成功。没有改 LISA 源码，也没有改 transformers 的源文件。
- **时间与显存：** 加载 5.29 s，推理 0.82 s。`torch.cuda.max_memory_allocated` 加载后是 15.44 GB，推理时是 17.19 GB。参数量 7,704,747,312。
- **磁盘：**
  - `$PROJ/ext/lisa` 合计 0.527 GB：venv 0.476 GB（建好时 0.012 GB），LISA 代码 0.050 GB。
  - 根盘可用从 39.272 GB 降到 38.726 GB，`$PROJ` 从 3.944 GB 升到 4.471 GB（份额 15 GB）。
  - 内存盘峰值 19.132 GB，其中权重 17.836 GB，已删除。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `59c44f9`。本任务书所在的 commit `36a776f` 在 HEAD 的历史里。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **改动范围：** 没有修改我们仓库的 `src/`、`scripts/`、`tests/`、`configs/`，也没有改 LISA 源码。LISA clone 的 `git status` 为空，见 `lisa_commit.txt`。
- **没有写全局文件：** 运行前后 `/root/.cache` 下没有新文件；`/root/.gitconfig` 的修改时间仍是 2026-09-02。
- **隔离：** 没有写 `$PROJ/data`。LISA 的代码与 venv 只在 `$PROJ/ext/lisa/`，产物只在本运行目录。

## 2. 执行记录

| 步骤 | 命令或做法 | 墙钟 | 结果 |
|---|---|---|---|
| 记账 | `diskcheck15.sh` | — | 见 §6 |
| 代码 | `git clone https://github.com/dvlab-research/LISA $PROJ/ext/lisa/LISA` | 数秒 | commit `3cb2d4301f1af4691bd4f3938335ef06e76f155a`（2024-12-31 "Release LISA++"），48 MB |
| 清内存盘 | `scan_cache_dir(...).delete_revisions(<Qwen3.5-0.8B 的 revision>)` | <1 s | 释放 1.770 GB，hf-hub 清空 |
| 核对体积 | HF API `model_info(files_metadata=True)`（`hf_repo_sizes.txt`） | 数秒 | LISA-7B-v1：10 个文件，16.122 GB（两片 `pytorch_model-*.bin`）；CLIP 全仓库 6.847 GB，其中 safetensors 一种格式 1.711 GB；revision 均与 API 的 sha 相同 |
| 下载 | `p18_download.py`（项目 venv，`HF_HUB_CACHE=/dev/shm/visint/hf-hub`；CLIP 用 `allow_patterns=["*.json","*.txt","model.safetensors"]`） | LISA 31.1 s，CLIP 3.9 s | 逐文件核对：LISA 10/10、16.122 GB；CLIP 8/8、1.714 GB |
| venv | `$ROOT/env/bin/python -m venv --system-site-packages $PROJ/ext/lisa/venv` | 数秒 | Python 3.12.13，继承 torch 2.13.0+cu130、torchvision 0.28.0+cu130 |
| 依赖 | 三步，见下 | 约 1 min | 见下 |
| 冒烟第 1 次 | `run_smoke.sh`（chat.py 流程，未加修法） | 8.2 s | 失败：CUDA device-side assert |
| 冒烟第 2 次 | 同上，加 `CUDA_LAUNCH_BLOCKING=1`（诊断，不是修法） | 7.8 s | 失败，得到准确位置：`llava_arch.py:187` 的 `embed_tokens(cur_input_ids[:image_token_start])` |
| 诊断 | CPU 上只加载分词器（`diag_tokens.txt`） | — | 见 §4 |
| 冒烟第 3 次 | `run_smoke_fix.sh`（加 `--tokenizer-special-str`，唯一一次修法） | 8.5 s | 成功，`smoke.json` |
| 收尾 | `delete_revisions(<LISA>, <CLIP>)` | <1 s | 释放 17.836 GB，hf-hub 清空 |

### 2.1 依赖安装（`install.log`；`PIP_NO_CACHE_DIR=1`、`TMPDIR=$PROJ/tmp`；全部 `--only-binary :all:`）

**安装前的核对**（`logs/binary_availability.txt`）：逐个用 `pip install --dry-run --only-binary :all: --no-deps` 查 LISA 钉住的版本在 Python 3.12 下有没有二进制包。

- 没有：`tokenizers<0.14`（transformers 4.31.0 依赖）、`numpy==1.24.2`、`Pillow==9.4.0`、`pycocotools==2.0.6`。
- 有：其余钉住的版本。

**安装步骤：**

1. `pip install --only-binary :all: --no-deps transformers==4.31.0`
2. `pip install --only-binary :all: peft==0.4.0 einops==0.4.1 sentencepiece opencv_python==4.8.0.74 Requests==2.31.0 tqdm==4.64.1 numpy==1.26.4 scipy==1.11.2 packaging accelerate==0.21.0 huggingface-hub==0.16.4 'safetensors>=0.3.1,<0.5' regex pyyaml filelock`
3. 修正第 2 步：第 2 步没加 `--no-deps`，pip 为满足 peft 对 transformers 的依赖，把 transformers 升到了 4.35.2，并装了 tokenizers 0.15.2。这是我的失误。补救是 `pip install --only-binary :all: --no-deps --force-reinstall transformers==4.31.0` 和 `pip uninstall -y tokenizers`。

**结果（`pip_freeze_lisa.txt`）：**

- 钉住的版本：transformers 4.31.0、peft 0.4.0、einops 0.4.1、opencv-python 4.8.0.74、requests 2.31.0、tqdm 4.64.1、scipy 1.11.2、sentencepiece 0.2.2。
- 替代的版本：numpy 1.26.4、huggingface-hub 0.16.4、accelerate 0.21.0、safetensors 0.4.5。
- 继承自基础环境：Pillow 12.3.0、torch 2.13.0+cu130、torchvision 0.28.0+cu130。

**`pip check`** 报两项：

- `transformers 4.31.0 requires tokenizers, which is not installed`；
- `ml-dtypes 0.6.0 has requirement numpy>=2.0.0`。ml-dtypes 继承自基础环境，不在 LISA 的推理路径上。

**跳过与替代**（任务书 §2.3 要求写进 report）：

| LISA requirements 里的项 | 处理 | 原因 |
|---|---|---|
| torch==1.13.1、torchvision==0.14.1 | 跳过，继承基础环境的 2.13.0 / 0.28.0（cu130） | 与 CUDA 绑定（任务书） |
| bitsandbytes==0.41.1 | 跳过 | 只用于 4/8 bit 量化，bf16 推理用不到 |
| flash-attn（README 另装） | 跳过 | 训练用；推理路径上 MPT 的 flash_attn/triton 导入不会触发 |
| gradio、fastapi、uvicorn、markdown2、shortuuid、openai、ray | 跳过 | web 演示、服务与训练用；`chat.py` 的推理路径不 import 它们 |
| pycocotools==2.0.6 | 跳过 | 没有 cp312 二进制包；只在训练与数据集代码里用 |
| numpy==1.24.2 | 改装 1.26.4 | 1.24.2 没有 cp312 二进制包；1.26.4 是有 cp312 包的 1.x 版本。opencv 4.8.0.74 是对着 numpy 1.x 编译的，所以没用基础环境的 numpy 2.5.2 |
| Pillow==9.4.0 | 用基础环境的 12.3.0 | 9.4.0 没有 cp312 二进制包 |
| tokenizers（transformers 4.31 依赖 <0.14） | 不装 | 没有 cp312 二进制包，本机没有 rustc/cargo；`chat.py` 用 `use_fast=False`（慢分词器，sentencepiece）。这一项导致了第 1 次失败，见 §4 |
| accelerate、huggingface-hub、safetensors（requirements 里没有） | 加装 0.21.0、0.16.4、0.4.5 | `from_pretrained(low_cpu_mem_usage=True)` 需要 accelerate；transformers 4.31 要求 hub <1.0、safetensors ≥0.3.1。所选版本与 transformers 4.31 同期 |

## 3. 结果（`smoke.json` 原样摘录）

| 项 | 值 |
|---|---|
| ok | true（第 3 次，`tokenizer_special_str_fix: true`） |
| 版本 | Python 3.12.13，torch 2.13.0+cu130，transformers 4.31.0，numpy 1.26.4，cv2 4.8.0，CUDA 13.0，NVIDIA B200（capability 10.0） |
| 分词器 | len 32004；bos `<s>`=1，eos `</s>`=2，unk `<unk>`=0，pad `<unk>`=0；`[SEG]`=32003 |
| config 原值 | `vision_tower` = `mm_vision_tower` = `openai/clip-vit-large-patch14`（探针改成本地快照路径，见 §4） |
| 参数量 | 7,704,747,312 |
| 加载时间 | 5.29 s（从内存盘加载；含分词器、两片 .bin、CLIP、`.bfloat16().cuda()`） |
| 推理时间 | 0.82 s（预处理、`model.evaluate(max_new_tokens=512)`、mask 输出） |
| 显存峰值（`torch.cuda.max_memory_allocated`） | 加载后 15,444,875,776 B；推理时 17,192,384,000 B |
| 生成 | 63 个 token（含提示）；文本 `<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start> <im_end> Can you segment the cat on the left in this image? ASSISTANT: Sure, the segmentation result is [SEG] .</s>` |
| mask | 1 个，480×640，51,993 像素（0.1692），外接框 xyxy [12, 56, 315, 473] |
| 目检 | `000000039769_masked_img_0.jpg` 上红色区域覆盖图左侧的橘猫 |

- **第 1、2 次（失败）：** `smoke_run1.json`、`smoke_run2_cuda_launch_blocking.json`。
  - 加载都成功：5.55 s 与 5.31 s，加载后显存都是 15,444,875,776 B。
  - 失败都在 `stage=inference`。
  - 第 1 次的日志里，`vectorized gather kernel index out of bounds` 断言出现了 512 行（CUDA 线程逐条打印）。
- **nvidia-smi 采样：** 每 5 s 一次，但每次冒烟只有约 8 s，采样没抓到 GPU 上的进程。显存峰值只有上表 torch 的读数。

## 4. 与任务书假设不符的发现

- **第 1 次失败的根因（`diag_tokens.txt`，CPU 上只加载分词器）：**
  - 未加修法时分词器长度是 32007。比 `added_tokens.json` 的 4 个多出 3 个：
    - `"AddedToken(content='<s>', ...)"`=32004；
    - `"AddedToken(content='</s>', ...)"`=32005；
    - `<unk>`=32006。
  - bos_token_id=32004、eos_token_id=32005。提示的第 1 个 id 是 32004，而 embed_tokens 只有 32004 行（有效下标 0–32003）。
  - 来源：LISA-7B-v1 的 `tokenizer_config.json` 把 bos/eos/unk 写成 `{"__type": "AddedToken", "content": "<s>", ...}`。transformers 4.31 在 `is_tokenizers_available()` 为 False 时，用 `tokenization_utils_base.py` 里的 dataclass `AddedToken` 替身，它没有 `__str__`，于是 `str()` 得到的是整个 repr。
  - 在 LISA 官方环境里（`pip install -r requirements.txt` 会带上 tokenizers 0.13.x），`str(AddedToken)` 是 content。
- **修法（只用了一次）：**
  - 我先试过在 `AutoTokenizer.from_pretrained` 里显式传字符串的 bos/eos/unk。在 CPU 上核对无效：`special_tokens_map.json` 的字典仍然覆盖了它。这一步没有上 GPU。
  - 实际用的修法：`p19_lisa_smoke.py --tokenizer-special-str`，加载分词器之前执行 `transformers.tokenization_utils_base.AddedToken.__str__ = lambda self: self.content`（只在没装 tokenizers 时）。CPU 上核对：len 32004，bos/eos/unk/pad = 1/2/0/0，提示的 id 最大 32002。随后 GPU 上第 3 次运行成功。
  - 官方 README 的 Installation 只有 `pip install -r requirements.txt` 与 `pip install flash-attn --no-build-isolation`，没有与这个报错对应的修法。这个修法是我根据上面的根因定的。
  - 其他修法我没有试，例如在 `$PROJ` 下装 Rust 来编译 tokenizers 0.13.x。
- **vision_tower 的路径（与 chat.py 的唯一流程差别）：**
  - LISA-7B-v1 的 config 有 `train_mask_decoder: true`，`LISAForCausalLM.__init__` 因此忽略 `vision_tower` 参数，改用 `config.vision_tower`，即 Hub id `openai/clip-vit-large-patch14`（`model/LISA.py` 第 129–138 行）。
  - 内存盘上的 CLIP 是按固定 revision 下载的快照，没有 `refs/main`，离线（`HF_HUB_OFFLINE=1`）时解析不到这个 Hub id。
  - 探针先读 config，把 `vision_tower` 和 `mm_vision_tower` 改成快照的本地路径（同一份权重 `32bd642`），再 `from_pretrained(config=config, ...)`。这一步从第 1 次起就有，不算修法。
- **我在安装上的失误：** 依赖安装第 2 步没加 `--no-deps`，pip 把 transformers 换成了 4.35.2，并装了 tokenizers 0.15.2。第 3 步强制装回 4.31.0、卸掉 tokenizers。代价是多一次安装，约 30 s；最终的 venv 里是 4.31.0（`pip_freeze_lisa.txt`）。
- **运行时警告（原文见 `logs/smoke_run3_fix.log`）：**
  - `Specified kernel cache directory could not be created! ... $PROJ/tmp/xdg/torch/kernels`：我在 `run_smoke.sh` 里设了 `XDG_CACHE_HOME=$PROJ/tmp/xdg`，但这个目录不存在，torch 因此关掉了 kernel 缓存，没有写到别处。
  - transformers 4.31 的 `_register_pytree_node` FutureWarning、generation config 的 UserWarning、慢分词器的 legacy 提示。

## 5. 生成的文件

`experiments/P0/0015_lisa_smoke/`，各文件的说明见该目录的 `README.md`：

- 结果：`smoke.json`、mask 与叠加图、`smoke_run1.json`、`smoke_run2_cuda_launch_blocking.json`、`diag_tokens.txt`；
- 安装：`install.log`、`pip_freeze_lisa.txt`、`pip_freeze_lisa_before.txt`；
- 权重与代码：`hf_repo_sizes.txt`、`weights_download.json`、`lisa_commit.txt`；
- 记账：`disk_ledger.tsv`；
- `logs/`、`probe_scripts/`、`README.md`。

保留在服务器上、不入库的：

- `$PROJ/ext/lisa/LISA`（0.050 GB）；
- `$PROJ/ext/lisa/venv`（0.476 GB）；
- `$PROJ/logs/0015/`。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 00:12:49 | 0015 start | 39.272 | 3.066 | 3.944 |
| 00:14:52 | after delete qwen3.5-0.8b from shm | 39.222 | 1.296 | 3.994 |
| 00:18:03 | after download LISA+CLIP | 38.728 | 19.132 | 4.470 |
| 00:21:29 | after smoke | 38.726 | 19.132 | 4.471 |
| 00:21:30 | after delete LISA+CLIP from shm | 38.726 | 1.296 | 4.471 |

- **根盘：** 可用空间从 39.272 GB 降到 38.726 GB（−0.546 GB），主要是 venv 的 0.465 GB 增量和 LISA clone 的 0.050 GB。全程远高于 3 GB 红线。
- **`$PROJ`：** 从 3.944 GB 升到 4.471 GB（+0.527 GB），在 15 GB 份额以内（D28-8）。
- **内存盘：** 峰值 19.132 GB，其中 LISA 16.122 GB、CLIP 1.714 GB，其余是 NYU 点对渲染缓存 1.277 GB 和 xet 0.019 GB。低于 25 GB 上限。
- **结束时：** hf-hub 为空，`/dev/shm/visint` 是 1.296 GB。Qwen3.5-0.8B 已删，之后的任务需要时按 `weights_manifest.json` 恢复（`p12_verify_weights.py --download`）。
