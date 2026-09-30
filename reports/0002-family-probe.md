# RUN-0002 — Phase 0：Gemma 4 E2B-it 与 InternVL3.5-2B-HF 的事实探针

服务器 CC，b200，2026-09-30 06:38–07:00 UTC（值守模式下从 origin/main 自动接单）。只贴数字与事实，不做解读。事实汇总在 `docs/model_notes.md` §6（Gemma）、§7（InternVL）和 `docs/env_report.md` §8；原始输出在 `experiments/P0/0002_family_probe/raw/`。

## 0. 一句话观察（只陈述，不解读）

D1 仪器在两个家族上都与预期一致：
- Gemma 4 E2B-it：图在前 36/36 层逐位相同，指令在前 1/36（只有第 0 层）；
- InternVL3.5-2B-HF：图在前 29/29，指令在前 1/29。

两家族的原生框格式不同：Gemma 文档为 `box_2d`，y 在前，0–1000；InternVL 文档为 `<box>[[x1,y1,x2,y2]]</box>`，x 在前，0–1000。

嵌入行数：Gemma 没有超出 tokenizer 的预留行，`resize_token_embeddings` 会扩行；InternVL 有 257 个预留行，resize 会缩行，与 Qwen 相同。

根盘可用最低 5.907 GB；内存盘现有 16.77 GB（Qwen3.5-0.8B、Gemma 4 E2B-it、InternVL3.5-2B-HF 三个模型的权重）。

## 1. 前置校验

- `git pull --ff-only`：HEAD 为 8c010e4（本服务器 RUN-0001 的最后一个 commit）。`git log --oneline -1 -- prompts/server/0002-family-probe.md` = `f194813`（`[mgr] D9 ...; RUN-0002 issued ...`），`git merge-base --is-ancestor f194813 HEAD` 成立。
- 任务书写明不跑仓库代码、不需要测试。本次没有修改仓库里的任何代码或 config。
- 期间管理员推送了 `4731b08`（D11、D9-3 修订、N10）。只改了 DECISIONS/DECISIONS_NEEDED/RESEARCH_LOG，与本任务的文件不重叠；批次 1 push 前用 `pull --rebase` 同步。
- 本次的推送：`aa87b10`（批次 1：Gemma）；批次 2（InternVL 与本 report）为本 commit。

## 2. 执行记录

| 步骤 | 脚本 | 墙钟 | 结果 |
|---|---|---|---|
| 删除内存盘上的 Qwen3.5-9B | `huggingface_hub.scan_cache_dir().delete_revisions(c202236235…).execute()`（见 §4 第 1 条） | <1 s | 内存盘 21.087 → 1.771 GB；0.8B 完好 |
| 下载 Gemma 4 E2B-it @3e22461f65e8 | `snapshot_download(revision=…)` | 18.3 s | 10.279 GB → /dev/shm |
| 下载 InternVL3.5-2B-HF @3f301ffcf3dc | 同上 | 10.4 s | 4.715 GB → /dev/shm |
| §3.1–3.4 Gemma / InternVL | `p7_facts.py` | 12.6 s / 8.8 s | ok（Gemma 第一次运行因 config 的按层属性报错而中断，修复探针后重跑，见 §5） |
| §3.5 Gemma / InternVL | `p7_box.py` | 10.7 s / 8.5 s | ok |
| §3.6 Gemma / InternVL | `p7_taskmod.py` | 9.1 s / 6.4 s | ok |
| §3.7 Gemma infer / lora | `p7_mem.py` | 21.5 s / 12.9 s | ok，无 OOM（第一次因批量图像的嵌套格式报错，见 §5） |
| §3.7 InternVL infer / lora | `p7_mem.py` | 13.0 s / 10.0 s | ok，无 OOM |

GPU：NVIDIA B200。§3.7 期间 nvidia-smi 每 500 ms 采样一次：
- Gemma：util 最高 100% / 99%，均值 41.9% / 21.1%，显存最高 36836 / 51250 MiB，功耗最高 774 / 745 W；
- InternVL：util 最高 79% / 99%，均值 20.7% / 12.1%，显存最高 14440 / 31890 MiB，功耗最高 716 / 581 W。

（每组前一个数为 infer，后一个为 lora。）

警告（原样）：
- Gemma：`Kwargs passed to processor.__call__ have to be in processor_kwargs dict, not in **kwargs`，出现在传 `chat_template_kwargs=` 时。加载时没有警告。
- InternVL 加载时：`The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. ...`
- InternVL 传 `enable_thinking=` 或 `chat_template_kwargs=` 时，也会出现同样的 processor_kwargs warning。

## 3. 结果表（原样；逐层表与全部细节见 model_notes §6/§7）

| 项 | Gemma 4 E2B-it | InternVL3.5-2B-HF |
|---|---|---|
| LLM 层数 / hidden | 35 / 1536 | 28 / 2048 |
| 层类型 | full_attention=7, sliding_attention=28（sliding_window 512） | full_attention=28 |
| 其他结构 | KV 共享层 15–34（复用第 13/14 层的 KV）；PLE 每层 256 维；共享层 MLP 宽度 12288 | Qwen3，head_dim 128，16Q/8KV |
| 视觉：每个 LLM token 对应的像素 | 48×48（patch 16 × pool 3） | 28×28（448 px 的 tile → 256 token） |
| 嵌入行 / len(tokenizer) | 262144 / 262144（0 个预留行） | 151936 / 151679（257 个预留行） |
| 占位 token | `<unusedN>` ×6227，普通词表项，注册后才是单个 token（id 不变） | 无 |
| `<g0>` 的 id / 是否单个 token | 262144 / 是 | 151679 / 是 |
| `resize_token_embeddings(len(tok))` | 262144→262146（扩行，PLE 同步扩） | 151936→151681（缩行） |
| 思考开关 | `enable_thinking` kwarg，默认关 | 模板没有开关；靠 R1 system prompt |
| generation_config | do_sample=True, T=1.0, top_k=64, top_p=0.95；eos [1, 106, 50] | 无采样参数；eos 151645 |
| 固定分辨率的参数 | `max_soft_tokens` ∈ {70, 140, 280, 560, 1120}，默认 280 | `crop_to_patches`、`max_patches`（processor 默认 True / 12；preprocessor_config.json 写的是 false） |
| 448×448 / 640×480 / 1024×768 的视觉 token（默认设置） | 256 / 266 / 266 | 256 / 3328 / 3328 |
| LLM 位置编码 | 一维 0..L−1（视觉编码器内部另有二维 patch 坐标） | 一维 0..L−1 |
| output_hidden_states | 36 个张量 | 29 个张量 |

§3.5 原始输出（greedy，思考关）：

| 模型 | 提示词（出处） | 000000039769 "the cat on the left" | 000000000139 "the television" |
|---|---|---|---|
| Gemma | `detect {obj}`（ai.google.dev 官方示例 "detect person and cat"） | `'```json\n[\n  {"box_2d": [107, 13, 987, 496], "label": "the cat on the left"}\n]\n```<turn|>'` | `'```json\n[\n  {"box_2d": [394, 13, 621, 244], "label": "the television"}\n]\n```<turn|>'` |
| InternVL | `Please provide the bounding box coordinate of the region this sentence describes: <ref>{obj}</ref>`（InternVL 文档的 RefCOCO 示例） | `'<ref>the cat on the left</ref><box>[[24, 114, 492, 984]]</box><|im_end|>'` | `'<ref>the television</ref><box>[[16, 395, 224, 612]]</box><|im_end|>'` |

COCO GT：猫为像素 xyxy [17.5, 54.4, 318.8, 469.4]，换算 ×1000 后 x 先为 [27, 113, 498, 978]、y 先为 [113, 27, 978, 498]。左侧的 tv 为 [7.0, 167.8, 156.3, 262.6]，x 先为 [11, 394, 244, 617]、y 先为 [394, 11, 617, 244]。

§3.6：

| 运行 | 图在前：逐位相同的层 | 指令在前：逐位相同的层 | 指令在前：最后一层 cos（展平） / 逐 token cos 最小 | 图像 token 数 |
|---|---|---|---|---|
| Gemma bf16 | 36/36 | 1/36 | 0.951982 / 0.564533 | 266 |
| InternVL bf16 | 29/29 | 1/29 | 0.967872 / 0.186147 | 3328 |

§3.7（torch 峰值 allocated / reserved，GiB）：

| 测量 | Gemma（1089 视觉 token） | InternVL（1280 视觉 token） |
|---|---|---|
| 仅加载权重 | 9.508 / 9.602 | 4.376 / 4.67 |
| 推理 bs1，logits_to_keep=1 | 10.011 / 10.123 | 4.567 / 4.674 |
| 推理 bs1，full logits | 10.686 / 11.205 | 4.807 / 5.041 |
| 推理 bs16，logits_to_keep=1 | 16.625 / 17.912 | 6.801 / 7.445 |
| 推理 bs16，full logits | 27.414 / 35.193 | 10.629 / 13.318 |
| LoRA r=32 LLM 线性层，GC，bs8，前向+反向（第 2 次） | 46.607 / 49.207（可训练 52,666,368） | 28.821 / 30.293（可训练 34,865,152） |
| generate bs16×64，样本/s（3 次） | 7.593 / 7.612 / 7.604 | 18.806 / 18.771 / 18.888 |

## 4. 与任务书假设不符的发现

1. **任务书 §2 让"只删 `models--Qwen--Qwen3.5-9B` 这一个目录"，但照做腾不出空间**：
   - 这个 huggingface_hub 版本（1.33.0）的缓存用了**全缓存共享的 blob 仓**：`hf-hub/blobs/<xx>/<hash>` 加 `.refs` 引用清单。大文件实际存放在那里，`models--X/blobs/` 下只有指向它的符号链接和小文件。
   - `models--Qwen--Qwen3.5-9B` 目录本身只有 10.3 MB，只删它会留下 19.3 GB 的孤儿 blob。
   - 实际做法：用库自带的 `scan_cache_dir('/dev/shm/visint/hf-hub').delete_revisions('c202236235762e1c871ad0ccb60c8ee5ba337b9a').execute()` 删除。dry-run 预计释放 19,316,585,679 B，恰好等于 9B 总量减去与 0.8B 共享的 tokenizer blob（12,807,982 B）。
   - 执行后，9B 的 4 个分片 blob 已删除，共享的 tokenizer blob 保留（`.refs` 只剩 0.8B）；0.8B 可以用 `local_files_only` 完整解析。删除前后的占用在 `raw/shm_before_delete.txt` 和 `raw/shm_after_delete.txt`。
2. **Gemma 4 的 config 不允许全局读取按层属性**：`config.text_config.head_dim` 会抛 `AmbiguousGlobalPerLayerAttributeError`，要用 `config.per_layer_config[i].head_dim`（sliding 层 256，full 层 512）。
3. **Gemma 没有预留嵌入行**：嵌入行数 = len(tokenizer) = 262144，新 token 需要 resize。resize 会**扩行**，并同步扩展 PLE 的 `embed_tokens_per_layer`，不会缩行。
   - 另有 6227 个 `<unusedN>`。直接写 `<unused0>` 会被切成 4 个 token；先 `add_tokens(['<unused0>',...])` 注册后，它是单个 token，id 仍为 6，len(tokenizer) 不变。
   - D9-3（经 4731b08 修订）的"复用预留行、从不缩行"是按 Qwen 写的。Gemma 的做法由管理员裁定。
4. **Gemma 的 generation_config 默认是采样**（do_sample=True, T=1.0, top_k=64, top_p=0.95），不显式设 greedy 就会随机生成。
5. **Gemma 的 processor 批处理要求图像按样本嵌套**（`images=[[img1], [img2], …]`），平铺的列表会报 `Received inconsistently sized batches of images (1) and text (16)`。
6. **InternVL 的两处默认值相反**：`preprocessor_config.json` 写 `crop_to_patches: false`，但 `InternVLProcessorKwargs._defaults` 是 `crop_to_patches: True`，实际生效的是后者。默认设置下 640×480 → 13 个 tile，3328 个视觉 token。
7. **InternVL 的 resize 同样缩行**（151936→151681）。它有 257 个预留行，新 id 落在其中。
8. **InternVL 的模板没有思考开关**：`enable_thinking=` 只触发 warning，不改变渲染。思考模式靠 card 给出的 R1 system prompt。
9. **三家族原生框格式的轴序不同**（只陈述文档）：
   - Qwen3.5：自己没有文档，Qwen3-VL 的 cookbook 为 `bbox_2d`，0–1000；
   - Gemma：`box_2d`，**y 在前**，0–1000；
   - InternVL：`<box>[[x1, y1, x2, y2]]</box>`，**x 在前**，0–1000。
10. **Gemma 文档的检测示例有两种写法**：`"detect person and cat"`，以及 `"detect person and car, output only ```json"`。本次用的是前一种，即 `detect {obj}`。
11. §3.7 的"约 1K 视觉 token"：
    - Gemma 用 1120 档，1024² 的图得到 1089 个 token；
    - InternVL 对 1024² 的方图只有 256、1280、2560 三档可选，取 1280（`crop_to_patches=True, max_patches=4`）。

## 5. 自己造成的错误或浪费

1. `p7_facts.py` 第一次在 Gemma 上运行，读 `text_config.head_dim` 时抛 `AmbiguousGlobalPerLayerAttributeError`（第 4 节第 2 条）。改为安全读取，并补读 per_layer_config 后重跑。代价：约 9 s 与一次模型加载。
2. `p7_mem.py` 第一次在 Gemma 上运行，infer 与 lora 都因批量图像没有按样本嵌套而报错（第 4 节第 5 条）。修复后重跑。代价：约 18 s。
3. 同一轮里第一版 `nvidia-smi` 的采样文件被第二轮覆盖了；marks 文件也已重置，只保留成功的那一轮。

## 6. 生成文件清单与磁盘收支

**仓库内**：
- `docs/model_notes.md`：§6（Gemma）、§7（InternVL）；
- `docs/env_report.md`：§8.1–8.3；
- `experiments/P0/0002_family_probe/`：README、`probe_scripts/`、`raw/`（单个文件都小于 5 MB）；
- `reports/0002-family-probe.md`（本文件）。

**仓库外**：`$PROJ/scripts/p7_*.py`、`$PROJ/logs/0002/`（原件）。

**内存盘**（`/dev/shm/visint/hf-hub`，按 D9-5 可随时删，目前保留供后续复用）：
- Qwen3.5-0.8B 1.770 GB
- Gemma 4 E2B-it 10.279 GB
- InternVL3.5-2B-HF 4.715 GB
- 合计 16.765 GB（上限 25 GB）

**磁盘收支**（`raw/disk_ledger.tsv`；GB = 1e9 字节）：

| 时刻 (UTC) | 步骤 | `df /` 可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 06:38:55 | 0002 start | 5.917 | 21.087 | 0.478 |
| 06:40:06 | before delete Qwen3.5-9B from shm | 5.917 | 21.087 | 0.478 |
| 06:40:07 | after delete Qwen3.5-9B from shm | 5.917 | 1.771 | 0.478 |
| 06:40:28 | before download google/gemma-4-E2B-it | 5.917 | 1.771 | 0.478 |
| 06:40:46 | after download google/gemma-4-E2B-it | 5.917 | 12.050 | 0.478 |
| 06:40:46 | before download OpenGVLab/InternVL3_5-2B-HF | 5.917 | 12.050 | 0.478 |
| 06:40:57 | after download OpenGVLab/InternVL3_5-2B-HF | 5.917 | 16.765 | 0.478 |
| 06:54:33 | before 0002 batch1 commit | 5.912 | 16.765 | 0.481 |
| 06:57:23 | before 0002 report | 5.907 | 16.765 | 0.481 |

- 根盘：本任务开始时可用 5.917 GB，最低 5.907 GB（红线 3 GB）。`$PROJ` 从 0.478 GB 增加到 0.481 GB，即本任务新增约 3 MB（上限 1 GB）。权重下载全部落在内存盘，根盘可用量没有变化。
- 内存盘：删除 9B 后为 1.771 GB，下载两个新模型后为 16.765 GB（上限 25 GB）。
- 没有删除、移动或修改任何非本项目文件。本次唯一的删除是本项目放在内存盘上的 Qwen3.5-9B 缓存，由任务书 §2 指定。
