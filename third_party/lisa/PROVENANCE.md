# third_party/lisa 的出处（DEV-0075；D154）

**硬约束第 1 条**：LISA 把 SAM 接在 LLM 旁边，是"外接"的路线。它只作为简报 §8.8 所说"经项目负责人确认的 LISA 式对照系统"
（D154-2：型号 LISA-7B）**单独运行**；输出只用来计分与出表，绝不进入我们的训练数据、标签、奖励或任何挑选
（`tests/test_lisa_isolation.py` 静态核对）。

## 来源

- 仓库：https://github.com/dvlab-research/LISA
- commit：`3cb2d4301f1af4691bd4f3938335ef06e76f155a`（2024-12-31T02:34:22+08:00，"Release LISA++"；这个 commit 只改 README，
  代码与 RUN-0015、`scripts/external/lisa_e5_infer.py` 用的是同一份）
- 论文：Lai et al., *LISA: Reasoning Segmentation via Large Language Model*，CVPR 2024（arXiv 2308.00692）
- 许可证：Apache License 2.0（`LICENSE` 原样保留；`model/segment_anything/` 各文件带 Meta 的版权头，同为 Apache 2.0）
- 取法：在开发机的临时目录 `git clone`（HEAD 即上面的 commit，`git status` 为空），逐个文件原样拷贝；下表的 sha256 在拷贝前后
  各算一次，相同。没有在它的目录里运行任何东西（import 闭包是对源码做 AST 分析得到的）。临时目录用完已删。

## 文件与 sha256

只放推理与官方 refer_seg 评测需要的文件：
- `model/`：LISA 模型、LLaVA 的推理部分（含 `model/llava/model/__init__.py` 顶层 import 的 MPT 分支）、segment_anything；
- `utils/`：官方评测的预处理与模板 `utils/dataset.py`（`ValDataset`、`collate_fn`）及它在模块顶层 import 的文件；
- `train_ds.py`（官方评测 `validate()` 与加载流程）、`chat.py`（官方推理脚本）：**只作协议出处，不 import、不运行**；
- `LICENSE`。

没有放训练脚本以外的数据、权重合并脚本、web 演示（`app.py`）、`utils/grefcoco.py`（要 detectron2，不在 import 闭包里）、README 与图片。

| 文件 | sha256 |
|---|---|
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `model/LISA.py` | `74f6eb516ab62bb415bcc79d7826d358c3d8208f48053c170df8732994574f10` |
| `model/llava/__init__.py` | `52508db720555fae39081e92e8b7f26244df4153b5f3f6bc7bd4a647b7071310` |
| `model/llava/constants.py` | `2c5a8a659b3c5c1c70360808ae901511db67da8fbe331e1329cd754f6f3a0879` |
| `model/llava/conversation.py` | `82d38a918abb26d4ef43d83bddf19be7ccc5de50f8234488acd8404ef82d24fe` |
| `model/llava/mm_utils.py` | `16b60e77f513b87145b82b9c07dc1b4e2c61b4f57ebfa34443d6f03251244b93` |
| `model/llava/model/__init__.py` | `2efaf816f12025ff64a40cfe75b5d6634f6910baf2241fc1661c58c09c314b07` |
| `model/llava/model/language_model/llava_llama.py` | `200c1e2e2d54de62f4cf555adb065af0f88892403eaaaef3006d1a135bf12943` |
| `model/llava/model/language_model/llava_mpt.py` | `b43a44bad3876f6513b14be637d287dcf4023f9ae53f9bb6b2c92c01d8e78efb` |
| `model/llava/model/language_model/mpt/adapt_tokenizer.py` | `54dde02b76dee66eca6461768eaaf1a665794528cce8c215567a36bfabb6296f` |
| `model/llava/model/language_model/mpt/attention.py` | `3726f5c8689e711b7b0775d47f283acbfb4bbd63273ed85189b2faf9cc07ace0` |
| `model/llava/model/language_model/mpt/blocks.py` | `069c4fb8755333ccc0f4fa5005bd3406d25d10f9440e543fa8e21d3e87f0941d` |
| `model/llava/model/language_model/mpt/configuration_mpt.py` | `079b130bc2fdb78fec0615076366da5408c00f81aab60167245afe2b2f102456` |
| `model/llava/model/language_model/mpt/custom_embedding.py` | `6bf41b019618a80354d3f2126a7d9eae45a326219c8ca25a487808c6d8c68fa2` |
| `model/llava/model/language_model/mpt/flash_attn_triton.py` | `ccf6df1769c9ca5cd23bfbb4b255f11589aff579c09c35a1ebe43ad7340c609c` |
| `model/llava/model/language_model/mpt/hf_prefixlm_converter.py` | `26aca4d2b638f47a20326855ebc6eee15a85d9945e48c24bc43a3647eaa91890` |
| `model/llava/model/language_model/mpt/meta_init_context.py` | `6a3a95b61114b9336ab47774794b58b4c6f786f08678e58f81b20100f488f1ca` |
| `model/llava/model/language_model/mpt/modeling_mpt.py` | `9a9ec7a2a302f27172305a764c5dec6256671f493aed7c68336b47706bcc6130` |
| `model/llava/model/language_model/mpt/norm.py` | `fafd512850ab45e78879e99fa81ba621bf188d5a35735fc3e2ab47b629f38f2b` |
| `model/llava/model/language_model/mpt/param_init_fns.py` | `e6d9620a3240cbe5ecabcd2f127ed458a2a244e405cc9c6490d6e71757468aa9` |
| `model/llava/model/llava_arch.py` | `3cb5251f022c9adc4b9f98fba70e900f87e52268cfa87b4d7c55c5e903efdd86` |
| `model/llava/model/multimodal_encoder/builder.py` | `bbe462c71d11762ec0345b34c3366eb8a0f7cd2767c54e1807b925674335c3e9` |
| `model/llava/model/multimodal_encoder/clip_encoder.py` | `88c25d9a4fff65885e664e71f5e1a25b0441b044221379ea704d28e8889e2cd0` |
| `model/segment_anything/__init__.py` | `ebcae56a18711b7571ed0c99c6ac0ee744688f4b210c866f3eb8528a4030215c` |
| `model/segment_anything/automatic_mask_generator.py` | `6b320bdd37792986990a305adc78baf71a7e8e16e6feef8d2aa1bba5734df05c` |
| `model/segment_anything/build_sam.py` | `756da55a7c219c7d7c9ea7c67c283a36a54b266c1c8750047ea7efc73d0caee5` |
| `model/segment_anything/modeling/__init__.py` | `a87111db3eb9c90c942d218c7146e4e23dc6e75a4781f9658f5f2f047f59ce19` |
| `model/segment_anything/modeling/common.py` | `09398b9fb6da00a767c1d3214b7cc8d34085ce0322a61fb9486a6d5cea946716` |
| `model/segment_anything/modeling/image_encoder.py` | `3d03b4299069c08aa7b56a785c57f1844243b1862b76988d0ae12bdecfe59a50` |
| `model/segment_anything/modeling/mask_decoder.py` | `7dae9c90da300d0df0c415f5f5a814f72ac41c12c53db33eb6cdfcb5e3092f67` |
| `model/segment_anything/modeling/prompt_encoder.py` | `860343d95a1f0e419ec3b172b9ca673a4c92162e7357e9697adf43625c18e60a` |
| `model/segment_anything/modeling/sam.py` | `eaff5d3d0c9bf5e31071ccc830d68f136a94a3b67fb3421654f08d86648d0803` |
| `model/segment_anything/modeling/transformer.py` | `f7a65a95664b3d7c1d1658dcad12d3703110180f922d9ea1307051eee949fd1f` |
| `model/segment_anything/predictor.py` | `baac59008e5abe9b412cf0e4d4dcbcdd6ed7bf0a1c12c9e368cfc413c2a49aa6` |
| `model/segment_anything/utils/__init__.py` | `34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa` |
| `model/segment_anything/utils/amg.py` | `1b3e98a7246ac4ecf9263690492f9c085af5f7e391833be5b22863ea3a8fbbc8` |
| `model/segment_anything/utils/transforms.py` | `e22f2298806bc3dc5551558fbdfe14c5da2f70ab1325933507004ced187d7275` |
| `utils/conversation.py` | `d06243aa4a1765400ec6d13c4e975b897f164ceeec7a9daded555dca158ef693` |
| `utils/data_processing.py` | `31f13818a2a937b46b0d14cf78a0a3d8437873151167e8f255242eb212ba14f2` |
| `utils/dataset.py` | `f454643f729273b9a67920e5e2247e7474efc3ede40fd7d55e94506ab4232e5c` |
| `utils/grefer.py` | `8620877ed08dcfd9d7b4a9e03a2bd3d84c839351c7d3289fbdb5d7888705a9b7` |
| `utils/reason_seg_dataset.py` | `3367b63d55291901277f7a8354eb5e28c84a29b08fced13aa0285a9416e8580f` |
| `utils/refer.py` | `2a6dbad9c7127ec9ebd4546edd8a8d8392dfb991e6665c2e48a1f242bac33044` |
| `utils/refer_seg_dataset.py` | `3b3c2b01e1a3c6a66c032a8e2be8f30150e4d7eee1aca7fff78b20df842152d0` |
| `utils/sem_seg_dataset.py` | `41a86f8dae27e65cbe316b3a864b2551bfdfd712be0c9ad2f115c25b4d0b549d` |
| `utils/utils.py` | `8008ac5b7a01e1a9bf2fa642f60ebb756741bedaba47f85c3059fef54aaa1ecb` |
| `utils/vqa_dataset.py` | `9481293547ee63fccd9a9d14f5452dd7a6bea9065c5d10e850e3e7e2bee72ec3` |
| `train_ds.py` | `644328e22607bd7bb180158c62d23c8caf791ca47ccca7b158ffb304d2bd7f93` |
| `chat.py` | `ca4aaa29df12c9aa6a300acfb2079fdb1df9af49f7e0a4b7e4013143e8fb4426` |

49 个文件，共 448,753 字节。`src/lisa/common.check_vendor()` 每次运行与计分前现算并与这张表逐项比较（文件集合也要相同）；
`tests/test_lisa_vendor.py` 也核对。本目录里我们自己写的只有本文件与 `requirements_lisa.txt`。

## 我们改了什么

**官方文件一个字节都没改。** 我们这边的代码在 `scripts/run_lisa.py`（LISA 环境）与 `src/lisa/`（共用约定、导出、计分）。
调用方式与官方评测的不同（逐条；都不改官方代码）：

1. **数据**：官方 `ValDataset.__init__`（`utils/dataset.py` 第 296–327 行）用 REFER 读整个划分，按图组织。我们不读 REFER：
   清单由我们的环境按 W2 的取样导出（与 LLMSeg 相同的 1,500 句，`configs/w2_llmseg_subsets.json`），LISA 环境每句在内存里建
   一份只含这一张图、这一句的 `refer_seg_ds`（结构同第 301–327 行：`images`、`annotations`、`img2refs`），再调用官方
   `ValDataset.__getitem__` 与 `collate_fn`，一行不改。`__init__` 第 329–333 行设的属性（`ds`、`image_size`、`tokenizer`、
   `transform`、`clip_image_processor`）照设。图像是同一个 COCO train2014 文件；句子是 REFER 的 `sent` 字段（W2 样本的
   `fields.expression`），官方代码自己做 `strip().lower()`。
2. **GT 不进 LISA 环境**：标注给空的 `segmentation`，官方代码因此生成原图尺寸的零 mask（第 427–428 行），只用来确定 mask 还原的
   尺寸（`labels` 的形状，第 452 行；`model/LISA.py` 第 292 行）。计分在我们的环境里用 W2 的 GT 与 W2 的函数做。
3. **每批一句**：官方 `val_batch_size = 1` 张图（`train_ds.py` 第 340 行），同一张图的全部句子在一次前向里（右补齐 + attention
   mask）。我们每次前向只放一句：有效位置的隐藏状态不受补齐影响（因果注意力加 mask），只有核函数层面的数值差别；SAM 编码
   每句一次（官方每图一次）。
4. **加载**：照 `train_ds.py` 第 119–174 行，去掉只与训练有关的三步，并照 `chat.py` 第 126–127 行整模型 bf16 上 GPU：
   - LoRA 包装（第 176–215 行）：新建的 LoRA 的 B 矩阵初始化为 0，推理时增量恒为 0，省去；
   - `resize_token_embeddings`（第 217 行）：LISA-7B-v1 的分词器已含 `[SEG]`、`<im_start>`、`<im_end>`，`add_tokens` 新增 0 个
     （运行器核对），词表 32,004 = `embed_tokens` 行数，不需要；
   - `deepspeed.initialize`（第 305–317 行）：推理时 engine 的 forward 就是模块的 forward，省去；
   - SAM 的权重：官方 `--vision_pretrained` 先读原版 SAM ViT-H，再被 LISA 检查点里的 594 个 `visual_model.*` 键覆盖；我们同
     `chat.py` 不读原版 SAM，`from_pretrained` 的 missing_keys 必须为空（运行器核对）。
5. **视觉塔的路径**：LISA-7B-v1 的 config 有 `train_mask_decoder`，`LISAForCausalLM` 因此用 `config.vision_tower`（Hub id，
   `model/LISA.py` 第 138 行）。离线时解析不到，运行器先把 config 的 `vision_tower`、`mm_vision_tower` 改成本地 CLIP 快照目录
   （同一份权重，revision 钉住）再 `from_pretrained`（RUN-0015 的 `vision_tower_local_snapshot`）。
6. **分词器的两处环境处理**（见"环境"）：屏蔽继承来的新版 `tokenizers`；没有 `tokenizers` 时给 transformers 4.31 自带的
   `AddedToken` 替身补上 `__str__ = content`（RUN-0015 §4 的 `addedtoken_str_patch`）。都是运行期处理，不改 transformers 的源文件。
7. **版本**：torch 2.13.0+cu130（官方钉 1.13.1）、numpy 1.26.4（官方 1.24.2）等，见 `requirements_lisa.txt`。

## 协议（官方 refer_seg 评测 = 论文表格的口径）

官方 `README.md` 的 Validation 一节：`deepspeed train_ds.py --version=<LISA> --vision_pretrained=<SAM> --eval_only`，即
`validate()`。逐条（行号是 commit 3cb2d430 的；`src/lisa/common.CITATIONS` 记同一份，`tests/test_lisa_vendor.py` 核对每条在引入的
文件里逐字存在，且每条都出现在本节）：

| 项 | 官方的做法 | 出处 |
|---|---|---|
| 对话模板 | llava_v1：system "A chat between a curious human and an artificial intelligence assistant. …"，角色 USER / ASSISTANT，sep `" "`，sep2 `"</s>"` | `model/llava/conversation.py:355`；`train_ds.py:102`（conv_type 默认 llava_v1）；`train_ds.py:172`（default_conversation = conv_templates[conv_type]）；`utils/dataset.py:389` |
| 句子 | REFER 的 `sent`，`strip().lower()` | `utils/dataset.py:372` |
| 提问 | `<image>` + `"\n What is {sent} in this image? Please output segmentation mask."` | `utils/dataset.py:405` |
| 回答 | **评测直接给出 `"[SEG]."`**（teacher forcing，不生成；`[SEG]` 的位置取自输入） | `utils/dataset.py:409`；`model/LISA.py:187` |
| 图像 token | `<image>` 前后加 `<im_start>`、`<im_end>` | `utils/dataset.py:74`；`train_ds.py:98` |
| 分词 | 慢分词器，pad = unk，`tokenizer_image_token`，推理时不截断 | `train_ds.py:124`；`train_ds.py:126`；`utils/dataset.py:80`；`utils/dataset.py:137`；`train_ds.py:39`（model_max_length 512） |
| 读图 | `cv2.imread`，BGR 转 RGB | `utils/dataset.py:377`；`utils/dataset.py:378` |
| CLIP 一侧 | `CLIPImageProcessor.from_pretrained(vision_tower).preprocess`（openai/clip-vit-large-patch14：短边缩到 224、bicubic、中心裁 224、归一化） | `utils/dataset.py:414`；`utils/dataset.py:333`；`train_ds.py:42` |
| SAM 一侧 | `ResizeLongestSide(1024).apply_image`，`(x − mean) / std`（mean 123.675、116.28、103.53，std 58.395、57.12、57.375），右下补零到 1024 × 1024 | `utils/dataset.py:332`；`train_ds.py:38`；`utils/dataset.py:419`；`utils/dataset.py:421`；`utils/dataset.py:273`；`utils/dataset.py:274`；`utils/dataset.py:350` |
| 前向 | `dict_to_cuda`，两路图像转 bf16，`model_engine(**input_dict)`：推理分支 LLM 一次前向（`output_hidden_states`），SAM 单 mask 输出 | `train_ds.py:533`；`train_ds.py:538`；`train_ds.py:539`；`train_ds.py:545`；`model/LISA.py:201`；`model/LISA.py:269` |
| mask 还原 | 低分辨率 logits 双线性放大到 1024，裁掉补零，再双线性缩放到原图尺寸（= labels 的形状 = 标注的 height、width） | `model/segment_anything/modeling/sam.py:161`；`model/segment_anything/modeling/sam.py:168`；`model/segment_anything/modeling/sam.py:169`；`model/LISA.py:292`；`utils/dataset.py:428`；`utils/dataset.py:452` |
| 阈值 | `pred_masks[0] > 0`（logit > 0） | `train_ds.py:549` |
| 精度 | bf16（`--precision` 默认 bf16；模型以 bf16 加载） | `train_ds.py:33`；`train_ds.py:146`；`train_ds.py:148`；`chat.py:127` |
| 每批 | 1 张图 | `train_ds.py:340` |
| 视觉塔 | config 有 `train_mask_decoder` 时取 `config.vision_tower` | `model/LISA.py:138` |

计分（我们的环境，W2 的口径，与 RUN-0078 同一组函数）：GT 是 W2 的那一份（`src/e1/scoring.gt_mask`），`mask_inter_union` 与
`sample_iou`；cIoU = Σinter / Σunion，gIoU = 逐句 IoU 的均值（官方 `validate()` 第 552–573 行同义：`intersection_meter.sum /
union_meter.sum`、`acc_iou` 的均值，并集为 0 记 1；`src/metrics/seg.py` 已注明对齐）。没有 `[SEG]`、没有 mask、推理异常的句子
按空 mask 计，失败数进 metrics.json 的 failures。

**注意（不是偏离）**：官方 README 写明 v1 权重是在 ReasonSeg 的 train + val 上训的，复现论文验证集读数要用 v0；这与 RefCOCO 的
test 划分无关。LISA 的训练数据里有 refCOCO 系列的 train 划分，以及用 COCO 图像的 COCO-Stuff、LLaVA-Instruct-150k：RefCOCO 的
test 图像同在 COCO train2014 里，图像层面与训练数据有重叠（指代句与 mask 没有）。这是该方法自身的设定，照实记录，不处理。
官方读图用 `cv2.imread`，它按 EXIF 方向旋转图像，而 COCO 标注（我们的 GT）是原始像素方向；带 EXIF 方向的少数图像上两者不一致。
照官方口径不处理，原始输出记每句的 `exif_orientation` 与读到的尺寸，metrics.json 记 `n_exif_rotated`、`n_image_hw_mismatch`。

## 环境

LISA 要 transformers 4.31（新版的 LLaMA 前向、分词器与它的 LLaVA 代码不兼容），不能装进我们的 `$VI/env`（transformers 5.x）。
在 HPC 上建单独的 venv：

```bash
VI=/home/mail/chenyuma/visint
$VI/env/bin/python -m venv --system-site-packages $VI/lisa/venv
$VI/lisa/venv/bin/python -m pip install --no-cache-dir --only-binary :all: --no-deps -r $VI/repo/third_party/lisa/requirements_lisa.txt --dry-run
$VI/lisa/venv/bin/python -m pip install --no-cache-dir --only-binary :all: --no-deps -r $VI/repo/third_party/lisa/requirements_lisa.txt
$VI/lisa/venv/bin/python -m pip freeze --path $VI/lisa/venv/lib/python3.12/site-packages   # 只列 venv 自己装的，贴进报告
```

- **首选：`--system-site-packages` 继承 `$VI/env`**，只在 venv 里装 `requirements_lisa.txt` 钉住的包（`--no-deps`，逐个钉死）。
  理由：torch 2.13.0+cu130 与 torchvision 0.28.0+cu130 已在 `$VI/env` 里，且在 06 上核对过（RUN-0044），不必再下约 3–4 GB 的
  CUDA 轮子、也不占 Lustre；RUN-0015 在 B200 上用同样的做法（继承 torch 2.13.0+cu130）跑通了 LISA-7B-v1。pip 只会写进 venv，
  不会动 `$VI/env`（继承来的包在 venv 外，pip 不卸载）。venv 里的同名包（numpy 1.26.4、huggingface-hub 0.16.4、safetensors、
  accelerate、scipy 1.11.2、opencv 4.8）排在 sys.path 前面，盖过 `$VI/env` 的新版本。
- **与 B200 的 RUN-0015 不同的一处**：`$VI/env` 里有新版 `tokenizers`（transformers 5.x 要它），继承过来后 transformers 4.31
  在 import 时核对它的版本（要 <0.14）并报错；`tokenizers<0.14` 又没有 cp312 的二进制包。所以 `scripts/run_lisa.py` 在 import
  transformers 之前屏蔽它（`src/lisa/common.hide_tokenizers`：`sys.modules["tokenizers"] = None`，`find_spec` 因此返回 None），
  状态与 RUN-0015 的 venv（没装 tokenizers）相同：官方本来就用慢分词器（`use_fast=False`），再加 RUN-0015 的 `AddedToken` 处理。
- **备选：完整独立的 venv**（不继承）：`$VI/env/bin/python -m venv $VI/lisa/venv_full`，先从 `download.pytorch.org/whl/cu130`
  装 torch 2.13.0、torchvision 0.28.0（RUN-0044 约 1,069 s，约 3–4 GB），再装 `requirements_lisa.txt` 与继承清单里的包
  （pillow、scikit-image、pycocotools、regex、filelock、fsspec、packaging、pyyaml、typing-extensions 等）。首选在 import 或
  前向上出问题时用；运行器不变。
- 开发机彩排（2026-10-29）：在 visint（Python 3.12，torch 2.14、transformers 5.17、tokenizers 0.23）上按首选的做法建了临时
  venv，装 `requirements_lisa.txt`（另加与 torch 2.14 配套的 torchvision 0.29.0，visint 没有 torchvision），官方代码全部 import
  成功；用小型随机 LISA 在 CPU 上走通了 `scripts/run_lisa.py` 的真模型路径（`LisaRunner`：屏蔽 tokenizers、AddedToken 处理、
  官方 `ValDataset.__getitem__` 与 `collate_fn`、官方前向、阈值）。见 DEV-0075 的完成总结。

## 权重

`configs/lisa_weights.json`（repo、revision、逐文件字节数与 sha256），与 RUN-0015、`scripts/external/lisa_e5_infer.py` 同一对 revision：

- `xinlai/LISA-7B-v1` @ `43c754eef75871fcc11c84d3930a402f7b0a754f`：9 个文件，16,121,904,615 字节（两片 `pytorch_model-*.bin`
  共 16.12 GB；LLaMA-7B 与 SAM ViT-H 的权重都在里面）；
- `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41`：`config.json`、`preprocessor_config.json`、
  `model.safetensors`，1,710,545,415 字节；
- 合计约 17.83 GB。服务器在 HPC 登录节点用 `scripts/lisa_w2.py weights --download` 下到 `$VI/hf/hub`（本项目自己的 HF 缓存），
  逐文件核对；运行器每次启动再核对一遍（字节数与 sha256）。
