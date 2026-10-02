#!/usr/bin/env python
"""LISA-7B-v1 在 E5 上的推理（外部对照系统那一侧；DEV-0019，D36-5）：读 run_e5.py --export-inputs 导出的输入，写外部预测目录。

硬约束 1：LISA 的输出只经外部预测目录、由 scripts/run_e5.py --system external 评分，绝不进入我们的数据、标签或奖励。
- 只读 --inputs、--lisa-src、两个权重目录；只写 --out。--out 不能与这几个目录重叠，也不能落在仓库的 src/ configs/ tests/
  scripts/ 或 $PROJ/data 下（启动前检查）。
- 在 LISA 自己的 venv 里运行（$PROJ/ext/lisa/venv，transformers 4.31），不在我们的项目 venv 里。所以本脚本不 import 我们
  仓库 src/ 下的任何模块（两个环境的依赖版本不同；tests/test_e5_lisa_infer.py 做静态检查）。外部预测格式的常量（标记文件名、
  字段名）与 a–d 的任务名在这里另写一份，由同一测试核对它们与 src/e5/external.py、src/e5/protocol.py 一致。
  顶层只 import 标准库与 numpy；torch、transformers、cv2 与 LISA 的代码只在加载真模型时 import（LisaRunner）。

用法（服务器）：
    $PROJ/ext/lisa/venv/bin/python scripts/external/lisa_e5_infer.py \\
        --lisa-src $PROJ/ext/lisa/LISA --inputs <export 目录> --out <外部预测目录> \\
        --lisa-weights <LISA-7B-v1 快照路径> --clip-weights <CLIP 快照路径> [--limit N] [--resume]
  - --inputs 是 run_e5.py --export-inputs 的目录：<inputs>/<任务>/inputs.jsonl 与 <inputs>/<任务>/images/*.png。
    a–d（rotation3d、viewpoint、occlusion、physics）逐条推理；e（imagery）没有图像，跳过，写进 run_config.json。
  - 两个权重目录必须是 HF 缓存的快照目录 .../models--<org>--<name>/snapshots/<revision>；repo 与 revision 必须等于 WEIGHTS
    （RUN-0015 下载并核对过的那一份）。--lisa-src 的 HEAD 必须是 LISA_COMMIT，且没有改过的已跟踪文件。
  - --limit N：每个任务只取 inputs.jsonl 的前 N 条（彩排）；N 记进运行身份，续跑必须相同。
  - --resume：--out 非空时必须给。标记文件与 run_config.json 必须与本次完全相同；每个 <任务>.jsonl 末尾不完整的一行截掉，
    已写出的 id 跳过，文件里同一 id 出现两次报错。
退出码：0 = 跑完；2 = 出错（用法、输入、身份或隔离不合格，连续 MAX_CONSECUTIVE_EXCEPTIONS 条异常而停止）。加载模型时的
其他错误照常抛出（带 traceback，退出码 1）；被中断时 logs/run_<k>.json 记 interrupted。

流程（与 RUN-0015 的冒烟脚本 experiments/P0/0015_lisa_smoke/probe_scripts/p19_lisa_smoke.py 同一条路径，即 LISA chat.py 的
推理流程，bf16、GPU，LISA 源码不改）：
- 加载：AutoTokenizer(use_fast=False, model_max_length=512, padding_side="right")，pad = unk；[SEG] 的 id；LlavaConfig →
  LISAForCausalLM.from_pretrained(low_cpu_mem_usage=True, torch_dtype=bf16) → initialize_vision_modules → .bfloat16().cuda()；
  CLIPImageProcessor、ResizeLongestSide(1024)。两处处理（FIXES，每次运行记进标记文件与 logs/run_<k>.json）：
  * vision_tower_local_snapshot（每次都用）：LISA-7B-v1 的 config 有 train_mask_decoder，LISAForCausalLM 因此忽略 vision_tower
    参数、改用 config.vision_tower（Hub id）；离线时 Hub id 解析不到，所以先把 config 的 vision_tower / mm_vision_tower 改成
    --clip-weights 的本地快照路径（同一份权重），再 from_pretrained。
  * addedtoken_str_patch（只在没装 tokenizers 时）：transformers 4.31 在没有 tokenizers 时用自带的 AddedToken 替身，它没有
    __str__，tokenizer_config 里的 bos/eos/unk 会被当成新词（32004/32005/32006），超出 embed_tokens。加载分词器之前给替身补上
    __str__ = content（等于装了 tokenizers 的官方环境的行为）。运行期打补丁，不改 LISA 或 transformers 的源文件。
  加载后的入口检查：[SEG] 是单个 token；分词器长度与 bos/eos/unk/pad 的 id 都在 embed_tokens 的行数以内；生成配置是
  greedy（do_sample 为假）。
- 每条：提示 = PROMPT_TEMPLATE.format(instruction=<导出的 E5 指令原文>)；套 chat.py 的对话模板（llava_v1，图像 token 前后加
  im_start / im_end）与图像预处理（cv2 读图转 RGB；CLIP 预处理；ResizeLongestSide(1024) + 归一化 + 右下补零到 1024）；
  model.evaluate(max_new_tokens=512)，num_beams=1、不采样（greedy，与冒烟脚本相同）。
- mask：取第一个 [SEG] 的 mask，阈值与 chat.py 相同（logit > 0），尺寸为原图（= 画布）。没有 [SEG] 记 failure "no_seg"，
  mask 全空记 "empty_mask"，推理抛异常记异常的类名（traceback 进 logs/exceptions.jsonl）。连续 MAX_CONSECUTIVE_EXCEPTIONS 条
  异常（环境坏了，如 CUDA 出错后的每一条都会失败）就停止，这一串异常不写出，续跑时重试；零星的异常照常写成 failure 行。

外部预测目录（src/e5/external.py 的 format_version 1）：
- EXTERNAL_PREDICTIONS.json（标记文件）：system lisa-7b-v1、LISA 的 commit、权重 repo 与 revision、提示模板、用到的处理；
  不含时间与路径（run_e5.py 把它的 sha256 记进评分的运行身份）。
- run_config.json：本次运行的完整身份（标记文件的内容、生成与预处理参数、mask 的取法与编码、处理过与跳过的任务、每个任务
  inputs.jsonl 的 sha256 与所选 id 的指纹、--limit）；不含时间与路径。
- <任务>.jsonl：每条一行，二选一：
  {"id", "mask_rle": {"size": [H, W], "counts": [整数...]}, "n_mask_pixels", "n_seg", "n_generated_tokens", "text", "prompt", "wall_s"}
  {"id", "failure": "no_seg" | "empty_mask" | <异常类名>, ...同上能给的字段, "error"（异常时）}
  mask 是未压缩 RLE，COCO 的列优先顺序（只用 numpy；LISA 的 venv 没有 pycocotools）。成功的行没有 failure 键。
  text 是新生成的部分（不含提示），decode 时不跳过特殊 token；n_seg 是生成里 [SEG] 的个数；wall_s 是这一条的墙钟
  （读图、预处理、生成、取 mask）。
- logs/run_<k>.json（每次调用一份：命令行、路径、版本、用到的处理、加载时间、显存峰值、各任务的条数与状态）、
  logs/exceptions.jsonl。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 约定（与 src/e5/external.py、src/e5/protocol.py 一致；本脚本不 import src/，由 tests/test_e5_lisa_infer.py 核对）
# ---------------------------------------------------------------------------

EXTERNAL_MARKER = "EXTERNAL_PREDICTIONS.json"
EXTERNAL_FORMAT_VERSION = 1
FIELD_ID = "id"
FIELD_MASK_RLE = "mask_rle"
FIELD_FAILURE = "failure"
INPUTS_FILE = "inputs.jsonl"  # src/e5/external.py 的 export_inputs 写的文件名
RUN_CONFIG = "run_config.json"

LISA_TASKS: tuple[str, ...] = ("rotation3d", "viewpoint", "occlusion", "physics")  # a–d
SKIPPED_TASKS: dict[str, str] = {"imagery": "e 类没有图像（画布只有文字描述），LISA 只分割图像里的物体，不适用；跳过"}

# ---------------------------------------------------------------------------
# 系统身份（RUN-0015 核对过的那一份）
# ---------------------------------------------------------------------------

SYSTEM = "lisa-7b-v1"
LISA_REPO_URL = "https://github.com/dvlab-research/LISA"
LISA_COMMIT = "3cb2d4301f1af4691bd4f3938335ef06e76f155a"  # experiments/P0/0015_lisa_smoke/lisa_commit.txt
WEIGHTS: dict[str, tuple[str, str]] = {  # 键 → (HF repo, revision)；experiments/P0/0015_lisa_smoke/weights_download.json
    "lisa": ("xinlai/LISA-7B-v1", "43c754eef75871fcc11c84d3930a402f7b0a754f"),
    "clip": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
}

# 提示：E5 指令原文 + LISA 训练时用的说法（D36-5；PREREGISTRATION.md 的 E5 草稿）
PROMPT_TEMPLATE = "{instruction} Please output segmentation mask."
# chat.py 的默认参数（LISA commit LISA_COMMIT）；MAX_NEW_TOKENS 同 chat.py 与冒烟脚本
CHAT_DEFAULTS: dict[str, Any] = {"precision": "bf16", "image_size": 1024, "model_max_length": 512, "use_mm_start_end": True,
                                 "conv_type": "llava_v1"}
MAX_NEW_TOKENS = 512
FAIL_NO_SEG = "no_seg"
FAIL_EMPTY_MASK = "empty_mask"
MAX_CONSECUTIVE_EXCEPTIONS = 3
FIX_VISION_TOWER = "vision_tower_local_snapshot"
FIX_ADDEDTOKEN = "addedtoken_str_patch"
FIXES: dict[str, str] = {
    FIX_VISION_TOWER: "config 的 vision_tower / mm_vision_tower 改成 --clip-weights 的本地快照路径（同一份权重）再 from_pretrained；"
                      "每次都用（RUN-0015 §4）",
    FIX_ADDEDTOKEN: "没装 tokenizers 时，加载分词器之前给 transformers 4.31 的 AddedToken 替身补上 __str__ = content；"
                    "只在没装 tokenizers 时用（RUN-0015 §4）",
}

SCRIPT_REL = "scripts/external/lisa_e5_infer.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
_REV_RE = re.compile(r"[0-9a-f]{40}")


class LisaInferError(RuntimeError):
    """用法、输入、身份或隔离不合格：报错退出（退出码 2）。"""


class ConsecutiveExceptions(LisaInferError):
    """连续 MAX_CONSECUTIVE_EXCEPTIONS 条推理异常：停止运行。"""


class MaskFormatError(ValueError):
    """推理给出的 mask 不是原图尺寸的 bool 数组（记为这一条的 failure）。"""


# ---------------------------------------------------------------------------
# mask → 未压缩 RLE（COCO：列优先展开，counts 从 0 的游程开始；只用 numpy）
# ---------------------------------------------------------------------------


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask 必须是二维，收到形状 {m.shape}")
    h, w = m.shape
    flat = m.astype(bool).ravel(order="F")
    if flat.size == 0:
        return {"size": [int(h), int(w)], "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat[0]:
        counts = np.concatenate(([0], counts))
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


# ---------------------------------------------------------------------------
# 推理接口：LisaRunner（真模型）；测试用替身实现同一接口
# ---------------------------------------------------------------------------


@dataclass
class InferOutput:
    text: str  # 新生成的部分，decode(skip_special_tokens=False)
    n_seg: int  # 生成里 [SEG] 的个数
    mask: np.ndarray | None  # 第一个 [SEG] 的 mask（bool，原图尺寸）；n_seg == 0 时为 None
    n_generated_tokens: int


class LisaRunner:
    """按 chat.py（= 冒烟脚本 p19_lisa_smoke.py）加载 LISA 并逐条推理。只在 LISA 的 venv 里、有 GPU 时可用。"""

    def __init__(self, lisa_src: Path, lisa_weights: Path, clip_weights: Path):
        os.environ["HF_HUB_OFFLINE"] = "1"  # 只读本地快照，不联网
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(lisa_src))
        os.chdir(lisa_src)  # 冒烟脚本的 cwd 是 LISA 仓库根；其余路径此前都已解析成绝对路径

        import cv2
        import torch
        import torch.nn.functional as F
        import transformers
        import transformers.tokenization_utils_base as tub
        from transformers import AutoTokenizer, CLIPImageProcessor

        from model.LISA import LISAForCausalLM
        from model.llava import conversation as conversation_lib
        from model.llava.mm_utils import tokenizer_image_token
        from model.llava.model.language_model.llava_llama import LlavaConfig
        from model.segment_anything.utils.transforms import ResizeLongestSide
        from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

        self.torch, self.cv2, self.F = torch, cv2, F
        self.conversation_lib, self.tokenizer_image_token = conversation_lib, tokenizer_image_token
        self.tokens = (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX)
        fixes: list[str] = []
        info: dict[str, Any] = {"versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                                             "transformers": transformers.__version__, "numpy": np.__version__,
                                             "cv2": cv2.__version__, "cuda": torch.version.cuda,
                                             "gpu": torch.cuda.get_device_name(0),
                                             "capability": list(torch.cuda.get_device_capability(0))}}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        info["tokenizers_available"] = bool(transformers.utils.is_tokenizers_available())
        if not info["tokenizers_available"]:
            tub.AddedToken.__str__ = lambda self: self.content
            fixes.append(FIX_ADDEDTOKEN)
        tok = AutoTokenizer.from_pretrained(str(lisa_weights), cache_dir=None, model_max_length=CHAT_DEFAULTS["model_max_length"],
                                            padding_side="right", use_fast=False)
        tok.pad_token = tok.unk_token
        seg_ids = tok("[SEG]", add_special_tokens=False).input_ids
        if len(seg_ids) != 1:
            raise LisaInferError(f"[SEG] 应当是单个 token，分词得到 {seg_ids}")
        seg_token_idx = int(seg_ids[0])
        dtype = torch.bfloat16
        config = LlavaConfig.from_pretrained(str(lisa_weights))
        info["config_vision_tower_original"] = {"vision_tower": config.vision_tower, "mm_vision_tower": config.mm_vision_tower}
        config.vision_tower = config.mm_vision_tower = str(clip_weights)
        fixes.append(FIX_VISION_TOWER)
        model = LISAForCausalLM.from_pretrained(str(lisa_weights), config=config, low_cpu_mem_usage=True,
                                                vision_tower=str(clip_weights), seg_token_idx=seg_token_idx, torch_dtype=dtype)
        model.config.eos_token_id = tok.eos_token_id
        model.config.bos_token_id = tok.bos_token_id
        model.config.pad_token_id = tok.pad_token_id
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=dtype)
        model = model.bfloat16().cuda()
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(device=0)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        self.transform = ResizeLongestSide(CHAT_DEFAULTS["image_size"])
        model.eval()
        torch.cuda.synchronize()
        # 入口检查：RUN-0015 第 1 次失败的根因（特殊 token 的 id 越过 embed_tokens）在这里先拦住，不等到 CUDA 断言
        n_emb = int(model.get_model().embed_tokens.num_embeddings)
        special = {k: getattr(tok, k + "_id") for k in ("bos_token", "eos_token", "unk_token", "pad_token")}
        if len(tok) > n_emb or any(v is None or not 0 <= int(v) < n_emb for v in special.values()):
            raise LisaInferError(f"分词器与 embed_tokens 不符：len={len(tok)}，特殊 token {special}，embed_tokens {n_emb} 行"
                                 "（RUN-0015 §4 的 AddedToken 问题？）")
        gc = model.generation_config
        if bool(getattr(gc, "do_sample", False)):
            raise LisaInferError(f"生成配置 do_sample={gc.do_sample}：任务书要求 greedy")
        self.model, self.tok, self.seg_token_idx = model, tok, seg_token_idx
        info.update({"fixes_applied": sorted(fixes), "load_s": round(time.perf_counter() - t0, 3),
                     "peak_mem_after_load_bytes": int(torch.cuda.max_memory_allocated()),
                     "n_params": int(sum(p.numel() for p in model.parameters())), "seg_token_idx": seg_token_idx,
                     "tokenizer": {"len": len(tok), **{k: int(v) for k, v in special.items()}}, "embed_tokens_rows": n_emb,
                     "generation_config": {"do_sample": bool(getattr(gc, "do_sample", False)),
                                           "num_beams": int(getattr(gc, "num_beams", 1) or 1)}})
        self.info = info
        torch.cuda.reset_peak_memory_stats()

    def _preprocess(self, x):
        """chat.py 的 preprocess：归一化，右下补零到 image_size 的正方形。"""
        torch = self.torch
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        size = CHAT_DEFAULTS["image_size"]
        return self.F.pad(x, (0, size - w, 0, size - h))

    def infer(self, image_path: str, prompt_text: str) -> InferOutput:
        torch, cv2 = self.torch, self.cv2
        image_tok, im_start, im_end, image_token_index = self.tokens
        conv = self.conversation_lib.conv_templates[CHAT_DEFAULTS["conv_type"]].copy()
        conv.messages = []
        prompt = image_tok + "\n" + prompt_text
        if CHAT_DEFAULTS["use_mm_start_end"]:
            prompt = prompt.replace(image_tok, im_start + image_tok + im_end)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        image_np = cv2.imread(image_path)
        if image_np is None:
            raise FileNotFoundError(f"cv2 读不出图像 {image_path}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        image_clip = (self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
                      .unsqueeze(0).cuda().bfloat16())
        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = self._preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().bfloat16()
        input_ids = self.tokenizer_image_token(prompt, self.tok, return_tensors="pt").unsqueeze(0).cuda()
        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                                         max_new_tokens=MAX_NEW_TOKENS, tokenizer=self.tok)
        torch.cuda.synchronize()
        seq = output_ids[0]
        n_in = int(input_ids.shape[1])
        if seq.shape[0] < n_in or not torch.equal(seq[:n_in], input_ids[0]):
            raise RuntimeError("generate 返回的序列不以输入 id 开头，无法切出新生成的部分")
        gen = seq[n_in:]
        gen = gen[gen != image_token_index]
        n_seg = int((gen == self.seg_token_idx).sum().item())
        masks = pred_masks[0]  # [该条里 [SEG] 的个数, H, W]，logit
        if int(masks.shape[0]) != n_seg:
            raise RuntimeError(f"mask 个数 {int(masks.shape[0])} 与生成里 [SEG] 的个数 {n_seg} 不同")
        mask = None if n_seg == 0 else masks[0].detach().float().cpu().numpy() > 0  # chat.py：pred_mask > 0
        return InferOutput(text=self.tok.decode(gen, skip_special_tokens=False), n_seg=n_seg, mask=mask,
                           n_generated_tokens=int(gen.numel()))

    def stats(self) -> dict[str, Any]:
        return {"peak_mem_inference_bytes": int(self.torch.cuda.max_memory_allocated())}


def git_source_info(lisa_src: Path) -> dict[str, Any]:
    """--lisa-src 的 HEAD 与改过的已跟踪文件（只读的 git 命令）。"""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(lisa_src), *a], capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = [x for x in git("status", "--porcelain", "--untracked-files=no").splitlines() if x.strip()]
    except (OSError, subprocess.CalledProcessError) as e:
        raise LisaInferError(f"读不出 {lisa_src} 的 git 信息：{e!r}") from e
    return {"commit": commit, "dirty_files": dirty}


# ---------------------------------------------------------------------------
# 输入、身份与隔离
# ---------------------------------------------------------------------------


def snapshot_identity(path: Path, key: str) -> dict[str, str]:
    """HF 缓存快照目录 → {repo, revision}，并核对等于 WEIGHTS[key]。"""
    p = Path(path)
    want_repo, want_rev = WEIGHTS[key]
    if p.parent.name != "snapshots" or not _REV_RE.fullmatch(p.name) or not p.parent.parent.name.startswith("models--"):
        raise LisaInferError(f"--{key}-weights 必须是 HF 缓存的快照目录 .../models--<org>--<name>/snapshots/<revision>，收到 {path}")
    repo = "/".join(p.parent.parent.name[len("models--"):].split("--"))
    if (repo, p.name) != (want_repo, want_rev):
        raise LisaInferError(f"--{key}-weights 是 {repo}@{p.name}，应为 {want_repo}@{want_rev}（RUN-0015 核对过的那一份）")
    return {"repo": repo, "revision": p.name}


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def forbidden_out_roots() -> tuple[Path, ...]:
    """--out 不能落在这些目录下：我们的代码、配置、测试，以及服务器上的数据目录 $PROJ/data（仓库在 $PROJ/repo）。"""
    return tuple(REPO_ROOT / d for d in ("src", "configs", "tests", "scripts")) + (REPO_ROOT.parent / "data",)


def check_isolation(out: Path, reads: Mapping[str, Path]) -> None:
    for name, p in reads.items():
        if _overlaps(out, p):
            raise LisaInferError(f"--out {out} 与 {name} {p} 重叠：本脚本只写 --out、只读 {name}")
    for root in forbidden_out_roots():
        if out == root or root in out.parents:
            raise LisaInferError(f"--out {out} 落在 {root} 下：外部预测不能写进我们的代码、配置或数据目录（硬约束 1）")


def read_inputs(inputs: Path, limit: int | None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """读 export 目录：a–d 每个任务的样本行（取前 limit 条）与身份（inputs.jsonl 的 sha256、所选 id 的指纹）。"""
    rows_of: dict[str, list[dict[str, Any]]] = {}
    ident: dict[str, Any] = {}
    for t in LISA_TASKS:
        p = inputs / t / INPUTS_FILE
        if not p.is_file():
            continue
        raw = p.read_bytes()
        rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        ids = [r.get(FIELD_ID) for r in rows]
        dup = sorted(i for i, c in Counter(ids).items() if c > 1)
        if dup:
            raise LisaInferError(f"{p}：id 重复 {dup[:5]}")
        task_dir = (inputs / t).resolve()
        for r in rows:
            sid, img = r.get(FIELD_ID), r.get("image")
            if not isinstance(sid, str) or not sid.startswith(t + "/") or r.get("task") != t:
                raise LisaInferError(f"{p}：行 {sid!r} 的 id 或 task 与目录 {t} 不符")
            if not isinstance(img, str) or not isinstance(r.get("instruction"), str) or not r["instruction"]:
                raise LisaInferError(f"{p}：{sid} 缺图像或指令（a–d 都有图像）")
            if task_dir not in (task_dir / img).resolve().parents:
                raise LisaInferError(f"{p}：{sid} 的图像路径 {img!r} 不在 {task_dir} 里")
            hw = r.get("canvas_hw")
            if not (isinstance(hw, list) and len(hw) == 2 and all(isinstance(v, int) and v > 0 for v in hw)):
                raise LisaInferError(f"{p}：{sid} 的 canvas_hw {hw!r} 不合规")
        sel = rows if limit is None else rows[:limit]
        rows_of[t] = sel
        ident[t] = {"inputs_sha256": hashlib.sha256(raw).hexdigest(), "n_rows": len(rows), "n_selected": len(sel),
                    # 与 src/e5/runner.py 的 ids_digest 同一算法：同样的子集 / 每档条数 / limit 下，等于评分 config.yaml 的
                    # data.sample_ids_sha256
                    "selected_ids_sha256": hashlib.sha256("\n".join(r[FIELD_ID] for r in sel).encode("utf-8")).hexdigest()}
    if not rows_of:
        raise LisaInferError(f"{inputs} 里没有 a–d 任何一个任务的 {INPUTS_FILE}（应为 run_e5.py --export-inputs 的目录）")
    return rows_of, ident


def build_identity(src_info: Mapping[str, Any], weights: Mapping[str, Any], fixes_applied: Sequence[str],
                   inputs_ident: Mapping[str, Any], limit: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = {"format_version": EXTERNAL_FORMAT_VERSION, "system": SYSTEM,
              "note": "LISA-7B-v1 外部对照（D36-5）：只经 scripts/run_e5.py --system external 评分，输出绝不进入我们的数据、"
                      "标签或奖励（硬约束 1）。由 " + SCRIPT_REL + "（DEV-0019）写出。",
              "lisa_repo": LISA_REPO_URL, "lisa_commit": src_info["commit"], "weights": dict(weights),
              "prompt_template": PROMPT_TEMPLATE, "fixes_applied": sorted(fixes_applied),
              "mask_encoding": "uncompressed RLE, COCO column-major (Fortran) order"}
    cfg = {"marker": marker, "script": SCRIPT_REL, "dev_task": "DEV-0019",
           "chat_py_defaults": dict(CHAT_DEFAULTS),
           "generation": {"path": "LISAForCausalLM.evaluate（chat.py）", "max_new_tokens": MAX_NEW_TOKENS, "num_beams": 1,
                          "do_sample": False},
           "mask": {"which": "第一个 [SEG]", "threshold": "logit > 0（chat.py）", "size": "原图 = 画布",
                    "failures": {"no_seg": "生成里没有 [SEG]", "empty_mask": "mask 全空", "<异常类名>": "推理抛异常"}},
           "fix_descriptions": {k: FIXES[k] for k in sorted(fixes_applied)},
           "tasks": [t for t in LISA_TASKS if t in inputs_ident], "skipped_tasks": dict(SKIPPED_TASKS),
           "inputs": dict(inputs_ident), "limit": limit, "max_consecutive_exceptions": MAX_CONSECUTIVE_EXCEPTIONS}
    return marker, cfg


def _norm(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False))


def _write_json(path: Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_out(out: Path, marker: Mapping[str, Any], cfg: Mapping[str, Any], resume: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise LisaInferError(f"{out} 已存在且非空；续跑请加 --resume")
        for name, new in ((EXTERNAL_MARKER, marker), (RUN_CONFIG, cfg)):
            p = out / name
            if not p.is_file():
                raise LisaInferError(f"{out} 里没有 {name}，不能续跑")
            old = json.loads(p.read_text(encoding="utf-8"))
            if old != _norm(new):
                diff = sorted(k for k in set(old) | set(_norm(new)) if old.get(k) != _norm(new).get(k))
                raise LisaInferError(f"{p}：续跑的身份与已有的不同（{diff}），拒绝混写")
    else:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / EXTERNAL_MARKER, marker)
        _write_json(out / RUN_CONFIG, cfg)
    (out / "logs").mkdir(exist_ok=True)


def read_done(path: Path, wanted: set[str]) -> set[str]:
    """已写出的 id：末尾不完整的一行截掉（写到一半被打断）；同一 id 两次、或不在本次输入里的 id 报错。"""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    ids = []
    for k, line in enumerate(raw[:keep].decode("utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            sid = json.loads(line).get(FIELD_ID)
        except json.JSONDecodeError as e:
            raise LisaInferError(f"{path}:{k + 1} 不是 JSON：{e}") from e
        if not isinstance(sid, str):
            raise LisaInferError(f"{path}:{k + 1} 缺 id")
        ids.append(sid)
    dup = sorted(i for i, c in Counter(ids).items() if c > 1)
    if dup:
        raise LisaInferError(f"{path}：id 重复 {dup[:5]}，拒绝续跑")
    extra = sorted(set(ids) - wanted)
    if extra:
        raise LisaInferError(f"{path}：有 {len(extra)} 个 id 不在本次输入里（如 {extra[:3]}）")
    if keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return set(ids)


def _append(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    if not recs:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# 逐条推理
# ---------------------------------------------------------------------------


def infer_one(runner: Any, row: Mapping[str, Any], task_dir: Path) -> tuple[dict[str, Any], str, str | None]:
    """一条样本 → (输出行, 类别 ok / no_seg / empty_mask / exception, traceback)。"""
    prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])
    h, w = row["canvas_hw"]
    t0 = time.perf_counter()
    try:
        res = runner.infer(str(task_dir / row["image"]), prompt)
        base = {"n_seg": int(res.n_seg), "n_generated_tokens": int(res.n_generated_tokens), "text": res.text,
                "prompt": prompt}
        if res.n_seg == 0:
            kind, rec = FAIL_NO_SEG, {FIELD_ID: row[FIELD_ID], FIELD_FAILURE: FAIL_NO_SEG, **base}
        else:
            m = res.mask
            if not isinstance(m, np.ndarray) or m.dtype != np.bool_ or m.shape != (h, w):
                raise MaskFormatError(f"mask 应为 bool、形状 {(h, w)}，收到 "
                                      f"{type(m).__name__} {getattr(m, 'dtype', None)} {getattr(m, 'shape', None)}")
            n_pix = int(m.sum())
            if n_pix == 0:
                kind, rec = FAIL_EMPTY_MASK, {FIELD_ID: row[FIELD_ID], FIELD_FAILURE: FAIL_EMPTY_MASK, **base}
            else:
                kind, rec = "ok", {FIELD_ID: row[FIELD_ID], FIELD_MASK_RLE: mask_to_uncompressed_rle(m), "n_mask_pixels": n_pix,
                                   **base}
        tb = None
    except Exception as e:  # noqa: BLE001 — 按任务书：推理异常记为这一条的 failure（异常类名）
        kind, tb = "exception", traceback.format_exc()[-4000:]
        rec = {FIELD_ID: row[FIELD_ID], FIELD_FAILURE: type(e).__name__, "error": str(e)[:500], "prompt": prompt}
    rec["wall_s"] = round(time.perf_counter() - t0, 4)
    return rec, kind, tb


def run_task(runner: Any, task: str, rows: Sequence[Mapping[str, Any]], inputs: Path, out: Path, run_k: int,
             log: Callable[[str], None], status: dict[str, Any]) -> None:
    """逐条推理并写出；status（调用方已放进运行日志）随进度原地更新，中途停止时也留下已写出的条数。"""
    path = out / f"{task}.jsonl"
    done = read_done(path, {r[FIELD_ID] for r in rows})
    todo = [r for r in rows if r[FIELD_ID] not in done]
    counts: Counter = Counter()
    pending: list[tuple[dict[str, Any], str]] = []  # 一串连续的异常：等到下一条不是异常才写出
    task_dir = inputs / task
    t_task = time.perf_counter()
    status.update({"task": task, "n_selected": len(rows), "n_done_before": len(done), "n_todo": len(todo), "counts": counts})

    def flush_pending() -> None:
        if not pending:
            return
        _append(path, [rec for rec, _tb in pending])
        _append(out / "logs" / "exceptions.jsonl", [{"run": run_k, "id": rec[FIELD_ID], "class": rec[FIELD_FAILURE],
                                                      "traceback": tb} for rec, tb in pending])
        counts["exception"] += len(pending)
        pending.clear()

    for k, row in enumerate(todo):
        rec, kind, tb = infer_one(runner, row, task_dir)
        if kind == "exception":
            pending.append((rec, tb or ""))
            if len(pending) >= MAX_CONSECUTIVE_EXCEPTIONS:
                _append(out / "logs" / "exceptions.jsonl", [{"run": run_k, "id": r[FIELD_ID], "class": r[FIELD_FAILURE],
                                                              "traceback": t, "written": False} for r, t in pending])
                raise ConsecutiveExceptions(f"{task}：连续 {len(pending)} 条推理异常（最后一条 {rec[FIELD_ID]}："
                                            f"{rec[FIELD_FAILURE]}: {rec['error']}），停止；这一串没有写出，续跑时重试")
            continue
        flush_pending()
        _append(path, [rec])
        counts[kind] += 1
        n_now = len(done) + k + 1
        if k == 0 or (k + 1) % 50 == 0 or k + 1 == len(todo):
            el = time.perf_counter() - t_task
            log(f"[lisa {task}] {n_now}/{len(rows)} {dict(counts)} {el / (k + 1):.2f} s/条")
    flush_pending()
    status["wall_s"] = round(time.perf_counter() - t_task, 3)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lisa-src", required=True, help="LISA 仓库（固定 commit），加进 sys.path")
    p.add_argument("--inputs", required=True, help="run_e5.py --export-inputs 的目录")
    p.add_argument("--out", required=True, help="外部预测目录（只写这里）")
    p.add_argument("--lisa-weights", required=True, help="xinlai/LISA-7B-v1 的 HF 快照目录")
    p.add_argument("--clip-weights", required=True, help="openai/clip-vit-large-patch14 的 HF 快照目录")
    p.add_argument("--limit", type=int, help="每个任务只取前 N 条（彩排）")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        p.error("--limit 必须是正整数")
    return a


def _default_runner(a: argparse.Namespace) -> LisaRunner:
    return LisaRunner(a.lisa_src, a.lisa_weights, a.clip_weights)


def main(argv: Sequence[str] | None = None, *, make_runner: Callable[[argparse.Namespace], Any] | None = None,
         source_info: Callable[[Path], Mapping[str, Any]] | None = None,
         log: Callable[[str], None] | None = None) -> int:
    """make_runner / source_info 默认是真模型与 git；测试换成替身。"""
    argv = sys.argv[1:] if argv is None else list(argv)
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    a = parse_args(argv)
    try:
        for name in ("lisa_src", "inputs", "out", "lisa_weights", "clip_weights"):
            setattr(a, name, Path(getattr(a, name)).resolve())
        reads = {"--inputs": a.inputs, "--lisa-src": a.lisa_src, "--lisa-weights": a.lisa_weights,
                 "--clip-weights": a.clip_weights}
        for name, p in reads.items():
            if not p.is_dir():
                raise LisaInferError(f"{name} {p} 不是目录")
        check_isolation(a.out, reads)
        weights = {"lisa": snapshot_identity(a.lisa_weights, "lisa"), "clip": snapshot_identity(a.clip_weights, "clip")}
        rows_of, inputs_ident = read_inputs(a.inputs, a.limit)
        if a.out.exists() and any(a.out.iterdir()) and not a.resume:
            raise LisaInferError(f"{a.out} 已存在且非空；续跑请加 --resume")
        src_info = dict((source_info or git_source_info)(a.lisa_src))
        if src_info["commit"] != LISA_COMMIT or src_info["dirty_files"]:
            raise LisaInferError(f"--lisa-src 的 HEAD 是 {src_info['commit']}（应为 {LISA_COMMIT}），改过的已跟踪文件 "
                                 f"{src_info['dirty_files'][:5]}：LISA 源码必须是固定 commit、不改")
        if (a.inputs / "imagery").is_dir():
            log(f"[lisa] imagery：{SKIPPED_TASKS['imagery']}")
        runner = (make_runner or _default_runner)(a)
        marker, cfg = build_identity(src_info, weights, runner.info["fixes_applied"], inputs_ident, a.limit)
        prepare_out(a.out, marker, cfg, a.resume)
    except LisaInferError as e:
        print(f"失败：{e}", file=sys.stderr)
        return 2

    run_k = len(list((a.out / "logs").glob("run_*.json"))) + 1
    run_path = a.out / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"run": run_k, "argv": argv,
                               "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "paths": {k: str(v) for k, v in {**reads, "--out": a.out}.items()},
                               "runner_info": runner.info, "fixes_applied": runner.info["fixes_applied"],
                               "status": "running", "tasks": []}
    _write_json(run_path, run_log)
    rc = 0
    t0 = time.perf_counter()
    try:
        for t in LISA_TASKS:
            if t in rows_of:
                run_log["tasks"].append(status := {"task": t})
                run_task(runner, t, rows_of[t], a.inputs, a.out, run_k, log, status)
        run_log["status"] = "done"
    except LisaInferError as e:  # 连续异常而停止、续跑时输出文件不合格
        print(f"失败：{e}", file=sys.stderr)
        run_log["status"], run_log["error"] = "aborted", str(e)
        rc = 2
    except BaseException as e:  # 代码出错或被中断（KeyboardInterrupt 等）：状态先落盘，再原样抛出
        run_log["status"] = "error" if isinstance(e, Exception) else "interrupted"
        run_log["error"] = repr(e)
        raise
    finally:
        run_log["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        run_log["wall_s"] = round(time.perf_counter() - t0, 3)
        if hasattr(runner, "stats"):
            run_log["runner_stats"] = runner.stats()
        _write_json(run_path, run_log)
    summary = {s["task"]: dict(s.get("counts", {})) for s in run_log["tasks"]}
    log(f"[lisa] {run_log['status']} {json.dumps(summary, ensure_ascii=False)} -> {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
