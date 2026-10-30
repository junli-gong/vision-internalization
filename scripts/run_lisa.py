#!/usr/bin/env python
"""LISA-7B 外部对照的运行器（DEV-0075；D154；简报 §8.8）：在 LISA 的单独 venv 里，按官方 refer_seg 评测跑导出的清单，写原始输出。

硬约束第 1 条：LISA 把 SAM 接在 LLM 旁边（外接）。它只作对照单独运行；输出只经 scripts/lisa_w2.py score 计分、出表，绝不进入我们的
训练数据、标签、奖励或任何挑选（tests/test_lisa_isolation.py 静态核对）。

三步（"导出清单 → 运行 → 计分"；<out> = experiments/W2/<NNNN>_lisa_<节点>，最后一级必须是四位运行编号加 _lisa）：
  1. 我们的环境：python scripts/lisa_w2.py export --out <out> --data-root $DATA --test-approval D154
  2. LISA 的 venv：$VI/lisa/venv/bin/python scripts/run_lisa.py --out <out> --test-approval D154 \\
         --lisa-weights <LISA-7B-v1 快照目录> --clip-weights <CLIP 快照目录> [--resume] [--limit N]
  3. 我们的环境：python scripts/lisa_w2.py score --out <out> --data-root $DATA
本脚本只 import 标准库、numpy 与 src/lisa/common.py（两个环境共用的约定）；torch、transformers、cv2 与官方代码只在加载真模型时
import（LisaRunner）。

- 放行：--test-approval 必须恰是 D154（src/lisa/common.TEST_APPROVAL，依据 D154-2），否则退出码 2，什么都不写；放行时批准编号进
  raw/run_config.json，放行记录追加进 <out>/lisa-7b-v1/test_unlock.jsonl。
- 官方代码：third_party/lisa（dvlab-research/LISA @ 3cb2d430），先 check_vendor()（PROVENANCE.md 的逐文件 sha256），一个字节都不改；
  模块必须加载自 third_party/lisa（同名模块冲突就报错）。不在它的目录里运行（cwd 不变）。
- 权重：--lisa-weights、--clip-weights 必须恰是 configs/lisa_weights.json 的文件，逐个核对字节数与 sha256（约 17.8 GB，读一遍约
  1 分钟）。
- 协议（src/lisa/common.PROTOCOL 与 CITATIONS；third_party/lisa/PROVENANCE.md 的"协议"一节逐条注明出处）：官方 train_ds.py
  validate() 的口径。每句：用官方 ValDataset.__getitem__（refer_seg 分支，内存里只放这一张图、这一句）与 collate_fn 组输入——
  llava_v1 对话模板、"What is {sent} in this image? Please output segmentation mask."、回答直接给 "[SEG]."（teacher forcing，不生成）、
  CLIP 与 SAM 两侧的预处理；dict_to_cuda，图像转 bf16，model(**input_dict)（推理分支：LLM 一次前向、SAM 编码一次、解码一次），
  mask = pred_masks[0] > 0，尺寸为原图。
- 输出（<out>/lisa-7b-v1/raw/）：run_config.json（身份：协议、官方代码、权重、清单的 sha256、批准编号、运行器、环境与处理；续跑必须
  相同）、<划分>.jsonl（每句一行：mask 的未压缩 RLE、[SEG] 个数、前向次数、输入 token 数、用时；失败记 failure）、logs/run_<k>.json、
  logs/exceptions.jsonl。续跑（--resume）：末尾不完整的一行截掉，已写出的前缀跳过。--limit N：每个划分只跑到第 N 句（试跑）。
- 失败：没有 [SEG]（no_seg）、多于一个（multi_seg）、有 [SEG] 没有 mask（no_mask）、推理抛异常（异常类名）都写成一行，计分时按
  空 mask 计；连续 MAX_CONSECUTIVE_EXCEPTIONS 条异常就停（环境坏了），这一串不写出，续跑时重试。图像的 sha256 与清单不同直接停。
- --runner-factory MODULE:ATTR：换成测试的替身（不加载权重，返回固定 mask；开发机彩排）。替身的输出在身份里标明，计分要
  --allow-stand-in 才收。
退出码：0 = 跑完；2 = 用法、放行、身份、数据或环境不合格，或连续异常而停止（logs/run_<k>.json 记 error）；加载模型时的其他错误
照常抛出（带 traceback，退出码 1）；被中断时 logs/run_<k>.json 记 interrupted。
开发机彩排（DEV-0075）：按 PROVENANCE.md"环境"的首选做法在 visint 上建临时 venv，用小型随机 LISA 在 CPU 上走通了 LisaRunner
（彩排脚本只在自己的进程里把 .cuda() 换成空操作、把 SAM ViT-H 换成小编码器、EXPECTED_DEVICE 设为 cpu；不入库）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.lisa import common as C  # noqa: E402

SCRIPT_REL = "scripts/run_lisa.py"
RUNNER_LISA = "lisa"
EXPECTED_DEVICE = "cuda"  # 加载后审计的设备（官方前向写死了 .cuda()）
FIX_VISION_TOWER = "vision_tower_local_snapshot"
FIX_ADDEDTOKEN = "addedtoken_str_patch"
FIX_TOKENIZERS_HIDDEN = "tokenizers_hidden"
FIXES: dict[str, str] = {  # 同 scripts/external/lisa_e5_infer.py 的 FIXES（RUN-0015 §4），另加屏蔽 tokenizers
    FIX_VISION_TOWER: "config 的 vision_tower / mm_vision_tower 改成 --clip-weights 的本地目录（同一份权重）再 from_pretrained",
    FIX_ADDEDTOKEN: "没有 tokenizers 时，加载分词器之前给 transformers 4.31 的 AddedToken 替身补上 __str__ = content（RUN-0015 §4）",
    FIX_TOKENIZERS_HIDDEN: "import transformers 之前屏蔽继承来的新版 tokenizers（src/lisa/common.hide_tokenizers）",
}


class MaskFormatError(ValueError):
    """推理给出的 mask 不是原图尺寸的 bool 数组。"""


# ---------------------------------------------------------------------------
# 真模型：官方 refer_seg 评测的一条
# ---------------------------------------------------------------------------


class LisaRunner:
    """按官方 train_ds.py --eval_only 的路径加载 LISA-7B-v1、逐句前向。只在 LISA 的 venv 里、有 GPU 时可用。

    加载照 train_ds.py main() 第 119–174 行与 chat.py 第 114–127 行（逐条出处见 PROVENANCE.md 的"协议"与"偏离"）：
    慢分词器、pad = unk、[SEG] 与 <im_start>/<im_end> 已在分词器里（add_tokens 必须新增 0 个）；bf16 的 LISAForCausalLM
    （SAM 的权重来自 LISA 的检查点，缺的键必须为空）；initialize_vision_modules；整模型 bf16 上 GPU；
    default_conversation = conv_templates["llava_v1"]；eval()。"""

    def __init__(self, lisa_dir: Path, clip_dir: Path):
        self.fixes: list[str] = []
        self.info: dict[str, Any] = {"tokenizers": C.hide_tokenizers()}
        self.fixes.append(FIX_TOKENIZERS_HIDDEN)
        os.environ["HF_HUB_OFFLINE"] = "1"  # 只读本地目录，不联网
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("MPLBACKEND", "Agg")  # utils/refer.py 在模块顶层 import matplotlib.pyplot（推理不用）
        vdir = str(C.VENDOR_DIR)
        if vdir not in sys.path:
            sys.path.insert(0, vdir)

        import cv2
        import torch
        import transformers
        from PIL import Image as PIL_Image
        import transformers.tokenization_utils_base as tub
        from transformers import AutoTokenizer, CLIPImageProcessor

        import model.LISA as lisa_mod
        from model.llava import conversation as conversation_lib
        from model.llava.model.language_model.llava_llama import LlavaConfig
        from model.segment_anything.utils.transforms import ResizeLongestSide
        import utils.dataset as ds_mod
        import utils.utils as uu

        for mod, rel in ((lisa_mod, "model/LISA.py"), (conversation_lib, "model/llava/conversation.py"),
                         (ds_mod, "utils/dataset.py"), (uu, "utils/utils.py")):
            C.check_module_origin(mod, rel)
        if not torch.cuda.is_available():
            raise C.LisaError("没有 CUDA：官方 LISA 的前向里写死了 .cuda()，只能在 GPU 上跑")
        self.torch, self.uu, self.ds_mod, self.conversation_lib, self.PIL_Image = torch, uu, ds_mod, conversation_lib, PIL_Image
        self.info["versions"] = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                                 "numpy": np.__version__, "cv2": cv2.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if not transformers.utils.is_tokenizers_available():
            tub.AddedToken.__str__ = lambda self: self.content
            self.fixes.append(FIX_ADDEDTOKEN)
        tok = AutoTokenizer.from_pretrained(str(lisa_dir), cache_dir=None, model_max_length=C.PROTOCOL["model_max_length"],
                                            padding_side="right", use_fast=False)  # train_ds.py:119-125
        tok.pad_token = tok.unk_token  # train_ds.py:126
        n_seg_added = tok.add_tokens("[SEG]")  # train_ds.py:127
        seg_token_idx = int(tok("[SEG]", add_special_tokens=False).input_ids[0])  # train_ds.py:128
        n_mm_added = tok.add_tokens([uu.DEFAULT_IM_START_TOKEN, uu.DEFAULT_IM_END_TOKEN], special_tokens=True)  # train_ds.py:130-133
        if n_seg_added or n_mm_added:
            raise C.LisaError(f"分词器新增了 {n_seg_added} + {n_mm_added} 个 token：LISA-7B-v1 的分词器应当已含 [SEG]、<im_start>、"
                              "<im_end>（官方此后 resize_token_embeddings，这里不做）")
        config = LlavaConfig.from_pretrained(str(lisa_dir))
        self.info["config_vision_tower_original"] = {"vision_tower": config.vision_tower, "mm_vision_tower": config.mm_vision_tower}
        config.vision_tower = config.mm_vision_tower = str(clip_dir)
        self.fixes.append(FIX_VISION_TOWER)
        dtype = torch.bfloat16  # train_ds.py:146-150（precision 默认 bf16）
        model, loading = lisa_mod.LISAForCausalLM.from_pretrained(
            str(lisa_dir), config=config, torch_dtype=dtype, low_cpu_mem_usage=True, output_loading_info=True,
            vision_tower=str(clip_dir), seg_token_idx=seg_token_idx, use_mm_start_end=C.PROTOCOL["use_mm_start_end"])
        missing = sorted(loading.get("missing_keys") or [])
        if missing:
            raise C.LisaError(f"LISA 检查点缺 {len(missing)} 个键（如 {missing[:3]}）：SAM 等权重会是随机初始化")
        model.config.eos_token_id = tok.eos_token_id  # train_ds.py:154-156
        model.config.bos_token_id = tok.bos_token_id
        model.config.pad_token_id = tok.pad_token_id
        model.get_model().initialize_vision_modules(model.get_model().config)  # train_ds.py:161
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=dtype)  # train_ds.py:163（设备随后随整模型）
        conversation_lib.default_conversation = conversation_lib.conv_templates[C.PROTOCOL["conv_type"]]  # train_ds.py:172-174
        model = model.bfloat16().cuda()  # chat.py:126-127（视觉塔是子模块，随整模型上 GPU；audit_model 核对）
        model.eval()  # validate：model_engine.eval()
        torch.cuda.synchronize()
        n_emb = int(model.get_model().embed_tokens.num_embeddings)
        special = {k: getattr(tok, k + "_id") for k in ("bos_token", "eos_token", "unk_token", "pad_token")}
        if len(tok) != n_emb or any(v is None or not 0 <= int(v) < n_emb for v in special.values()):
            raise C.LisaError(f"分词器与 embed_tokens 不符：len={len(tok)}，特殊 token {special}，embed_tokens {n_emb} 行（RUN-0015 §4）")
        self.audit = audit_model(model, torch)
        ds = ds_mod.ValDataset.__new__(ds_mod.ValDataset)  # 官方 __init__ 第 329–333 行设的属性；REFER 的读取换成 set_item
        ds.base_image_dir = None
        ds.data_type = "refer_seg"
        ds.ds = "refcoco"
        ds.image_size = C.PROTOCOL["image_size"]
        ds.tokenizer = tok
        ds.transform = ResizeLongestSide(C.PROTOCOL["image_size"])
        ds.clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        self.ds, self.model, self.tok, self.seg_token_idx = ds, model, tok, seg_token_idx
        self.info.update({"load_s": round(time.perf_counter() - t0, 3), "peak_mem_after_load_bytes": int(torch.cuda.max_memory_allocated()),
                          "n_params": int(sum(p.numel() for p in model.parameters())), "seg_token_idx": seg_token_idx,
                          "tokenizer": {"len": len(tok), **{k: int(v) for k, v in special.items()}}, "embed_tokens_rows": n_emb,
                          "missing_keys": missing, "unexpected_keys": sorted(loading.get("unexpected_keys") or [])[:20],
                          "audit": self.audit})
        torch.cuda.reset_peak_memory_stats()

    def set_item(self, row: Mapping[str, Any]) -> None:
        """官方 ValDataset 第 296–327 行由 REFER 建的 refer_seg_ds，这里只放这一张图、这一句（GT 不给：空标注 → 原图尺寸的零 mask）。"""
        image_id, ann_id = int(row["image_id"]), int(row["ann_id"])
        self.ds.refer_seg_ds = {
            "images": [{"file_name": row["image"], "id": image_id, "height": int(row["height"]), "width": int(row["width"])}],
            "annotations": {ann_id: {"segmentation": []}},
            "img2refs": {image_id: [{"image_id": image_id, "ann_id": ann_id, "sentences": [{"sent": row["sent"]}]}]},
        }

    def infer(self, row: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        self.set_item(row)
        item = self.ds[0]  # 官方 ValDataset.__getitem__
        # 官方 cv2.imread 读到的尺寸与 EXIF 方向（cv2 按 EXIF 旋转，我们的 GT 是原始像素方向）：只记录，照官方口径不处理
        image_hw = [int(x) for x in self.ds_mod.cv2.imread(row["image"]).shape[:2]]
        with self.PIL_Image.open(row["image"]) as im:
            exif_orientation = im.getexif().get(0x0112)
        batch = self.ds_mod.collate_fn([item], tokenizer=self.tok, conv_type=C.PROTOCOL["conv_type"],
                                       use_mm_start_end=C.PROTOCOL["use_mm_start_end"], local_rank=0)
        conversation = batch["conversation_list"][0]
        input_ids = batch["input_ids"]
        n_seg = int((input_ids == self.seg_token_idx).sum().item())
        input_dict = self.uu.dict_to_cuda(batch)  # train_ds.py:533
        input_dict["images"] = input_dict["images"].bfloat16()  # train_ds.py:537-539
        input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        check_dtypes(input_dict, self.model, torch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(**input_dict)  # train_ds.py:544-545
        torch.cuda.synchronize()
        forward_s = time.perf_counter() - t0
        pred_masks = out["pred_masks"]
        if len(pred_masks) != 1:
            raise RuntimeError(f"pred_masks 有 {len(pred_masks)} 组，应为 1（每批一张图）")
        output_list = (pred_masks[0] > 0).int()  # train_ds.py:549
        n_masks = int(output_list.shape[0])
        mask = None if n_masks == 0 else output_list[0].bool().cpu().numpy()
        return {"mask": mask, "n_seg": n_seg, "n_masks": n_masks, "conversation": conversation, "n_input_ids": int(input_ids.shape[1]),
                "n_visual_tokens": C.CLIP_PATCH_TOKENS, "image_hw_decoded": image_hw, "sam_resize": [int(x) for x in item[6]],
                "exif_orientation": None if exif_orientation is None else int(exif_orientation),
                "forwards": {"llm": 1, "sam_image_encoder": 1, "sam_mask_decoder": n_masks}, "forward_s": round(forward_s, 4)}

    def stats(self) -> dict[str, Any]:
        return {"peak_mem_inference_bytes": int(self.torch.cuda.max_memory_allocated())}


def audit_model(model: Any, torch: Any) -> dict[str, Any]:
    """加载后的一次审计：全部参数都应是 bf16、在 CUDA 上（官方 validate 的 bf16 口径）。首个不符即报模块名。"""
    dtypes: dict[str, int] = {}
    for name, p in model.named_parameters():
        key = f"{p.dtype}|{p.device.type}"
        dtypes[key] = dtypes.get(key, 0) + int(p.numel())
        if p.dtype != torch.bfloat16 or p.device.type != EXPECTED_DEVICE:
            raise C.LisaError(f"参数 {name} 是 {p.dtype} / {p.device}，应为 bf16 / {EXPECTED_DEVICE}")
    return {"param_dtype_device": dtypes}


def check_dtypes(input_dict: Mapping[str, Any], model: Any, torch: Any) -> None:
    """进前向之前：两路图像都是 bf16、与模型同一设备（入口断言，报出两个值）。"""
    p = next(model.parameters())
    for k in ("images", "images_clip"):
        x = input_dict[k]
        if x.dtype != torch.bfloat16 or x.device != p.device:
            raise C.LisaError(f"{k} 是 {x.dtype} / {x.device}，模型是 {p.dtype} / {p.device}：应都为 bf16、同一设备")


# ---------------------------------------------------------------------------
# 逐句
# ---------------------------------------------------------------------------


def check_mask(mask: Any, h: int, w: int) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (h, w):
        raise MaskFormatError(f"mask 应为 bool、形状 {(h, w)}，收到 {type(mask).__name__} {getattr(mask, 'dtype', None)} "
                              f"{getattr(mask, 'shape', None)}")
    return mask


def infer_row(runner: Any, row: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """一句 → (原始输出行, traceback 或 None)。图像的 sha256 与清单不同时抛 LisaError（停下，不写）。"""
    t0 = time.perf_counter()
    data = Path(row["image"]).read_bytes()
    if C.sha256_bytes(data) != row["image_sha256"]:
        raise C.LisaError(f"{row['id']}：图像 {row['image']} 的 sha256 与清单不同（数据变了？）")
    base = {"id": row["id"], "w2_split": row["w2_split"]}
    try:
        res = runner.infer(row)
        h, w = int(row["height"]), int(row["width"])
        n_seg = int(res["n_seg"])
        failure = None
        if n_seg == 0:
            failure = C.FAIL_NO_SEG
        elif n_seg > 1:
            failure = C.FAIL_MULTI_SEG
        elif res["mask"] is None:
            failure = C.FAIL_NO_MASK
        mask = None if failure else check_mask(res["mask"], h, w)
        hw = res.get("image_hw_decoded")
        rec = {**base, "status": C.STATUS_FAILED if failure else C.STATUS_OK, "failure": failure,
               "mask_rle": None if mask is None else C.mask_to_uncompressed_rle(mask),
               "n_mask_pixels": None if mask is None else int(mask.sum()),
               "n_seg": n_seg, "n_masks": int(res.get("n_masks", 0)), "seg_source": "teacher_forced", "generated_text": None,
               "conversation": res["conversation"], "n_input_ids": int(res["n_input_ids"]), "n_visual_tokens": int(res["n_visual_tokens"]),
               "forwards": dict(res["forwards"]), "image_hw_decoded": hw,
               "image_hw_mismatch": bool(hw is not None and list(hw) != [h, w]), "exif_orientation": res.get("exif_orientation"),
               "sam_resize": res.get("sam_resize"),
               "forward_s": float(res.get("forward_s", 0.0))}
        tb = None
    except C.LisaError:
        raise
    except Exception as e:  # noqa: BLE001 — 推理异常记为这一句的失败（异常类名），按空 mask 计
        rec = {**base, "status": C.STATUS_FAILED, "failure": type(e).__name__, "error": str(e)[:500], "mask_rle": None}
        tb = traceback.format_exc()[-4000:]
    rec["wall_s"] = round(time.perf_counter() - t0, 4)
    return rec, tb


# ---------------------------------------------------------------------------
# 身份与运行
# ---------------------------------------------------------------------------


def _load_attr(spec: str) -> Callable[..., Any]:
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise C.LisaError(f"--runner-factory 应写成 MODULE:ATTR，收到 {spec!r}")
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def run_identity(a: argparse.Namespace, *, code: Mapping[str, Any], weights: Mapping[str, Any] | None, meta: Mapping[str, Any],
                 manifest_sha: Mapping[str, str], runner: Any, stand_in: bool) -> dict[str, Any]:
    info = getattr(runner, "info", {}) or {}
    return {
        "experiment": C.EXPERIMENT, "system": C.SYSTEM, "script": SCRIPT_REL, "dev_task": "DEV-0075", "approval": C.TEST_APPROVAL,
        "note": "LISA-7B 外部对照（D154；简报 §8.8）：单独运行，输出只经 scripts/lisa_w2.py score 计分与出表，绝不进入我们的训练数据、"
                "标签、奖励或任何挑选（硬约束第 1 条）。",
        "protocol": dict(C.PROTOCOL), "citations": C.citation_ids(), "code": dict(code), "paper": C.PAPER,
        "weights": None if weights is None else dict(weights),
        "manifest": {"files_sha256": dict(manifest_sha), "splits": {s: {"n": e["n"], "ids_sha256": e["ids_sha256"]} for s, e in meta["splits"].items()},
                     "tiny": bool(meta.get("tiny")), "limit": meta.get("limit")},
        "runner": {"kind": "stand_in" if stand_in else RUNNER_LISA, "factory": a.runner_factory,
                   "fixes_applied": sorted(getattr(runner, "fixes", []) or []),
                   "fix_descriptions": {k: FIXES[k] for k in sorted(getattr(runner, "fixes", []) or []) if k in FIXES}},
        "env": {"versions": info.get("versions"), "tokenizers": info.get("tokenizers"), "seg_token_idx": info.get("seg_token_idx"),
                "tokenizer": info.get("tokenizer"), "n_params": info.get("n_params"), "audit": info.get("audit")},
        "max_consecutive_exceptions": C.MAX_CONSECUTIVE_EXCEPTIONS,
    }


def prepare_raw(out: Path, ident: Mapping[str, Any], resume: bool) -> None:
    d = C.raw_dir(out)
    p = d / C.RUN_CONFIG
    if d.exists() and any(d.iterdir()):
        if not resume:
            raise C.LisaError(f"{d} 已存在且非空；续跑请加 --resume")
        if not p.is_file():
            raise C.LisaError(f"{d} 里没有 {C.RUN_CONFIG}，不能续跑")
        old, new = json.loads(p.read_text(encoding="utf-8")), C.normalized(ident)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise C.LisaError(f"{p}：续跑的身份与已有的不同（{diff}），拒绝混写")
    else:
        d.mkdir(parents=True, exist_ok=True)
        C.write_json(p, ident)
    (d / "logs").mkdir(exist_ok=True)


Pending = list[tuple[str, dict[str, Any], str]]  # 还没写出的连续异常：(划分, 行, traceback)，跨划分累计


def run_split(runner: Any, out: Path, split: str, rows: list[dict[str, Any]], limit: int | None, status: dict[str, Any],
              pending: Pending, log: Callable[[str], None]) -> None:
    """一个划分：清单顺序逐句推理，写出即 fsync。异常行先挂起，下一句成功时一并写出（零星的异常照常记为失败）；连续
    MAX_CONSECUTIVE_EXCEPTIONS 句异常（可以跨划分）就停，这一串不写出（续跑时从它们重试）。"""
    path = C.raw_rows_path(out, split)
    ids = [r["id"] for r in rows]
    done = C.read_raw_prefix(path, ids, repair=True)
    stop = len(rows) if limit is None else min(len(rows), limit)
    st = status.setdefault(split, {"n": len(rows), "done_before": len(done), "written": 0, "failures": {}})
    for row in rows[len(done):stop]:
        rec, tb = infer_row(runner, row)
        if tb is None:
            flush(out, pending, status)
            _write(out, split, rec, None, st)
        else:
            pending.append((split, rec, tb))
            if len(pending) >= C.MAX_CONSECUTIVE_EXCEPTIONS:
                C.append_jsonl(C.raw_dir(out) / C.EXCEPTIONS_LOG,
                               [{"id": r0["id"], "traceback": tb0, "written": False} for _, r0, tb0 in pending])
                raise C.LisaError(f"连续 {len(pending)} 句推理异常（{pending[-1][1]['failure']}：{pending[-1][1].get('error')}），停止；"
                                  "这一串没有写出，续跑时重试")
        n = len(done) + st["written"]
        if n and n % 50 == 0:
            log(f"[LISA {split}] {n}/{len(rows)}")
    log(f"[LISA {split}] {len(done) + st['written']}/{len(rows)}（本次 {st['written']}；失败 {st['failures']}；挂起的异常 {len(pending)}）")


def flush(out: Path, pending: Pending, status: dict[str, Any]) -> None:
    for split, r0, tb0 in pending:
        _write(out, split, r0, tb0, status[split])
    pending.clear()


def _write(out: Path, split: str, rec: Mapping[str, Any], tb: str | None, st: dict[str, Any]) -> None:
    C.append_jsonl(C.raw_rows_path(out, split), [rec])
    if tb is not None:
        C.append_jsonl(C.raw_dir(out) / C.EXCEPTIONS_LOG, [{"id": rec["id"], "traceback": tb, "written": True}])
    st["written"] += 1
    if rec.get("failure"):
        st["failures"][rec["failure"]] = st["failures"].get(rec["failure"], 0) + 1


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help=f"本次运行的目录（{C.RUN_DIR_EXAMPLE}），里面要有 scripts/lisa_w2.py export 写的清单")
    p.add_argument("--test-approval", metavar="ID", help=f"W2 test 子集的批准编号，必须恰是 {C.TEST_APPROVAL}")
    p.add_argument("--lisa-weights", help="xinlai/LISA-7B-v1 的快照目录（configs/lisa_weights.json 的 revision）")
    p.add_argument("--clip-weights", help="openai/clip-vit-large-patch14 的快照目录（同上）")
    p.add_argument("--weights-manifest", default=str(C.WEIGHTS_MANIFEST_PATH), help="默认 configs/lisa_weights.json")
    p.add_argument("--split", help="只跑清单里的这一个划分（默认全部，按清单的顺序）")
    p.add_argument("--limit", type=int, help="每个划分只跑到第 N 句（试跑；之后 --resume 接着跑）")
    p.add_argument("--resume", action="store_true", help="续跑：身份必须相同，已写出的前缀跳过")
    p.add_argument("--runner-factory", help="MODULE:ATTR，返回替身运行器（测试与开发机彩排；不加载权重）")
    a = p.parse_args(argv)
    try:
        C.check_run_dir(a.out)
        C.check_approval(a.test_approval)
    except C.LisaError as e:
        p.error(str(e))
    if a.limit is not None and a.limit < 1:
        p.error("--limit 必须是正整数")
    if a.runner_factory is None and (a.lisa_weights is None or a.clip_weights is None):
        p.error("真模型要给 --lisa-weights 与 --clip-weights")
    return a


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    out = Path(a.out)
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    t_all = time.perf_counter()
    status: dict[str, Any] = {}
    run_log: dict[str, Any] | None = None
    run_path: Path | None = None
    try:
        meta, rows_by_split, manifest_sha = C.read_manifest(out)
        splits = list(rows_by_split)
        if a.split is not None:
            if a.split not in rows_by_split:
                raise C.LisaError(f"--split {a.split} 不在清单里（{splits}）")
            splits = [a.split]
        code = C.check_vendor()
        stand_in = a.runner_factory is not None
        weights = None
        wdoc = C.load_weights_manifest(a.weights_manifest)
        if not stand_in:
            weights = {"manifest": {"path": a.weights_manifest, "sha256": C.sha256_file(a.weights_manifest),
                                    "default": Path(a.weights_manifest).resolve() == C.WEIGHTS_MANIFEST_PATH.resolve()},
                       "lisa": C.verify_weights(a.lisa_weights, "lisa", wdoc), "clip": C.verify_weights(a.clip_weights, "clip", wdoc)}
            for k in ("lisa", "clip"):
                weights[k].pop("dir")  # 路径不进身份（换个挂载点续跑也认）
            runner = LisaRunner(Path(a.lisa_weights), Path(a.clip_weights))
        else:
            runner = _load_attr(a.runner_factory)(a)
        ident = run_identity(a, code=code, weights=weights, meta=meta, manifest_sha=manifest_sha, runner=runner, stand_in=stand_in)
        prepare_raw(out, ident, a.resume)
        C.append_unlock(out, {"step": "run", "script": SCRIPT_REL, "splits": splits, "runner": ident["runner"]["kind"],
                              "manifest_rows_sha256": manifest_sha[C.MANIFEST_ROWS], "argv": argv})
        logs = C.raw_dir(out) / "logs"
        k = 1 + max([int(x.stem.split("_")[1]) for x in logs.glob("run_*.json") if x.stem.split("_")[1].isdigit()], default=0)
        run_path = logs / f"run_{k}.json"
        run_log = {"run": k, "argv": argv, "started_utc": C.now_utc(), "host": platform.node(), "pid": os.getpid(),
                   "cwd": os.getcwd(), "lisa_weights": a.lisa_weights, "clip_weights": a.clip_weights, "limit": a.limit,
                   "splits": splits, "runner_info": getattr(runner, "info", {}), "status": status, "state": "running"}
        C.write_json(run_path, run_log)
        pending: Pending = []
        for split in splits:
            run_split(runner, out, split, rows_by_split[split], a.limit, status, pending, log)
            C.write_json(run_path, run_log)
        flush(out, pending, status)  # 最后不足一串的异常照常写出
        run_log.update(state="done", wall_s=round(time.perf_counter() - t_all, 3), finished_utc=C.now_utc(),
                       stats=runner.stats() if hasattr(runner, "stats") else None)
        C.write_json(run_path, run_log)
        return 0
    except (C.LisaError, OSError, ValueError) as e:
        if run_log is not None and run_path is not None:
            run_log.update(state="error", error=repr(e), traceback=traceback.format_exc()[-6000:], wall_s=round(time.perf_counter() - t_all, 3))
            C.write_json(run_path, run_log)
        print(f"失败：{e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if run_log is not None and run_path is not None:
            run_log.update(state="interrupted", wall_s=round(time.perf_counter() - t_all, 3))
            C.write_json(run_path, run_log)
        raise


if __name__ == "__main__":
    sys.exit(main())
