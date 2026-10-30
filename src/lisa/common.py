"""LISA-7B 外部对照（DEV-0075；D154）两个环境共用的约定。只 import 标准库与 numpy：LISA 的单独环境（transformers 4.31）也 import 它。

- 批准：test 子集只经负责人的批准编号 D154 放行（TEST_APPROVAL）。
- 布局（<out> 是本次运行的目录 experiments/W2/<NNNN_lisa…>，名字带运行编号且含 "lisa"，RUN_DIR_NAME）：
    <out>/lisa-7b-v1/test_unlock.jsonl                 放行记录（导出、运行各追加一行）
    <out>/lisa-7b-v1/inputs/manifest.json、manifest.jsonl  我们的环境导出的清单（id、图像路径与 sha256、句子、原图尺寸；不含 GT）
    <out>/lisa-7b-v1/raw/run_config.json、<划分>.jsonl、logs/   LISA 环境写的原始输出（mask 的 RLE、[SEG] 个数、前向次数、用时）
    <out>/lisa-7b-v1/refseg/lisa_7b/<划分>/            我们的环境计分后写的 W2 叶目录（scripts/mgr/w2_test_table.py 的 preds 能读）
- 官方代码：third_party/lisa（dvlab-research/LISA @ 3cb2d430），逐文件 sha256 记在 PROVENANCE.md，check_vendor() 运行前核对。
- 权重：configs/lisa_weights.json（repo、revision、逐文件字节数与 sha256），verify_weights() 运行前核对。
- 协议：官方 refer_seg 评测（train_ds.py validate() + utils/dataset.py 的 ValDataset 与 collate_fn），PROTOCOL 与逐条出处
  CITATIONS（tests/test_lisa_vendor.py 核对每条出处在引入的文件里逐字存在）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "third_party" / "lisa"
PROVENANCE_PATH = VENDOR_DIR / "PROVENANCE.md"
REQUIREMENTS_PATH = VENDOR_DIR / "requirements_lisa.txt"
OURS_IN_VENDOR: tuple[str, ...] = ("PROVENANCE.md", "requirements_lisa.txt")  # third_party/lisa 里我们自己写的文件
WEIGHTS_MANIFEST_PATH = REPO_ROOT / "configs" / "lisa_weights.json"

UPSTREAM_URL = "https://github.com/dvlab-research/LISA"
UPSTREAM_COMMIT = "3cb2d4301f1af4691bd4f3938335ef06e76f155a"  # 同 scripts/external/lisa_e5_infer.py（RUN-0015）
UPSTREAM_COMMIT_DATE = "2024-12-31T02:34:22+08:00"
UPSTREAM_LICENSE = "Apache-2.0"
PAPER = "Lai et al., LISA: Reasoning Segmentation via Large Language Model（CVPR 2024；arXiv 2308.00692）"

# ---------------------------------------------------------------------------
# 批准（D154）
# ---------------------------------------------------------------------------

# 依据：DECISIONS.md D154-2（项目负责人 2026-10-30 指示，原文 docs/lead_answers/2026-10-30-lang-table-and-freeze.md）：
#   统一主表新跑三项（E1 结构强制多边形、E3b 网格、LISA-7B），"这三项在 test 子集上运行的批准编号是 D154，写进运行器的放行检查，
#   同 D147"；LISA-7B 即简报 §8.8 所说经负责人确认的 LISA 式对照系统。预注册 U-1、U-2。别的编号一律不认。
TEST_APPROVAL = "D154"
TEST_UNLOCK_LOG = "test_unlock.jsonl"

# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------

SYSTEM = "lisa-7b-v1"  # W2 叶目录里"模型"那一级
SETTING = "lisa_7b"
TASK = "refseg"
EXPERIMENT = "W2-ext-LISA"
INPUTS_DIR = "inputs"
MANIFEST_META = "manifest.json"
MANIFEST_ROWS = "manifest.jsonl"
RAW_DIR = "raw"
RUN_CONFIG = "run_config.json"
EXCEPTIONS_LOG = "logs/exceptions.jsonl"
# --out 的最后一级：四位运行编号 + "_lisa"（CLAUDE.md"服务器驱动的写法"；含 "lisa" 让隔离扫描能认出它）
RUN_DIR_NAME = re.compile(r"[0-9]{4}_lisa[A-Za-z0-9_.-]*")
RUN_DIR_EXAMPLE = "experiments/W2/0099_lisa_hpcgpu06"


class LisaError(RuntimeError):
    """用法、身份、隔离或数据不合格（运行器以退出码 2 退出）。"""


class ApprovalError(LisaError):
    """test 的放行不合格（D154）。"""


class VendorError(LisaError):
    """引入的官方文件与 PROVENANCE.md 不符，或官方模块加载自别处。"""


class WeightsError(LisaError):
    """权重目录与 configs/lisa_weights.json 不符。"""


def check_run_dir(out: str | Path) -> None:
    name = Path(out).name
    if not RUN_DIR_NAME.fullmatch(name):
        raise LisaError(f"--out 的最后一级必须是四位运行编号加 _lisa（形如 {RUN_DIR_EXAMPLE}），收到 {str(out)!r}")


def check_approval(approval: str | None) -> None:
    if approval is None:
        raise ApprovalError(f"W2 的 test 子集只经负责人的批准编号放行：要给 --test-approval {TEST_APPROVAL}（D154）")
    if approval != TEST_APPROVAL:
        raise ApprovalError(f"--test-approval 必须恰是 {TEST_APPROVAL}（D154：LISA-7B 等三项新基线上 test 子集的批准编号），"
                            f"收到 {approval!r}")


def system_dir(out: str | Path) -> Path:
    return Path(out) / SYSTEM


def inputs_dir(out: str | Path) -> Path:
    return system_dir(out) / INPUTS_DIR


def raw_dir(out: str | Path) -> Path:
    return system_dir(out) / RAW_DIR


def raw_rows_path(out: str | Path, split: str) -> Path:
    return raw_dir(out) / f"{split}.jsonl"


def leaf_dir(out: str | Path, split: str) -> Path:
    """W2 的叶目录 <out>/<模型>/<任务>/<设置>/<划分>（同 src/w2/runner.w2_leaf）。"""
    return system_dir(out) / TASK / SETTING / split


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_unlock(out: str | Path, rec: Mapping[str, Any]) -> Path:
    """放行记录追加进 <out>/lisa-7b-v1/test_unlock.jsonl（同 DEV-0072）。"""
    p = system_dir(out) / TEST_UNLOCK_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"approval": TEST_APPROVAL, **rec, "unlocked_utc": now_utc()}, ensure_ascii=False) + "\n")
    return p


def read_unlock(out: str | Path) -> list[dict[str, Any]]:
    p = system_dir(out) / TEST_UNLOCK_LOG
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# 引入的官方代码
# ---------------------------------------------------------------------------

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def provenance_table(path: str | Path = PROVENANCE_PATH) -> dict[str, str]:
    """PROVENANCE.md 里"文件与 sha256"表（| `路径` | `<64 位十六进制>` |）→ {文件: sha256}。"""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            if m.group(1) in out:
                raise VendorError(f"{path}：{m.group(1)} 在表里出现两次")
            out[m.group(1)] = m.group(2)
    return out


def files_on_disk(root: str | Path = VENDOR_DIR) -> list[str]:
    """third_party/lisa 里的文件（相对路径），不含我们自己写的与 __pycache__。"""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.relative_to(root).as_posix() not in OURS_IN_VENDOR)


def check_vendor(root: str | Path = VENDOR_DIR) -> dict[str, Any]:
    """现算的逐文件 sha256 必须与 PROVENANCE.md 的表逐项相同，文件集合也相同（多一个、少一个都不行）。返回运行身份里 code 一节。"""
    root = Path(root)
    want = provenance_table(root / "PROVENANCE.md")
    have_files = files_on_disk(root)
    if sorted(want) != have_files:
        extra, missing = sorted(set(have_files) - set(want)), sorted(set(want) - set(have_files))
        raise VendorError(f"third_party/lisa 的文件与 PROVENANCE.md 的表不同：多 {extra}，缺 {missing}")
    have = {f: sha256_file(root / f) for f in sorted(want)}
    bad = sorted(f for f in want if have[f] != want[f])
    if bad:
        raise VendorError(f"{bad} 的 sha256 与 PROVENANCE.md 记的不同（官方文件被改过？）")
    return {"upstream": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "license": UPSTREAM_LICENSE, "dir": "third_party/lisa",
            "provenance_sha256": sha256_file(root / "PROVENANCE.md"), "n_files": len(have),
            "files_sha256_digest": sha256_bytes("\n".join(f"{f} {have[f]}" for f in sorted(have)).encode("utf-8")),
            "modified_files": []}


def check_module_origin(mod: Any, rel: str, root: str | Path = VENDOR_DIR) -> None:
    """官方模块必须加载自 third_party/lisa/<rel>（同名的别的模块先被 import 了就报错，不混用）。"""
    f = Path(getattr(mod, "__file__", "") or "").resolve()
    want = (Path(root) / rel).resolve()
    if f != want:
        raise VendorError(f"模块 {getattr(mod, '__name__', mod)!r} 加载自 {f}，不是 {want}（同名模块冲突）")


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------


def load_weights_manifest(path: str | Path = WEIGHTS_MANIFEST_PATH) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("lisa", "clip"):
        m = doc.get("models", {}).get(key)
        if not m or not re.fullmatch(r"[0-9a-f]{40}", str(m.get("revision"))) or not m.get("files"):
            raise WeightsError(f"{path}：models.{key} 缺 repo、revision 或 files")
    return doc


def weights_total_bytes(doc: Mapping[str, Any]) -> dict[str, int]:
    return {k: int(sum(f["bytes"] for f in m["files"])) for k, m in doc["models"].items()}


def verify_weights(directory: str | Path, key: str, doc: Mapping[str, Any], *, full_hash: bool = True) -> dict[str, Any]:
    """权重目录（HF 快照目录或同内容的本地目录）必须恰好是清单里的文件（不多不少，隐藏文件除外），字节数相同；full_hash 时 sha256
    也逐个相同。返回运行身份里的一节。"""
    d = Path(directory)
    m = doc["models"][key]
    if not d.is_dir():
        raise WeightsError(f"{key} 的权重目录 {d} 不存在")
    want = {f["path"]: f for f in m["files"]}
    have = sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                  if (p.is_file() or p.is_symlink()) and not any(part.startswith(".") for part in p.relative_to(d).parts))
    if sorted(want) != have:
        raise WeightsError(f"{d}：文件 {have} 与 configs/lisa_weights.json 的 {key}（{sorted(want)}）不同")
    digests = {}
    for rel, f in sorted(want.items()):
        p = d / rel
        if p.stat().st_size != int(f["bytes"]):
            raise WeightsError(f"{p}：{p.stat().st_size} 字节，清单是 {f['bytes']}")
        if full_hash:
            h = sha256_file(p)
            if h != f["sha256"]:
                raise WeightsError(f"{p}：sha256 {h[:16]}… 与清单的 {f['sha256'][:16]}… 不同")
            digests[rel] = h
    return {"repo": m["repo"], "revision": m["revision"], "dir": str(d), "n_files": len(want),
            "total_bytes": int(sum(f["bytes"] for f in m["files"])), "sha256_verified": bool(full_hash)}


# ---------------------------------------------------------------------------
# 协议（官方 refer_seg 评测 = 论文表格的口径）
# ---------------------------------------------------------------------------

PROTOCOL_NAME = "refer_seg_validate"
PROTOCOL: dict[str, Any] = {
    "name": PROTOCOL_NAME,
    "what": "官方 refer_seg 评测（论文表格的口径）：train_ds.py --eval_only 的 validate()，数据走 utils/dataset.py 的 ValDataset（refer_seg 分支）"
            "与 collate_fn；回答由评测直接给出 \"[SEG].\"（teacher forcing），LLM 一次前向，取 [SEG] 的隐藏状态经 SAM 解码",
    "conv_type": "llava_v1",
    "use_mm_start_end": True,
    "question": "<image>\\n What is {sent} in this image? Please output segmentation mask.",
    "answer": "[SEG].",
    "sentence": "REFER 的 sent 字段（我们 W2 样本的 fields.expression）.strip().lower()",
    "clip_preprocess": "CLIPImageProcessor.from_pretrained(vision_tower).preprocess(RGB uint8)：短边缩到 224（bicubic）、中心裁 224、归一化",
    "sam_preprocess": "ResizeLongestSide(1024).apply_image（长边 1024，PIL 双线性）→ (x − mean)/std（mean 123.675/116.28/103.53，"
                      "std 58.395/57.12/57.375）→ 右下补零到 1024×1024",
    "image_read": "cv2.imread → BGR 转 RGB",
    "mask": "SAM 低分辨率 logits → 双线性放大到 1024 → 裁掉补零 → 双线性缩放到原图尺寸（标注的 height、width）→ logit > 0",
    "precision": "bf16",
    "image_size": 1024,
    "model_max_length": 512,
    "vision_tower": "openai/clip-vit-large-patch14（config.vision_tower）",
    "batch": "一句一次前向（官方 val_batch_size = 1 张图，同一张图的全部句子在一次前向里；这里每批一句，见 PROVENANCE.md 的偏离 2）",
}
CLIP_PATCH_TOKENS = 256  # CLIP ViT-L/14 在 224×224 上的 16 × 16 个 patch（model/LISA.py 的 255 偏移就是它减去占位的 1 个）
CLIP_PIXELS = 224 * 224

# 逐条出处：(文件, 行号, 该行逐字含有的片段, 项)。行号是 commit 3cb2d430 的；tests/test_lisa_vendor.py 逐条核对。
CITATIONS: tuple[tuple[str, int, str, str], ...] = (
    ("utils/dataset.py", 372, 'sents.append(sent["sent"].strip().lower())', "句子：REFER 的 sent，strip 再 lower"),
    ("utils/dataset.py", 389, "conv = conversation_lib.default_conversation.copy()", "对话模板：模块的 default_conversation"),
    ("train_ds.py", 172, "conversation_lib.default_conversation = conversation_lib.conv_templates[", "default_conversation = conv_templates[conv_type]"),
    ("train_ds.py", 102, 'default="llava_v1"', "conv_type 默认 llava_v1"),
    ("model/llava/conversation.py", 355, "conv_llava_v1 = Conversation(", "llava_v1 模板（system、USER/ASSISTANT、sep ' '、sep2 '</s>'）"),
    ("utils/dataset.py", 405, '"\\n What is {} in this image? Please output segmentation mask.".format(', "提问模板（refer_seg，非 is_sentence）"),
    ("utils/dataset.py", 409, 'conv.append_message(conv.roles[1], "[SEG].")', "回答直接给出 \"[SEG].\"（teacher forcing，不生成）"),
    ("utils/dataset.py", 74, "DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN", "<image> 前后加 <im_start>、<im_end>"),
    ("train_ds.py", 98, '"--use_mm_start_end", action="store_true", default=True', "use_mm_start_end 默认真"),
    ("utils/dataset.py", 80, "tokenizer_image_token(prompt, tokenizer, return_tensors=\"pt\")", "分词：tokenizer_image_token"),
    ("train_ds.py", 124, "use_fast=False,", "慢分词器（sentencepiece）"),
    ("train_ds.py", 126, "tokenizer.pad_token = tokenizer.unk_token", "pad = unk"),
    ("utils/dataset.py", 137, "if inferences[0] == False:", "推理时不截断"),
    ("utils/dataset.py", 377, "image = cv2.imread(image_path)", "读图：cv2.imread"),
    ("utils/dataset.py", 378, "image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)", "BGR 转 RGB"),
    ("utils/dataset.py", 414, 'image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[', "CLIP 一侧：CLIPImageProcessor.preprocess"),
    ("utils/dataset.py", 333, "self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)", "CLIP 处理器取自视觉塔"),
    ("utils/dataset.py", 332, "self.transform = ResizeLongestSide(image_size)", "SAM 一侧：ResizeLongestSide"),
    ("train_ds.py", 38, '"--image_size", default=1024', "SAM 输入 1024"),
    ("utils/dataset.py", 419, "image = self.transform.apply_image(image)", "长边缩到 1024"),
    ("utils/dataset.py", 421, "image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())", "归一化并补零"),
    ("utils/dataset.py", 273, "pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)", "SAM 归一化的均值"),
    ("utils/dataset.py", 274, "pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)", "SAM 归一化的标准差"),
    ("utils/dataset.py", 350, "x = F.pad(x, (0, padw, 0, padh))", "右下补零到正方形"),
    ("utils/dataset.py", 428, 'm = np.zeros((image_info["height"], image_info["width"], 1))', "空标注 → 原图尺寸的零 mask（我们不给 GT，只用它定尺寸）"),
    ("utils/dataset.py", 452, "labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label", "labels 的形状 = 原图尺寸"),
    ("model/LISA.py", 292, "original_size=label_list[i].shape,", "mask 还原到 labels 的形状（原图）"),
    ("model/segment_anything/modeling/sam.py", 161, "masks = F.interpolate(", "双线性放大到 1024"),
    ("model/segment_anything/modeling/sam.py", 168, "masks = masks[..., : input_size[0], : input_size[1]]", "裁掉补零"),
    ("model/segment_anything/modeling/sam.py", 169, "masks = F.interpolate(", "双线性缩放到原图"),
    ("train_ds.py", 549, "output_list = (pred_masks[0] > 0).int()", "阈值：logit > 0"),
    ("model/LISA.py", 201, "if inference:", "推理分支：LLM 一次前向（output_hidden_states）"),
    ("model/LISA.py", 187, "seg_token_mask = input_ids[:, 1:] == self.seg_token_idx", "[SEG] 的位置取自输入（teacher forcing）"),
    ("model/LISA.py", 269, "multimask_output = False", "SAM 单 mask 输出"),
    ("train_ds.py", 33, 'default="bf16"', "精度：bf16（默认）"),
    ("train_ds.py", 146, "torch_dtype = torch.float32", "模型按 precision 取 dtype"),
    ("train_ds.py", 148, "torch_dtype = torch.bfloat16", "bf16 → torch.bfloat16"),
    ("train_ds.py", 538, 'input_dict["images"] = input_dict["images"].bfloat16()', "SAM 输入转 bf16"),
    ("train_ds.py", 539, 'input_dict["images_clip"] = input_dict["images_clip"].bfloat16()', "CLIP 输入转 bf16"),
    ("train_ds.py", 340, "assert args.val_batch_size == 1", "评测每批 1 张图"),
    ("train_ds.py", 533, "input_dict = dict_to_cuda(input_dict)", "dict_to_cuda"),
    ("train_ds.py", 545, "output_dict = model_engine(**input_dict)", "前向：模型的 forward（推理分支）"),
    ("train_ds.py", 39, '"--model_max_length", default=512', "model_max_length 512"),
    ("train_ds.py", 42, '"--vision-tower", default="openai/clip-vit-large-patch14"', "视觉塔"),
    ("chat.py", 127, "model = model.bfloat16().cuda()", "官方推理脚本：bf16 整模型上 GPU"),
    ("model/LISA.py", 138, "config.mm_vision_tower = config.vision_tower", "有 train_mask_decoder 的 config 用 config.vision_tower（忽略参数）"),
)


def citation_ids() -> list[str]:
    return [f"{f}:{n}" for f, n, _, _ in CITATIONS]


# ---------------------------------------------------------------------------
# 环境（LISA 的 venv）
# ---------------------------------------------------------------------------


def hide_tokenizers() -> dict[str, Any]:
    """在 import transformers 之前屏蔽 tokenizers 包：venv 继承 $VI/env 的新版 tokenizers，transformers 4.31 发现它在就核对版本
    （<0.14）并在 import 时报错；屏蔽后与 RUN-0015 的 B200 venv（没装 tokenizers）相同：用慢分词器（官方 use_fast=False），
    transformers 走它自带的 AddedToken 替身。sys.modules[名] = None 使 importlib.util.find_spec 返回 None、import 报 ImportError。"""
    if "tokenizers" in sys.modules and sys.modules["tokenizers"] is None and _HIDDEN:  # 已经屏蔽过（重复调用）
        return dict(_HIDDEN)
    if "transformers" in sys.modules:
        raise LisaError("hide_tokenizers() 要在 import transformers 之前调用")
    if "tokenizers" in sys.modules:
        raise LisaError("tokenizers 已被 import，屏蔽不了")
    found = importlib.util.find_spec("tokenizers") is not None
    version = None
    if found:
        try:
            from importlib.metadata import version as _v

            version = _v("tokenizers")
        except Exception:  # noqa: BLE001 — 只为记录
            version = "unknown"
    sys.modules["tokenizers"] = None  # type: ignore[assignment]
    _HIDDEN.update({"hidden": True, "found_in_environment": bool(found), "found_version": version})
    return dict(_HIDDEN)


_HIDDEN: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 清单（我们的环境导出，LISA 环境只读）
# ---------------------------------------------------------------------------

MANIFEST_KEYS: tuple[str, ...] = ("id", "w2_split", "sent_id", "image_id", "ann_id", "image", "image_sha256", "height", "width", "sent")


def validate_manifest_row(r: Mapping[str, Any]) -> None:
    if tuple(r) != MANIFEST_KEYS:
        raise LisaError(f"清单行的键 {tuple(r)} 不是 {MANIFEST_KEYS}")
    if not isinstance(r["id"], str) or not isinstance(r["sent"], str) or not r["sent"].strip():
        raise LisaError(f"清单行 {r.get('id')!r}：id 或句子不合规")
    if not isinstance(r["image"], str) or not os.path.isabs(r["image"]):
        raise LisaError(f"清单行 {r['id']}：图像路径必须是绝对路径，收到 {r['image']!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(r["image_sha256"])):
        raise LisaError(f"清单行 {r['id']}：image_sha256 不合规")
    for k in ("height", "width"):
        if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 1:
            raise LisaError(f"清单行 {r['id']}：{k} 不合规")


def read_manifest(out: str | Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, str]]:
    """<out>/lisa-7b-v1/inputs 的清单 → (meta, {划分: 行（清单顺序）}, {文件名: sha256})。meta 记的 rows 文件 sha256 必须对得上。"""
    d = inputs_dir(out)
    meta_p, rows_p = d / MANIFEST_META, d / MANIFEST_ROWS
    if not meta_p.is_file() or not rows_p.is_file():
        raise LisaError(f"{d} 里没有清单（{MANIFEST_META}、{MANIFEST_ROWS}）：先在我们的环境里 scripts/lisa_w2.py export")
    raw = rows_p.read_bytes()
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("rows_sha256") != sha256_bytes(raw):
        raise LisaError(f"{rows_p} 的 sha256 与 {meta_p} 记的不同（清单被改过？）")
    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in meta["splits"]}
    seen: set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        validate_manifest_row(r)
        if r["id"] in seen:
            raise LisaError(f"清单里 {r['id']} 出现两次")
        seen.add(r["id"])
        if r["w2_split"] not in rows:
            raise LisaError(f"清单行 {r['id']} 的划分 {r['w2_split']!r} 不在 meta.splits 里")
        rows[r["w2_split"]].append(r)
    for s, e in meta["splits"].items():
        if len(rows[s]) != e["n"]:
            raise LisaError(f"清单的 {s}：{len(rows[s])} 行，meta 记 {e['n']}")
    return meta, rows, {MANIFEST_META: sha256_file(meta_p), MANIFEST_ROWS: sha256_bytes(raw)}


# ---------------------------------------------------------------------------
# 原始输出（LISA 环境写，我们的环境读）
# ---------------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_FAILED = "failed"
FAIL_NO_SEG = "no_seg"  # 输入里没有 [SEG]（teacher forcing 下不该发生；照官方口径按空 mask 计）
FAIL_MULTI_SEG = "multi_seg"  # 多于一个 [SEG]
FAIL_NO_MASK = "no_mask"  # 有 [SEG] 却没有 mask
MAX_CONSECUTIVE_EXCEPTIONS = 3


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """bool mask → COCO 未压缩 RLE（列优先，counts 从 0 的游程开始；只用 numpy）。src/executors/raster.rle_to_mask 能读。"""
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


def read_raw_prefix(path: str | Path, expected_ids: Sequence[str], *, repair: bool) -> list[dict[str, Any]]:
    """原始输出的已完成行：末尾不完整的一行丢掉（repair 时把文件截掉）；id 必须是清单顺序的前缀。"""
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_bytes()
    keep = raw.rfind(b"\n") + 1
    recs = [json.loads(x) for x in raw[:keep].decode("utf-8").splitlines() if x.strip()]
    ids = [r.get("id") for r in recs]
    if ids != list(expected_ids[: len(ids)]):
        dup = sorted(i for i, c in Counter(ids).items() if c > 1)
        raise LisaError(f"{p} 的 id 不是清单顺序的前缀（重复 {dup[:3]}），拒绝续跑或计分")
    if repair and keep != len(raw):
        with open(p, "r+b") as f:
            f.truncate(keep)
    return recs


def append_jsonl(path: str | Path, recs: Iterable[Mapping[str, Any]]) -> None:
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(lines)
        f.flush()
        os.fsync(f.fileno())


def write_json(path: str | Path, doc: Any) -> None:
    Path(path).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def normalized(doc: Any) -> Any:
    """JSON 往返后的形状（元组变列表），续跑时比较身份用。"""
    return json.loads(json.dumps(doc, ensure_ascii=False))


__all__ = [
    "ApprovalError",
    "CITATIONS",
    "CLIP_PATCH_TOKENS",
    "CLIP_PIXELS",
    "EXPERIMENT",
    "LisaError",
    "MANIFEST_KEYS",
    "PROTOCOL",
    "PROTOCOL_NAME",
    "RUN_DIR_NAME",
    "SETTING",
    "SYSTEM",
    "TASK",
    "TEST_APPROVAL",
    "UPSTREAM_COMMIT",
    "VENDOR_DIR",
    "VendorError",
    "WeightsError",
    "append_unlock",
    "check_approval",
    "check_module_origin",
    "check_run_dir",
    "check_vendor",
    "hide_tokenizers",
    "leaf_dir",
    "load_weights_manifest",
    "mask_to_uncompressed_rle",
    "provenance_table",
    "read_manifest",
    "read_raw_prefix",
    "verify_weights",
]
