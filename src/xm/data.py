"""X-M 的数据接口与评测划分（DEV-0079；D165；指令第二节）。只读真值，不跑任何模型。

**单位**：XMPair = "图像 × 该图中出现的类别"（ReasonSeg 是"图像 × 表达式"），每对要求输出该类别的全部实例。

**数据根**（--data-root，HPC 上 $VI/data，RUN-0102 落地；开发机用 tests/_xm_fakes 造的小数据）：
- coco2017/annotations/instances_{train2017,val2017}.json、coco2017/images/{train2017,val2017}/<file_name>；
- lvis/lvis_v1_val.json（图像用 COCO 2017 的：coco_url 的末两级 = <train2017|val2017>/<文件名>）；
- reasonseg/val/<名字>.jpg 与同名 .json（LISA 官方格式：text、is_sentence、shapes[label, points]）。
XMPair.image 是相对数据根的路径。

**COCO 2017 的对**（coco_pairs）：每张图、每个出现的类别一对；非 crowd 标注进 instances（按 ann_id 升序），同类的 crowd 标注进
crowd_rles（评测时忽略落在其中的预测，COCO 惯例）。某类在图中只有 crowd 时不建这一对，记数（n_crowd_only_skipped）。
- RLE：同 pycocotools COCO.annToRLE（多边形 frPyObjects + merge；未压缩 RLE frPyObjects；压缩 RLE 原样），counts 存 ASCII 字符串。
  少于 3 个顶点的多边形（< 6 个数）先丢掉并记数（pycocotools 会把 4 个数的多边形当成框而报错；它们的面积本来就是 0）。
- box_xyxy = COCO bbox [x, y, w, h] → (x, y, x + w, y + h)，原图像素；area = RLE 的前景像素数（pycocotools area，整数；不是 COCO 的
  多边形面积）。
- id = coco2017/<val2017|train2017>/<image_id>/<category_id>；category = COCO 类别名原文（提示里给模型的词）；meta = {image_id}。

**三个评测划分**（build_splits_doc；configs/xm_splits.json，服务器在 HPC 上生成，管理员审过再入库）：
- 总体：COCO val2017 里至少有一个非 crowd 实例的图，先剔除（D20：凡以 COCO val2017 为来源的评测集都剔除探针用图，先剔除再抽样）
  PROBE_IMAGE_KEYS 的两张图，再剔除 E1 的 COCO val2017 检测子集（500 张，src/data/land/coco.select_subset 按同一规则现算，条数与
  sha256 核对 configs/e1_eval_subsets.json 的 coco_val2017 总体）：E1 在这 500 张上的基座零样本读数已经看过，判定集要求"判定前不得查看
  任何系统在上面的结果"（同 D128 的 devcheck3 按句排除 E1 子集的做法；任务书没写这一步，上报待裁，EXCLUDE_E1_COCO 一处开关）。
- 抽法：总体的 image_id（十进制字符串）按 sha256(xm_splits_v1|coco2017_val2017|image_id) 升序（src/data/subsets.hash_rank），依次取前
  300 张 = xm_tune、接着 500 张 = xm_decision、再接着 1,000 张 = xm_final；三份按图互不重叠。各划分的 image_ids 按数值升序列出。
- xm_decision **密封**（门名 G-XM）：load_pairs 不给 gate = "G-XM" 就拒绝（require_unsealed，同 devcheck3 的 G-EX 守卫）。
- xm_final 与 lvis_sub **另加一道**：必须给 check_xm_gate 认过的 gate 文件（XMGatePass）。gate 文件是 DEV-0081 的判定脚本写的
  gate_xm.json，约定的键：verdict（= XM_GATE_PASS "通过"）、rehearsal（= false）、checks（非空、全部为 true）、
  candidate.manifest_sha256（= 本次检查点 <checkpoint>/manifest.json 的 sha256）。DEV-0080 的运行器调 check_xm_gate，再把结果传给
  load_pairs(xm_gate=…)。reasonseg_val 在 load_pairs 里不设卡：对 X-M 的检查点要不要 gate 由运行器按系统决定（X-2 的描述性运行不要）。

**LVIS 子集**（D165 的管理员补充）：LVIS v1 val 里来自 COCO val2017 的图（coco_url 是 val2017/…，且 id 在 instances_val2017 里）、至少
有一个标注、不是 D20 探针图、不在三个 COCO 划分里，按 sha256(xm_lvis_v1|lvis_v1_val|image_id) 升序取 1,000 张。每张图每个有标注的
类别一对（LVIS 没有 crowd）；category = LVIS 类名把 "_" 换成空格（原名进 meta.lvis_name）；meta 记该类的 frequency、该图的
not_exhaustive_category_ids 与 neg_category_ids（官方惯例：not_exhaustive 的类，没匹配上的预测不计 FP；评测时 DEV-0080 用）以及
本对的 not_exhaustive。id = lvis_v1/val/<image_id>/<category_id>。

**ReasonSeg val**（reasonseg_pairs）：同 LISA 官方的验证读法（utils/data_processing.get_mask_from_json + ValDataset）：
- 每个 .jpg 一对，表达式取 text[0]（LISA 的 ValDataset：sampled_sents = [sampled_sents[0]]）；category 放表达式原文，category_id = 0；
- mask：label 为 flag 的形状不要；其余按 cv2.polylines + fillPoly（int32 顶点）算面积，按面积从大到小（np.argsort(...)[::-1]）依次画，
  label 含 ignore 的画 255、其余画 1；目标 = 1 的像素，ignore（= 255）的 RLE 进 meta.ignore_rle（LISA 计分时忽略这些像素）；
- instances 只有 1 个（目标 mask，框 = 它的外接框）；目标为空时不建，记数；
- meta：is_sentence（长句 / 短语）、text_index = 0、n_texts、file、exif_orientation（图的 EXIF 方向；≠ 1 时 PIL 与 cv2 读出的方向
  不同，记数上报）。图像尺寸取 PIL 读到的（与 src/explore/x2_tools.load_rgb 一致）。
- id = reasonseg/val/<文件名去后缀>/<text_index>。

**训练**（load_pairs("train")）：COCO train2017 全部的对，按图剔除 configs/xm_train_exclusions.json（src/xm/exclusions.py）里的图；
清单记的划分文件 sha256 必须等于本次读的 xm_splits.json 的。

读取时（load_pairs）逐项核对：标注文件的 sha256 等于清单记的；重建的对数、对的 id 指纹、实例数等于清单记的。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils

from src.data.constants import PROBE_IMAGE_KEYS
from src.data.subsets import dump_config, hash_rank, population_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
XM_SPLITS_PATH = REPO_ROOT / "configs" / "xm_splits.json"
XM_EXCLUSIONS_PATH = REPO_ROOT / "configs" / "xm_train_exclusions.json"
E1_EVAL_SUBSETS_PATH = REPO_ROOT / "configs" / "e1_eval_subsets.json"
SPLITS_KIND = "xm_splits"
SPLITS_VERSION = 1
FRAMEWORK = "src/xm/data.py（DEV-0079；D165）"

SOURCE_COCO, SOURCE_LVIS, SOURCE_REASONSEG = "coco2017", "lvis_v1", "reasonseg"
XM_TUNE, XM_DECISION, XM_FINAL = "xm_tune", "xm_decision", "xm_final"
LVIS_SUB, REASONSEG_VAL, TRAIN = "lvis_sub", "reasonseg_val", "train"
SPLITS: tuple[str, ...] = (XM_TUNE, XM_DECISION, XM_FINAL, LVIS_SUB, REASONSEG_VAL, TRAIN)
COCO_EVAL_SPLITS: tuple[str, ...] = (XM_TUNE, XM_DECISION, XM_FINAL)  # 按这个先后从同一个排序里依次取
COCO_EVAL_N: Mapping[str, int] = {XM_TUNE: 300, XM_DECISION: 500, XM_FINAL: 1000}
COCO_EVAL_N_TINY: Mapping[str, int] = {XM_TUNE: 2, XM_DECISION: 3, XM_FINAL: 4}
SPLITS_SALT, SPLITS_PURPOSE = "xm_splits_v1", "coco2017_val2017"
LVIS_SALT, LVIS_PURPOSE = "xm_lvis_v1", "lvis_v1_val"
LVIS_N, LVIS_N_TINY = 1000, 2
EXCLUDE_E1_COCO = True  # 剔除 E1 的 COCO val2017 检测子集（模块说明；上报待裁）
E1_COCO_TINY_K = 2  # --tiny 时 E1 子集按同一规则只取 2 张（fixture 只有十几张图）
REASONSEG_TEXT_INDEX = 0  # LISA 的 ValDataset 只用 text[0]

GATE = "G-XM"  # 唯一能打开 xm_decision 的门名
SEAL_DECISION = "D165（指令第二节：判定集密封，判定前不得查看任何系统在上面的结果；第五节）"
SEALED_SPLITS: tuple[str, ...] = (XM_DECISION,)
FINAL_GATED: tuple[str, ...] = (XM_FINAL, LVIS_SUB)  # 判定通过后才能运行（指令第二节），要 gate 文件
XM_GATE_PASS = "通过"  # gate_xm.json 的 verdict（同 scripts/gate_explore.py 与 scripts/w2_gate.py 的 PASS）
XM_GATE_KEYS: tuple[str, ...] = ("verdict", "rehearsal", "checks", "candidate")

# 数据根下的布局（RUN-0102）
COCO_SPLITS: tuple[str, ...] = ("val2017", "train2017")
COCO_ANN: Mapping[str, str] = {s: f"coco2017/annotations/instances_{s}.json" for s in COCO_SPLITS}
COCO_IMAGE_DIR: Mapping[str, str] = {s: f"coco2017/images/{s}" for s in COCO_SPLITS}
LVIS_ANN = "lvis/lvis_v1_val.json"
REASONSEG_DIR = "reasonseg/val"
RUN_DIR_RE = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


class XMDataError(ValueError):
    """X-M 的数据不能读、不合格，或与清单对不上。"""


class SealedSplitError(PermissionError):
    """密封的 xm_decision 没有给对门名（G-XM）。"""


class XMGateError(PermissionError):
    """xm_final / lvis_sub 没有合格的 gate 文件（G-XM 判定通过）。"""


# ---------------------------------------------------------------------------
# 接口（DEV-0079 / 0080 / 0081 共用；字段名不改）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XMInstance:
    ann_id: int
    box_xyxy: tuple[float, float, float, float]  # 原图像素，COCO bbox 换算
    rle: dict  # COCO RLE（size、counts），由多边形或 RLE 统一转
    area: int


@dataclass(frozen=True)
class XMPair:  # 评测与训练的单位：图像 × 该图中出现的类别
    id: str  # "coco2017/<val2017|train2017>/<image_id>/<category_id>"；LVIS 与 ReasonSeg 照此命名
    source: str  # "coco2017" | "lvis_v1" | "reasonseg"
    split: str  # "xm_tune" | "xm_decision" | "xm_final" | "lvis_sub" | "reasonseg_val" | "train"
    image: str  # 相对数据根的路径
    width: int
    height: int
    category_id: int
    category: str  # 类别名就是提示里给模型的词
    instances: tuple[XMInstance, ...]  # 非 crowd 实例；只为"图中出现的类别"建对，所以不为空
    crowd_rles: tuple[dict, ...]  # 同类的 crowd 区域，评测时用来忽略落在其中的预测（COCO 惯例）
    meta: Mapping[str, Any] = field(default_factory=dict)  # 来源特有的元数据（LVIS 的 not_exhaustive 等、ReasonSeg 的 is_sentence）


def pair_to_dict(p: XMPair) -> dict[str, Any]:
    """XMPair → 可写 JSON 的 dict（tuple 写成 list）。"""
    d = asdict(p)
    d["instances"] = [{**i, "box_xyxy": list(i["box_xyxy"])} for i in d["instances"]]
    d["crowd_rles"] = list(d["crowd_rles"])
    d["meta"] = dict(p.meta)
    return d


def pair_from_dict(d: Mapping[str, Any]) -> XMPair:
    inst = tuple(XMInstance(ann_id=int(i["ann_id"]), box_xyxy=tuple(float(v) for v in i["box_xyxy"]), rle=dict(i["rle"]), area=int(i["area"]))
                 for i in d["instances"])
    return XMPair(id=str(d["id"]), source=str(d["source"]), split=str(d["split"]), image=str(d["image"]), width=int(d["width"]),
                  height=int(d["height"]), category_id=int(d["category_id"]), category=str(d["category"]), instances=inst,
                  crowd_rles=tuple(dict(r) for r in d["crowd_rles"]), meta=dict(d.get("meta") or {}))


def pair_by_id(pairs: Iterable[XMPair]) -> dict[str, XMPair]:
    """id → 对；id 重复时报错。"""
    out: dict[str, XMPair] = {}
    for p in pairs:
        if p.id in out:
            raise XMDataError(f"对的 id 重复：{p.id}")
        out[p.id] = p
    return out


def coco_pair_id(coco_split: str, image_id: int, category_id: int) -> str:
    return f"{SOURCE_COCO}/{coco_split}/{int(image_id)}/{int(category_id)}"


def lvis_pair_id(image_id: int, category_id: int) -> str:
    return f"{SOURCE_LVIS}/val/{int(image_id)}/{int(category_id)}"


def reasonseg_pair_id(stem: str, text_index: int = REASONSEG_TEXT_INDEX) -> str:
    return f"{SOURCE_REASONSEG}/val/{stem}/{int(text_index)}"


def check_run_dir(out: str | Path) -> None:
    """--out 的最后一级目录名必须以四位运行编号开头（CLAUDE.md"服务器驱动的写法"）。"""
    name = Path(out).name
    if not RUN_DIR_RE.fullmatch(name):
        raise XMDataError(f"--out 的最后一级目录名必须以四位运行编号开头，形如 0103_xm_data，收到 {name!r}")


# ---------------------------------------------------------------------------
# 文件与 RLE
# ---------------------------------------------------------------------------


def file_sha256(p: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_record(root: str | Path, rel: str) -> dict[str, Any]:
    p = Path(root) / rel
    if not p.is_file():
        raise XMDataError(f"{p} 不在（RUN-0102 的落地）")
    return {"path": rel, "sha256": file_sha256(p), "bytes": p.stat().st_size}


def read_json(p: str | Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _rle_out(rle: Mapping[str, Any]) -> dict:
    c = rle["counts"]
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": c.decode("ascii") if isinstance(c, bytes) else str(c)}


def empty_rle(h: int, w: int) -> dict:
    return _rle_out(mask_utils.encode(np.asfortranarray(np.zeros((int(h), int(w)), np.uint8))))


def segm_to_rle(segm: Any, h: int, w: int) -> tuple[dict, int]:
    """COCO / LVIS 的 segmentation → (压缩 RLE（counts 为 str）, 丢掉的退化多边形个数)。同 pycocotools COCO.annToRLE（模块说明）。"""
    h, w = int(h), int(w)
    if isinstance(segm, list):
        polys = [list(map(float, p)) for p in segm if len(p) >= 6]
        dropped = len(segm) - len(polys)
        if not polys:
            return empty_rle(h, w), dropped
        return _rle_out(mask_utils.merge(mask_utils.frPyObjects(polys, h, w))), dropped
    if isinstance(segm, Mapping) and isinstance(segm.get("counts"), list):
        return _rle_out(mask_utils.frPyObjects(dict(segm), h, w)), 0
    if isinstance(segm, Mapping) and isinstance(segm.get("counts"), (str, bytes)):
        if [int(v) for v in segm["size"]] != [h, w]:
            raise XMDataError(f"RLE 的尺寸 {segm['size']} 与图像 {[h, w]} 不符")
        return _rle_out(segm), 0
    raise XMDataError(f"认不出的 segmentation：{type(segm).__name__}")


def rle_area(rle: Mapping[str, Any]) -> int:
    return int(mask_utils.area({"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}))


def xywh_to_xyxy(b: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in b)
    return (x, y, x + w, y + h)


# ---------------------------------------------------------------------------
# COCO 2017
# ---------------------------------------------------------------------------


def coco_index(doc: Mapping[str, Any]) -> tuple[dict[int, str], dict[int, Mapping[str, Any]], dict[int, list[Mapping[str, Any]]]]:
    cats = {int(c["id"]): str(c["name"]) for c in doc["categories"]}
    images = {int(im["id"]): im for im in doc["images"]}
    anns: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for a in doc["annotations"]:
        anns[int(a["image_id"])].append(a)
    return cats, images, anns


def images_with_noncrowd(doc: Mapping[str, Any]) -> list[int]:
    """至少有一个非 crowd 实例的图（升序）。"""
    return sorted({int(a["image_id"]) for a in doc["annotations"] if not int(a.get("iscrowd", 0))})


def coco_pairs(doc: Mapping[str, Any], *, coco_split: str, split: str, image_ids: Iterable[int] | None = None,
               index: tuple | None = None) -> tuple[list[XMPair], dict[str, Any]]:
    """COCO 2017 的对（模块说明）。image_ids = None：全部图。返回 (对（按 image_id、category_id 升序）, 计数)。"""
    if coco_split not in COCO_SPLITS:
        raise XMDataError(f"coco_split 必须是 {COCO_SPLITS} 之一，收到 {coco_split!r}")
    cats, images, anns = index or coco_index(doc)
    ids = sorted(images) if image_ids is None else sorted({int(i) for i in image_ids})
    missing = [i for i in ids if i not in images]
    if missing:
        raise XMDataError(f"{coco_split}：{len(missing)} 张图不在标注里，例如 {missing[:5]}")
    st = Counter()
    out: list[XMPair] = []
    for iid in ids:
        im = images[iid]
        h, w = int(im["height"]), int(im["width"])
        by_cat: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for a in anns.get(iid, []):
            by_cat[int(a["category_id"])].append(a)
        st["n_images"] += 1
        made = 0
        for cid in sorted(by_cat):
            if cid not in cats:
                raise XMDataError(f"{coco_split} 图 {iid}：类别 {cid} 不在 categories 里")
            group = sorted(by_cat[cid], key=lambda a: int(a["id"]))
            inst, crowd = [], []
            for a in group:
                rle, dropped = segm_to_rle(a["segmentation"], h, w)
                st["n_degenerate_polygons_dropped"] += dropped
                if int(a.get("iscrowd", 0)):
                    crowd.append(rle)
                    continue
                area = rle_area(rle)
                st["n_empty_mask_instances"] += int(area == 0)
                inst.append(XMInstance(ann_id=int(a["id"]), box_xyxy=xywh_to_xyxy(a["bbox"]), rle=rle, area=area))
            if not inst:
                st["n_crowd_only_skipped"] += 1
                continue
            st["n_pairs"] += 1
            st["n_instances"] += len(inst)
            st["n_crowd_regions"] += len(crowd)
            made += 1
            out.append(XMPair(id=coco_pair_id(coco_split, iid, cid), source=SOURCE_COCO, split=split,
                              image=f"{COCO_IMAGE_DIR[coco_split]}/{im['file_name']}", width=w, height=h, category_id=cid,
                              category=cats[cid], instances=tuple(inst), crowd_rles=tuple(crowd), meta={"image_id": iid}))
        st["n_images_without_pairs"] += int(made == 0)
    keys = ("n_images", "n_images_without_pairs", "n_pairs", "n_instances", "n_crowd_regions", "n_crowd_only_skipped",
            "n_empty_mask_instances", "n_degenerate_polygons_dropped")
    return out, {k: int(st[k]) for k in keys}


# ---------------------------------------------------------------------------
# LVIS v1 val
# ---------------------------------------------------------------------------


def lvis_image_split(im: Mapping[str, Any]) -> tuple[str, str]:
    """LVIS 图像 → (COCO 2017 的划分, 文件名)：coco_url 的末两级。"""
    parts = str(im.get("coco_url", "")).rstrip("/").split("/")
    if len(parts) < 2 or parts[-2] not in COCO_SPLITS:
        raise XMDataError(f"LVIS 图 {im.get('id')} 的 coco_url {im.get('coco_url')!r} 认不出 COCO 2017 的划分")
    return parts[-2], parts[-1]


def lvis_category_word(name: str) -> str:
    """LVIS 类名 → 提示里的词："_" 换成空格（如 bow_(weapon) → bow (weapon)）。"""
    return str(name).replace("_", " ")


def lvis_pairs(doc: Mapping[str, Any], *, image_ids: Iterable[int], split: str = LVIS_SUB) -> tuple[list[XMPair], dict[str, Any]]:
    """LVIS 子集的对（模块说明）。图必须来自 COCO val2017。"""
    cats = {int(c["id"]): c for c in doc["categories"]}
    images = {int(im["id"]): im for im in doc["images"]}
    anns: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for a in doc["annotations"]:
        anns[int(a["image_id"])].append(a)
    st = Counter()
    out: list[XMPair] = []
    for iid in sorted({int(i) for i in image_ids}):
        if iid not in images:
            raise XMDataError(f"LVIS：图 {iid} 不在 lvis_v1_val.json 里")
        im = images[iid]
        sp, fn = lvis_image_split(im)
        if sp != "val2017":
            raise XMDataError(f"LVIS 子集只取 COCO val2017 的图：{iid} 来自 {sp}")
        h, w = int(im["height"]), int(im["width"])
        neg = sorted(int(c) for c in im.get("neg_category_ids", []))
        nex = sorted(int(c) for c in im.get("not_exhaustive_category_ids", []))
        by_cat: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for a in anns.get(iid, []):
            by_cat[int(a["category_id"])].append(a)
        st["n_images"] += 1
        for cid in sorted(by_cat):
            c = cats[cid]
            inst = []
            for a in sorted(by_cat[cid], key=lambda a: int(a["id"])):
                rle, dropped = segm_to_rle(a["segmentation"], h, w)
                st["n_degenerate_polygons_dropped"] += dropped
                area = rle_area(rle)
                st["n_empty_mask_instances"] += int(area == 0)
                inst.append(XMInstance(ann_id=int(a["id"]), box_xyxy=xywh_to_xyxy(a["bbox"]), rle=rle, area=area))
            st["n_pairs"] += 1
            st["n_instances"] += len(inst)
            st["n_not_exhaustive_pairs"] += int(cid in nex)
            out.append(XMPair(id=lvis_pair_id(iid, cid), source=SOURCE_LVIS, split=split, image=f"{COCO_IMAGE_DIR['val2017']}/{fn}",
                              width=w, height=h, category_id=cid, category=lvis_category_word(c["name"]), instances=tuple(inst),
                              crowd_rles=(), meta={"image_id": iid, "lvis_name": str(c["name"]), "frequency": c.get("frequency"),
                                                   "not_exhaustive": cid in nex, "not_exhaustive_category_ids": nex,
                                                   "neg_category_ids": neg}))
    keys = ("n_images", "n_pairs", "n_instances", "n_not_exhaustive_pairs", "n_empty_mask_instances", "n_degenerate_polygons_dropped")
    return out, {k: int(st[k]) for k in keys}


# ---------------------------------------------------------------------------
# ReasonSeg val（LISA 官方格式）
# ---------------------------------------------------------------------------


def read_reasonseg_json(p: str | Path) -> dict[str, Any]:
    """同 LISA：先按 UTF-8 读，失败再按 cp1252。"""
    raw = Path(p).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))


def reasonseg_mask(shapes: Sequence[Mapping[str, Any]], h: int, w: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """LISA 的 get_mask_from_json（模块说明）→ (目标 mask, ignore mask, 计数)。"""
    import cv2

    valid, areas = [], []
    n_flag = 0
    for s in shapes:
        if str(s["label"]).lower() == "flag":  # LISA：meaningless deprecated annotations
            n_flag += 1
            continue
        pts = np.array([s["points"]], dtype=np.int32)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(tmp, pts, True, 1, 1)
        cv2.fillPoly(tmp, pts, 1)
        areas.append(tmp.sum())
        valid.append(s)
    order = list(np.argsort(areas)[::-1].astype(np.int32)) if areas else []
    mask = np.zeros((h, w), dtype=np.uint8)
    n_ignore = 0
    for k in order:
        s = valid[int(k)]
        v = 255 if "ignore" in str(s["label"]).lower() else 1
        n_ignore += int(v == 255)
        pts = np.array([s["points"]], dtype=np.int32)
        cv2.polylines(mask, pts, True, v, 1)
        cv2.fillPoly(mask, pts, v)
    return mask == 1, mask == 255, {"n_flag": n_flag, "n_ignore_polygons": n_ignore, "n_target_polygons": len(valid) - n_ignore}


def _mask_rle(m: np.ndarray) -> dict:
    return _rle_out(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))


def _mask_box(m: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def reasonseg_files(data_root: str | Path) -> list[str]:
    """ReasonSeg val 的名字（.jpg 去后缀，升序）；每个都要有同名 .json。"""
    d = Path(data_root) / REASONSEG_DIR
    if not d.is_dir():
        raise XMDataError(f"{d} 不在（RUN-0102 的落地）")
    stems = sorted(p.stem for p in d.glob("*.jpg"))
    if not stems:
        raise XMDataError(f"{d} 里没有 .jpg")
    lacking = [s for s in stems if not (d / f"{s}.json").is_file()]
    if lacking:
        raise XMDataError(f"{d}：{len(lacking)} 张图没有同名 .json，例如 {lacking[:3]}")
    return stems


def reasonseg_pairs(data_root: str | Path, stems: Sequence[str] | None = None, *, split: str = REASONSEG_VAL) -> tuple[list[XMPair], dict[str, Any]]:
    from PIL import Image

    d = Path(data_root) / REASONSEG_DIR
    stems = reasonseg_files(data_root) if stems is None else list(stems)
    st = Counter()
    out: list[XMPair] = []
    for s in stems:
        doc = read_reasonseg_json(d / f"{s}.json")
        with Image.open(d / f"{s}.jpg") as im:
            w, h = im.size
            orient = int(im.getexif().get(0x0112, 1) or 1)
        texts = list(doc["text"])
        if not texts:
            raise XMDataError(f"ReasonSeg {s}：text 为空")
        tgt, ign, c = reasonseg_mask(doc["shapes"], h, w)
        st["n_files"] += 1
        st["n_multi_text"] += int(len(texts) > 1)
        st["n_exif_rotated"] += int(orient != 1)
        st["n_with_ignore"] += int(bool(ign.any()))
        if not tgt.any():
            st["n_empty_target_skipped"] += 1
            continue
        is_sent = bool(doc["is_sentence"])
        st["n_sentence" if is_sent else "n_phrase"] += 1
        inst = XMInstance(ann_id=0, box_xyxy=_mask_box(tgt), rle=_mask_rle(tgt), area=int(tgt.sum()))
        out.append(XMPair(id=reasonseg_pair_id(s), source=SOURCE_REASONSEG, split=split, image=f"{REASONSEG_DIR}/{s}.jpg", width=int(w),
                          height=int(h), category_id=0, category=str(texts[REASONSEG_TEXT_INDEX]), instances=(inst,), crowd_rles=(),
                          meta={"is_sentence": is_sent, "text_index": REASONSEG_TEXT_INDEX, "n_texts": len(texts), "file": f"{s}.jpg",
                                "ignore_rle": _mask_rle(ign) if ign.any() else None, "exif_orientation": orient, **c}))
        st["n_pairs"] += 1
    keys = ("n_files", "n_pairs", "n_sentence", "n_phrase", "n_multi_text", "n_with_ignore", "n_empty_target_skipped", "n_exif_rotated")
    return out, {k: int(st[k]) for k in keys}


def reasonseg_file_records(data_root: str | Path, stems: Sequence[str]) -> dict[str, dict[str, str]]:
    d = Path(data_root) / REASONSEG_DIR
    return {s: {"jpg": file_sha256(d / f"{s}.jpg"), "json": file_sha256(d / f"{s}.json")} for s in stems}


# ---------------------------------------------------------------------------
# 划分（纯函数）
# ---------------------------------------------------------------------------


def probe_image_ids() -> list[int]:
    """D20 的探针用图（COCO image_id）。"""
    return sorted(int(k.split(":")[1]) for k in PROBE_IMAGE_KEYS if k.startswith("coco:"))


def e1_coco_subset(val_doc: Mapping[str, Any], *, tiny: bool) -> list[int]:
    """E1 的 COCO val2017 检测子集（src/data/land/coco.select_subset，同一规则现算：全部 5,000 张先剔除 D20 再哈希取 500 张）。
    真运行时条数与 sha256 核对 configs/e1_eval_subsets.json 的 coco_val2017 总体。"""
    from src.data.land.coco import select_subset

    all_ids = sorted(int(im["id"]) for im in val_doc["images"])
    chosen, _ = select_subset(all_ids, E1_COCO_TINY_K) if tiny else select_subset(all_ids)
    if not tiny:
        want = read_json(E1_EVAL_SUBSETS_PATH)["populations"]["coco_val2017"]
        got = {"n": len(chosen), "sha256": population_digest(str(i) for i in chosen)}
        if (got["n"], got["sha256"]) != (int(want["n"]), want["sha256"]):
            raise XMDataError(f"现算的 E1 COCO 子集 {got} 与 configs/e1_eval_subsets.json 记的 {want['n']} 张 {want['sha256'][:16]}… 不同")
    return sorted(chosen)


def _digest(ids: Iterable[Any]) -> str:
    return population_digest(str(i) for i in ids)


def rank_images(ids: Iterable[int], purpose: str, salt: str) -> list[int]:
    """image_id 按 sha256(salt|purpose|image_id) 升序。"""
    return [int(s) for s in hash_rank([str(int(i)) for i in ids], purpose, salt)]


def coco_population(val_doc: Mapping[str, Any], *, tiny: bool) -> tuple[list[int], dict[str, Any], list[int]]:
    """三个划分的总体（模块说明）：至少一个非 crowd 实例的图 − D20 − E1 的 COCO 子集。"""
    all_ids = sorted(int(im["id"]) for im in val_doc["images"])
    base = images_with_noncrowd(val_doc)
    probe = set(probe_image_ids())
    after_probe = [i for i in base if i not in probe]
    e1 = e1_coco_subset(val_doc, tiny=tiny) if EXCLUDE_E1_COCO else []
    e1s = set(e1)
    pop = [i for i in after_probe if i not in e1s]
    steps = {
        "n_images_total": len(all_ids),
        "with_noncrowd": {"n": len(base), "sha256": _digest(base)},
        "exclude_d20_probe": {"keys": list(PROBE_IMAGE_KEYS), "n_removed": len(base) - len(after_probe)},
        "exclude_e1_coco_val2017": {"enabled": EXCLUDE_E1_COCO, "n_listed": len(e1), "sha256": _digest(e1) if e1 else None,
                                    "n_removed": len(after_probe) - len(pop),
                                    "rule": "src/data/land/coco.select_subset（E1 的 500 张；--tiny 取 2 张）" if EXCLUDE_E1_COCO else None},
        "population": {"n": len(pop), "sha256": _digest(pop)},
    }
    return pop, steps, e1


def select_coco_splits(pop: Sequence[int], sizes: Mapping[str, int]) -> dict[str, list[int]]:
    ranked = rank_images(pop, SPLITS_PURPOSE, SPLITS_SALT)
    need = sum(sizes[s] for s in COCO_EVAL_SPLITS)
    if len(ranked) < need:
        raise XMDataError(f"总体只有 {len(ranked)} 张图，不够三个划分的 {need} 张")
    out, k = {}, 0
    for s in COCO_EVAL_SPLITS:
        out[s] = sorted(ranked[k: k + sizes[s]])
        k += sizes[s]
    return out


def lvis_population(lvis_doc: Mapping[str, Any], val_ids: Iterable[int], exclude: Iterable[int]) -> tuple[list[int], dict[str, Any]]:
    """LVIS 子集的总体（模块说明）。"""
    val = {int(i) for i in val_ids}
    with_ann = {int(a["image_id"]) for a in lvis_doc["annotations"]}
    n_total = len(lvis_doc["images"])
    from_val = sorted(int(im["id"]) for im in lvis_doc["images"] if lvis_image_split(im)[0] == "val2017")
    not_in_val_json = [i for i in from_val if i not in val]
    if not_in_val_json:
        raise XMDataError(f"LVIS：{len(not_in_val_json)} 张 coco_url 是 val2017 的图不在 instances_val2017 里，例如 {not_in_val_json[:5]}")
    annotated = [i for i in from_val if i in with_ann]
    probe = set(probe_image_ids())
    no_probe = [i for i in annotated if i not in probe]
    ex = {int(i) for i in exclude}
    pop = [i for i in no_probe if i not in ex]
    return pop, {"n_images_total": n_total, "from_val2017": {"n": len(from_val), "sha256": _digest(from_val)},
                 "with_annotations": {"n": len(annotated)}, "exclude_d20_probe": {"n_removed": len(annotated) - len(no_probe)},
                 "exclude_coco_splits": {"n_listed": len(ex), "n_removed": len(no_probe) - len(pop)},
                 "population": {"n": len(pop), "sha256": _digest(pop)}}


METHOD = ("COCO val2017 里至少有一个非 crowd 实例的图，先剔除 D20 的两张探针用图与 E1 的 COCO val2017 检测子集（500 张），按 "
          "sha256(xm_splits_v1|coco2017_val2017|image_id) 升序依次取 300 张 xm_tune、500 张 xm_decision、1,000 张 xm_final；LVIS 子集 = LVIS v1 val "
          "里来自 COCO val2017、有标注、不是 D20 探针图、不在三个划分里的图，按 sha256(xm_lvis_v1|lvis_v1_val|image_id) 升序取 1,000 张；"
          "reasonseg_val = ReasonSeg val 的全部图。sha256 = image_id（或文件名）自然顺序、换行连接后的 sha256（src/data/subsets.population_digest）；"
          "pairs_sha256 = 对的 id 的同一种指纹。规则见 src/xm/data.py 的模块说明")
SEAL = {"split": XM_DECISION, "gate": GATE, "decision": SEAL_DECISION,
        "rule": (f"判定 {GATE} 之前，任何人不看任何系统在 xm_decision 上的结果；读取必须经 src/xm/data.require_unsealed（gate = {GATE}）。"
                 f"xm_final 与 lvis_sub 另要 G-XM 判定通过的 gate 文件（src/xm/data.check_xm_gate）")}


def _split_entry(split: str, ids: Sequence[int], pairs: Sequence[XMPair], stats: Mapping[str, Any], *, rule: str,
                 extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"source": SOURCE_LVIS if split == LVIS_SUB else SOURCE_COCO, "coco_split": "val2017", "unit": "image_id", "rule": rule,
            **dict(extra or {}), "sealed": split in SEALED_SPLITS, "gate": GATE if split in SEALED_SPLITS else None,
            "requires_xm_gate": split in FINAL_GATED, "n_images": len(ids), "sha256": _digest(ids), "n_pairs": len(pairs),
            "pairs_sha256": population_digest(p.id for p in pairs), "n_instances": int(sum(len(p.instances) for p in pairs)),
            "stats": dict(stats), "image_ids": [int(i) for i in ids]}


def build_splits_doc(data_root: str | Path, *, tiny: bool = False, log=None) -> dict[str, Any]:
    """在数据上生成 configs/xm_splits.json 的内容（键序固定，dump_config 的文本即入库文件）。"""
    from src.data.land.coco import SPEC as COCO_SPEC

    log = log or (lambda m: None)
    root = Path(data_root)
    sizes = COCO_EVAL_N_TINY if tiny else COCO_EVAL_N
    lvis_n = LVIS_N_TINY if tiny else LVIS_N
    src_val = file_record(root, COCO_ANN["val2017"])
    pin_ok = src_val["sha256"] == COCO_SPEC["ann_sha256"]
    if not tiny and not pin_ok:
        raise XMDataError(f"instances_val2017.json 的 sha256 {src_val['sha256'][:16]}… 与 P0 记录的官方文件（src/data/land/coco.SPEC）不同")
    src_val["check"] = "= P0 记录的官方 instances_val2017.json（src/data/land/coco.SPEC）" if pin_ok else "tiny：不核对"
    src_lvis = file_record(root, LVIS_ANN)
    val_doc = read_json(root / COCO_ANN["val2017"])
    log(f"[X-M 划分] instances_val2017：{len(val_doc['images'])} 张图、{len(val_doc['annotations'])} 个标注")
    pop, steps, e1_ids = coco_population(val_doc, tiny=tiny)
    chosen = select_coco_splits(pop, sizes)
    index = coco_index(val_doc)
    splits: dict[str, Any] = {}
    k = 0
    for s in COCO_EVAL_SPLITS:
        pairs, st = coco_pairs(val_doc, coco_split="val2017", split=s, image_ids=chosen[s], index=index)
        rule = f"总体按 sha256({SPLITS_SALT}|{SPLITS_PURPOSE}|image_id) 升序的第 {k + 1}–{k + sizes[s]} 张"
        splits[s] = _split_entry(s, chosen[s], pairs, st, rule=rule, extra={"rank_range": [k, k + sizes[s]], "salt": SPLITS_SALT,
                                                                            "purpose": SPLITS_PURPOSE})
        k += sizes[s]
        log(f"[X-M 划分] {s}：{len(chosen[s])} 张图、{len(pairs)} 对、{splits[s]['n_instances']} 个实例")
    used = sorted(set().union(*(set(chosen[s]) for s in COCO_EVAL_SPLITS)))
    lvis_doc = read_json(root / LVIS_ANN)
    lpop, lsteps = lvis_population(lvis_doc, (int(im["id"]) for im in val_doc["images"]), used)
    if len(lpop) < lvis_n:
        raise XMDataError(f"LVIS 子集的总体只有 {len(lpop)} 张图，不够 {lvis_n} 张")
    lids = sorted(rank_images(lpop, LVIS_PURPOSE, LVIS_SALT)[:lvis_n])
    lpairs, lst = lvis_pairs(lvis_doc, image_ids=lids)
    splits[LVIS_SUB] = _split_entry(LVIS_SUB, lids, lpairs, lst, rule=f"LVIS 子集的总体按 sha256({LVIS_SALT}|{LVIS_PURPOSE}|image_id) 升序取前 {lvis_n} 张",
                                    extra={"salt": LVIS_SALT, "purpose": LVIS_PURPOSE, "population_steps": lsteps})
    log(f"[X-M 划分] {LVIS_SUB}：{len(lids)} 张图、{len(lpairs)} 对")
    stems = reasonseg_files(root)
    rpairs, rst = reasonseg_pairs(root, stems)
    recs = reasonseg_file_records(root, stems)
    splits[REASONSEG_VAL] = {"source": SOURCE_REASONSEG, "unit": "文件名（.jpg 去后缀）", "rule": "ReasonSeg val 的全部图；表达式取 text[0]（LISA 的 ValDataset）",
                             "sealed": False, "gate": None, "requires_xm_gate": False, "n_images": len(stems), "sha256": population_digest(stems),
                             "n_pairs": len(rpairs), "pairs_sha256": population_digest(p.id for p in rpairs),
                             "n_instances": len(rpairs), "stats": rst, "files": recs}
    log(f"[X-M 划分] {REASONSEG_VAL}：{len(stems)} 张图、{len(rpairs)} 对（长句 {rst['n_sentence']}、短语 {rst['n_phrase']}）")
    checks = splits_checks(splits, val_ids=[int(im["id"]) for im in val_doc["images"]], e1_ids=e1_ids)
    bad = [f"{k} = {v}" for k, v in checks.items() if v]
    if bad:
        raise XMDataError("划分的重叠核对不过：" + "；".join(bad))
    return {"version": SPLITS_VERSION, "kind": SPLITS_KIND, "framework": FRAMEWORK, "tiny": bool(tiny), "method": METHOD, "seal": dict(SEAL),
            "sizes": {**{s: sizes[s] for s in COCO_EVAL_SPLITS}, LVIS_SUB: lvis_n},
            "sources": {"coco_val2017": src_val, "lvis_v1_val": src_lvis, "reasonseg_val": {"dir": REASONSEG_DIR, "n_files": len(stems)}},
            "population": steps, "splits": splits, "checks": checks}


def splits_checks(splits: Mapping[str, Any], *, val_ids: Iterable[int] | None, e1_ids: Iterable[int] = ()) -> dict[str, int]:
    """划分之间按图的重叠、与 D20 / E1 子集的重叠、LVIS 子集不在 val2017 的图（都必须是 0）。"""
    out: dict[str, int] = {}
    imgs = {s: set(int(i) for i in splits[s]["image_ids"]) for s in (*COCO_EVAL_SPLITS, LVIS_SUB)}
    names = list(imgs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            out[f"shared_images:{names[a]}|{names[b]}"] = len(imgs[names[a]] & imgs[names[b]])
    probe = set(probe_image_ids())
    e1s = {int(i) for i in e1_ids}
    for s, ids in imgs.items():
        out[f"d20_probe_in:{s}"] = len(ids & probe)
        if s != LVIS_SUB:
            out[f"e1_coco_in:{s}"] = len(ids & e1s)
        if val_ids is not None:
            out[f"not_val2017:{s}"] = len(ids - {int(i) for i in val_ids})
    return out


def check_splits_doc(doc: Mapping[str, Any]) -> list[str]:
    """不要数据的核对：格式、各划分的条数与 sha256、密封字段、互不重叠、大小等于规定（--tiny 的按 tiny 大小）。返回问题列表。"""
    if doc.get("kind") != SPLITS_KIND or doc.get("version") != SPLITS_VERSION:
        return [f"不是 X-M 的划分清单（kind {doc.get('kind')!r}，version {doc.get('version')!r}）"]
    bad: list[str] = []
    want_names = {*COCO_EVAL_SPLITS, LVIS_SUB, REASONSEG_VAL}
    if set(doc.get("splits") or {}) != want_names:
        return [f"划分名不对：{sorted(doc.get('splits') or {})}，应是 {sorted(want_names)}"]
    tiny = bool(doc.get("tiny"))
    sizes = {**(COCO_EVAL_N_TINY if tiny else COCO_EVAL_N), LVIS_SUB: LVIS_N_TINY if tiny else LVIS_N}
    for s, e in doc["splits"].items():
        if s == REASONSEG_VAL:
            files = sorted(e["files"])
            if population_digest(files) != e["sha256"] or len(files) != e["n_images"]:
                bad.append(f"{s}：文件名的条数或 sha256 不对")
            continue
        ids = [int(i) for i in e["image_ids"]]
        if ids != sorted(set(ids)) or len(ids) != e["n_images"] or _digest(ids) != e["sha256"]:
            bad.append(f"{s}：image_ids 的顺序、条数或 sha256 不对")
        if len(ids) != sizes[s]:
            bad.append(f"{s}：{len(ids)} 张图，应是 {sizes[s]} 张")
        if e["sealed"] is not (s in SEALED_SPLITS) or e["gate"] != (GATE if s in SEALED_SPLITS else None) \
                or e["requires_xm_gate"] is not (s in FINAL_GATED):
            bad.append(f"{s}：密封或 gate 字段不对")
    try:
        nz = {k: v for k, v in splits_checks(doc["splits"], val_ids=None).items() if v}
    except (KeyError, TypeError, ValueError) as ex:
        nz = {"格式": str(ex)}
    bad += [f"{k} = {v}" for k, v in nz.items()]
    rec = doc.get("checks") or {}
    bad += [f"checks.{k} = {v}" for k, v in rec.items() if v]
    return bad


def load_splits_doc(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else XM_SPLITS_PATH
    if not p.is_file():
        raise XMDataError(f"{p} 不在：先在服务器上生成（scripts/xm/build_configs.py），管理员审过再入库")
    doc = read_json(p)
    bad = check_splits_doc(doc)
    if bad:
        raise XMDataError(f"{p} 不合格：" + "；".join(bad[:10]))
    return doc


# ---------------------------------------------------------------------------
# 密封与 gate
# ---------------------------------------------------------------------------


def is_sealed(split: str) -> bool:
    return split in SEALED_SPLITS


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def require_unsealed(split: str, gate: str | None, *, now: str | None = None) -> dict[str, Any] | None:
    """密封守卫：xm_decision 时 gate 必须恰是 G-XM，否则 SealedSplitError；放行时返回 {gate, decision, unsealed_utc}。其余划分返回 None。"""
    if not is_sealed(split):
        return None
    if gate != GATE:
        got = "没有给" if gate is None else f"给的是 {gate!r}"
        raise SealedSplitError(f"划分 {split} 是密封的判定集（{SEAL_DECISION}）。只有判定 {GATE} 能用它：显式给 --gate {GATE}（{got}）")
    return {"gate": GATE, "decision": SEAL_DECISION, "unsealed_utc": now or _utc_now()}


def seal_identity(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """放行记录里进运行身份的部分（不含时间）。"""
    return None if record is None else {"gate": record["gate"], "decision": record["decision"]}


@dataclass(frozen=True)
class XMGatePass:
    """check_xm_gate 认过的 gate 文件：xm_final / lvis_sub 的通行证。"""

    path: str
    sha256: str
    checkpoint_manifest_sha256: str
    verdict: str

    def identity(self) -> dict[str, Any]:
        return {"gate": GATE, "gate_xm": {"path": self.path, "sha256": self.sha256}, "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256}


def check_xm_gate(gate_path: str | Path, *, checkpoint: str | Path) -> XMGatePass:
    """xm_final / lvis_sub 的放行（模块说明）：gate_xm.json 的 verdict = "通过"、rehearsal = false、checks 非空且全部为 true、
    candidate.manifest_sha256 = <checkpoint>/manifest.json 的 sha256。不满足抛 XMGateError。"""
    p = Path(gate_path)
    try:
        raw = p.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as e:
        raise XMGateError(f"gate 文件 {p} 读不了：{e}") from e
    if not isinstance(doc, dict) or any(k not in doc for k in XM_GATE_KEYS):
        raise XMGateError(f"gate 文件 {p} 不是 G-XM 的判定输出（要有键 {list(XM_GATE_KEYS)}）")
    if doc["rehearsal"] is not False:
        raise XMGateError(f"gate 文件 {p} 是彩排（rehearsal = {doc['rehearsal']!r}），不作数")
    if doc["verdict"] != XM_GATE_PASS:
        raise XMGateError(f"gate 文件 {p} 的结论不是“{XM_GATE_PASS}”（verdict = {doc['verdict']!r}）")
    checks = doc["checks"]
    if not isinstance(checks, dict) or not checks or any(v is not True for v in checks.values()):
        bad = sorted(k for k, v in checks.items() if v is not True) if isinstance(checks, dict) else checks
        raise XMGateError(f"gate 文件 {p} 的判定点核对没有全部为真：{bad or '空'}")
    cand = doc["candidate"] if isinstance(doc["candidate"], dict) else {}
    man = Path(checkpoint) / "manifest.json"
    if not man.is_file():
        raise XMGateError(f"检查点 {checkpoint} 里没有 manifest.json，核对不了")
    want = file_sha256(man)
    if cand.get("manifest_sha256") != want:
        raise XMGateError(f"gate 文件 {p} 判定的检查点（manifest sha256 {str(cand.get('manifest_sha256'))[:16]}…）不是本次的（{want[:16]}…）")
    return XMGatePass(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), checkpoint_manifest_sha256=want, verdict=str(doc["verdict"]))


def require_xm_gate(split: str, xm_gate: XMGatePass | None) -> dict[str, Any] | None:
    """xm_final / lvis_sub 必须带 check_xm_gate 的结果；其余划分返回 None。"""
    if split not in FINAL_GATED:
        return None
    if not isinstance(xm_gate, XMGatePass):
        raise XMGateError(f"划分 {split} 只在 G-XM 判定通过后运行（指令第二节）：要 check_xm_gate 认过的 gate 文件，收到 {type(xm_gate).__name__}")
    return xm_gate.identity()


# ---------------------------------------------------------------------------
# 取用
# ---------------------------------------------------------------------------


def _verify_source(root: Path, rec: Mapping[str, Any], what: str) -> None:
    p = root / rec["path"]
    if not p.is_file():
        raise XMDataError(f"{p} 不在")
    if file_sha256(p) != rec["sha256"]:
        raise XMDataError(f"{what}：{p} 的 sha256 与清单记的不同（标注换过了？）")


def _verify_pairs(split: str, pairs: Sequence[XMPair], e: Mapping[str, Any]) -> None:
    got = (len(pairs), population_digest(p.id for p in pairs), int(sum(len(p.instances) for p in pairs)))
    want = (e["n_pairs"], e["pairs_sha256"], e["n_instances"])
    if got != want:
        raise XMDataError(f"{split}：重建的（对数、对的指纹、实例数）{got[0]}、{got[1][:12]}…、{got[2]} 与清单记的 {want[0]}、{want[1][:12]}…、{want[2]} 不同")


def _pairs(split: str, data_root: str | Path, *, splits_path: str | Path | None, exclusions_path: str | Path | None) -> list[XMPair]:
    if split not in SPLITS:
        raise XMDataError(f"没有划分 {split!r}；可用：{list(SPLITS)}")
    root = Path(data_root)
    sp = Path(splits_path) if splits_path is not None else XM_SPLITS_PATH
    doc = load_splits_doc(sp)
    if split == TRAIN:
        from src.xm.exclusions import load_exclusions  # 按需 import

        ex = load_exclusions(exclusions_path, splits_path=sp)
        _verify_source(root, ex["sources"]["coco_train2017"], "COCO train2017 的标注")
        tr = read_json(root / COCO_ANN["train2017"])
        drop = set(int(i) for i in ex["exclude_train2017_image_ids"])
        keep = [int(im["id"]) for im in tr["images"] if int(im["id"]) not in drop]
        pairs, _ = coco_pairs(tr, coco_split="train2017", split=TRAIN, image_ids=keep)
        return pairs
    e = doc["splits"][split]
    if split in COCO_EVAL_SPLITS:
        _verify_source(root, doc["sources"]["coco_val2017"], "COCO val2017 的标注")
        pairs, _ = coco_pairs(read_json(root / COCO_ANN["val2017"]), coco_split="val2017", split=split, image_ids=e["image_ids"])
    elif split == LVIS_SUB:
        _verify_source(root, doc["sources"]["lvis_v1_val"], "LVIS v1 val 的标注")
        pairs, _ = lvis_pairs(read_json(root / LVIS_ANN), image_ids=e["image_ids"])
    else:
        stems = sorted(e["files"])
        if reasonseg_file_records(root, stems) != e["files"]:
            raise XMDataError("ReasonSeg val 的文件与清单记的 sha256 不同")
        pairs, _ = reasonseg_pairs(root, stems)
    _verify_pairs(split, pairs, e)
    return pairs


def load_pairs(split: str, data_root: str | Path, *, gate: str | None = None, xm_gate: XMGatePass | None = None,
               splits_path: str | Path | None = None, exclusions_path: str | Path | None = None) -> list[XMPair]:
    """一个划分的全部对（按 id 的来源顺序：image_id、category_id 升序；ReasonSeg 按文件名）。
    xm_decision 要 gate = "G-XM"（require_unsealed）；xm_final、lvis_sub 要 xm_gate（check_xm_gate 的结果）；train 自动剔除去重清单里的图。
    splits_path / exclusions_path 缺省是入库的 configs/xm_splits.json 与 configs/xm_train_exclusions.json。"""
    if split not in SPLITS:
        raise XMDataError(f"没有划分 {split!r}；可用：{list(SPLITS)}")
    require_unsealed(split, gate)
    require_xm_gate(split, xm_gate)
    return _pairs(split, data_root, splits_path=splits_path, exclusions_path=exclusions_path)


def stats_pairs(split: str, data_root: str | Path, *, splits_path: str | Path | None = None,
                exclusions_path: str | Path | None = None) -> list[XMPair]:
    """只给数据统计用（scripts/xm/data_stats.py）：读真值、不跑任何系统，所以不过密封守卫（同 explore_splits"生成清单只写 id，不受守卫
    限制"）。运行器一律用 load_pairs。"""
    return _pairs(split, data_root, splits_path=splits_path, exclusions_path=exclusions_path)


__all__ = [
    "COCO_EVAL_N", "COCO_EVAL_SPLITS", "FINAL_GATED", "GATE", "LVIS_SUB", "REASONSEG_VAL", "SPLITS", "TRAIN", "XM_DECISION", "XM_FINAL",
    "XM_GATE_PASS", "XM_TUNE", "SealedSplitError", "XMDataError", "XMGateError", "XMGatePass", "XMInstance", "XMPair", "build_splits_doc",
    "check_run_dir", "check_splits_doc", "check_xm_gate", "coco_pairs", "dump_config", "load_pairs", "load_splits_doc", "lvis_pairs",
    "pair_by_id", "pair_from_dict", "pair_to_dict", "reasonseg_pairs", "require_unsealed",
    "require_xm_gate", "seal_identity", "segm_to_rle", "stats_pairs",
]
