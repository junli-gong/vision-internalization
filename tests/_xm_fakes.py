"""X-M 的假数据根（DEV-0079）：格式照 COCO 2017、LVIS v1、LISA 的 ReasonSeg 官方文件，图是 64×48 的小 JPEG，确定性生成。

    python -m tests._xm_fakes <dir>          # 写出 <dir>/coco2017、<dir>/lvis、<dir>/reasonseg（开发机彩排用）

RefCOCO 用已有的 tests/fixtures/explore/root（coco:901–906、911–926 是评测图；W2 与探索的 tiny 划分都在上面）。

构造（build_fake_root 返回这些编号，测试照它核对）：
- val2017：24 张图，id = 139、39769（D20 的探针用图，有标注）与 2001–2022。
  - 2001 没有标注；2002 只有一个 crowd 的 person；
  - 2003：person 两个非 crowd + 一个 crowd（crowd_rles 1 个），car 只有 crowd（这一对不建）；
  - 2004：dog 一个正常多边形 + 一个 4 个数的退化多边形（丢掉、记数）；
  - 其余每张 1–2 个类别、1–3 个实例（矩形或三角形多边形）。
- train2017：903、920（RefCOCO 的评测图，要剔除）、3001–3006（普通）、3007（字节 = ReasonSeg 的 rs_a.jpg，要剔除）、
  3008（字节 = xm_decision 的第一张图，要剔除）。3001 的 person 有 33 个实例（截到 30 个）。
- LVIS v1 val：val2017 的 2004–2022 与 139（coco_url 是 val2017/…），另有 3001、3002（coco_url 是 train2017/…，不能进子集）；
  类名带下划线（aerosol_can、bow_(weapon)）；每张图记 neg_category_ids 与 not_exhaustive_category_ids。
- ReasonSeg val：rs_a（长句，一个目标多边形）、rs_b（短语，两条 text，目标 + ignore + flag）、rs_c（长句，两个目标多边形）、
  000000003006（与 train2017 的 3006 同名、字节不同：按文件名命中）。
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
W, H = 64, 48
VAL_IDS = [139, 39769] + list(range(2001, 2023))
TRAIN_IDS = [903, 920] + list(range(3001, 3009))
CATS = [{"id": 1, "name": "person", "supercategory": "person"}, {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 18, "name": "dog", "supercategory": "animal"}, {"id": 62, "name": "chair", "supercategory": "furniture"}]
LVIS_CATS = [{"id": 1, "name": "aerosol_can", "synonyms": ["aerosol_can"], "frequency": "c"},
             {"id": 2, "name": "bow_(weapon)", "synonyms": ["bow_(weapon)"], "frequency": "r"},
             {"id": 3, "name": "dog", "synonyms": ["dog"], "frequency": "f"}]


def _jpeg(seed: int) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    a = np.stack([(x * 4 + seed) % 256, (y * 5 + 3 * seed) % 256, rng.integers(0, 256, (H, W))], axis=2).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _ann(aid: int, iid: int, cid: int, poly: list[float] | list[list[float]], *, crowd: bool = False) -> dict[str, Any]:
    from pycocotools import mask as M

    if crowd:
        m = np.zeros((H, W), np.uint8)
        xs, ys = poly[0::2], poly[1::2]
        m[int(min(ys)): int(max(ys)), int(min(xs)): int(max(xs))] = 1
        # 未压缩 RLE（COCO 的 crowd 格式）：列优先的游程
        flat = m.flatten(order="F")
        counts, cur, run = [], 0, 0
        for v in flat:
            if v != cur:
                counts.append(run)
                cur, run = v, 1
            else:
                run += 1
        counts.append(run)
        seg: Any = {"counts": counts, "size": [H, W]}
        rle = M.frPyObjects(seg, H, W)
    else:
        seg = poly if isinstance(poly[0], list) else [poly]
        rle = M.merge(M.frPyObjects([p for p in seg if len(p) >= 6], H, W))
    x, y, w, h = (float(v) for v in M.toBbox(rle))
    return {"id": aid, "image_id": iid, "category_id": cid, "segmentation": seg, "area": float(M.area(rle)), "bbox": [x, y, w, h],
            "iscrowd": int(crowd)}


def _coco_doc(images: list[dict[str, Any]], anns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"info": {"description": "X-M fake"}, "licenses": [], "images": images, "annotations": anns, "categories": CATS}


def _image_entry(iid: int, split: str) -> dict[str, Any]:
    fn = f"{iid:012d}.jpg"
    return {"id": iid, "file_name": fn, "height": H, "width": W, "coco_url": f"http://images.cocodataset.org/{split}/{fn}"}


def _val_anns() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    aid = [100000]

    def add(iid, cid, poly, crowd=False):
        aid[0] += 1
        out.append(_ann(aid[0], iid, cid, poly, crowd=crowd))

    add(2002, 1, _rect(5, 5, 30, 30), crowd=True)
    add(2003, 1, _rect(2, 2, 20, 30))
    add(2003, 1, _rect(40, 4, 60, 40))
    add(2003, 1, _rect(22, 10, 38, 44), crowd=True)
    add(2003, 3, _rect(0, 30, 63, 47), crowd=True)
    add(2004, 18, [[4, 4, 40, 4, 22, 40], [10, 10, 12, 12]])  # 第二个多边形只有 4 个数
    for k, iid in enumerate([139, 39769] + list(range(2005, 2023))):
        n = 1 + k % 3
        cid = (1, 3, 18, 62)[k % 4]
        for j in range(n):
            x0 = 2 + 20 * j
            add(iid, cid, _rect(x0, 3 + k % 5, x0 + 15, 25 + k % 7))
        if k % 2:
            add(iid, (3, 18, 62, 1)[k % 4], [30, 30, 60, 30, 45, 46])
    return out


def _train_anns() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    aid = [200000]

    def add(iid, cid, poly, crowd=False):
        aid[0] += 1
        out.append(_ann(aid[0], iid, cid, poly, crowd=crowd))

    for r in range(3):  # 3001：33 个 person（3 行 × 11 个），截到 30 个
        for c in range(11):
            add(3001, 1, _rect(1 + 5.5 * c, 2 + 15 * r, 5 + 5.5 * c, 13 + 15 * r))
    add(3001, 3, _rect(10, 10, 50, 40), crowd=True)
    for k, iid in enumerate([903, 920, 3002, 3003, 3004, 3005, 3006, 3007, 3008]):
        add(iid, (1, 3, 18, 62)[k % 4], _rect(3 + k, 4, 30 + k, 40))
        add(iid, 18, [35, 5, 60, 10, 50, 44])
    return out


def _lvis_doc(val_ids: list[int]) -> dict[str, Any]:
    from pycocotools import mask as M

    images, anns = [], []
    aid = 300000
    lvis_val = [i for i in val_ids if i not in (39769, 2001, 2002, 2003)]
    for k, iid in enumerate(lvis_val + [3001, 3002]):
        split = "train2017" if iid >= 3000 else "val2017"
        e = _image_entry(iid, split)
        del e["file_name"]  # LVIS v1 的 images 没有 file_name，只有 coco_url
        e.update(neg_category_ids=[2] if k % 2 else [], not_exhaustive_category_ids=[1] if k % 3 == 0 else [])
        images.append(e)
        for j, cid in enumerate((1, 3) if k % 2 == 0 else (3,)):
            aid += 1
            poly = _rect(4 + 20 * j, 4, 18 + 20 * j, 30)
            rle = M.merge(M.frPyObjects([poly], H, W))
            anns.append({"id": aid, "image_id": iid, "category_id": cid, "segmentation": [poly], "area": float(M.area(rle)),
                         "bbox": [float(v) for v in M.toBbox(rle)]})
    return {"info": {}, "licenses": [], "images": images, "annotations": anns, "categories": LVIS_CATS}


REASONSEG = {
    "rs_a": {"text": ["What object in the picture could be used to sit on while waiting?"], "is_sentence": True,
             "shapes": [{"label": "target", "labelme_type": "polygon", "points": [[5, 5], [30, 5], [30, 30], [5, 30]]}]},
    "rs_b": {"text": ["the red thing", "red object"], "is_sentence": False,
             "shapes": [{"label": "target", "labelme_type": "polygon", "points": [[10, 10], [50, 10], [50, 40], [10, 40]]},
                        {"label": "ignore", "labelme_type": "polygon", "points": [[20, 20], [30, 20], [30, 30], [20, 30]]},
                        {"label": "flag", "labelme_type": "polygon", "points": [[0, 0], [5, 0], [5, 5]]}]},
    "rs_c": {"text": ["Which items would a cat most likely knock over?"], "is_sentence": True,
             "shapes": [{"label": "target", "labelme_type": "polygon", "points": [[2, 2], [15, 2], [15, 20]]},
                        {"label": "target", "labelme_type": "polygon", "points": [[40, 20], [60, 20], [60, 45], [40, 45]]}]},
    "000000003006": {"text": ["the dog"], "is_sentence": False,
                     "shapes": [{"label": "target", "labelme_type": "polygon", "points": [[30, 10], [55, 10], [55, 40], [30, 40]]}]},
}


def build_fake_root(root: str | Path) -> dict[str, Any]:
    """写出假数据根；返回测试要核对的编号。"""
    from src.xm import data as D

    root = Path(root)
    for sub in ("coco2017", "lvis", "reasonseg"):
        if (root / sub).exists():
            shutil.rmtree(root / sub)
    ann_dir = root / "coco2017" / "annotations"
    ann_dir.mkdir(parents=True)
    for split, ids in (("val2017", VAL_IDS), ("train2017", TRAIN_IDS)):
        d = root / "coco2017" / "images" / split
        d.mkdir(parents=True)
        for iid in ids:
            (d / f"{iid:012d}.jpg").write_bytes(_jpeg(iid))
    val_doc = _coco_doc([_image_entry(i, "val2017") for i in VAL_IDS], _val_anns())
    (ann_dir / "instances_val2017.json").write_text(json.dumps(val_doc), encoding="utf-8")
    rs_dir = root / "reasonseg" / "val"
    rs_dir.mkdir(parents=True)
    for k, (stem, doc) in enumerate(REASONSEG.items()):
        (rs_dir / f"{stem}.jpg").write_bytes(_jpeg(9000 + k))
        (rs_dir / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
    # 3007 = rs_a 的字节；3008 = xm_decision 的第一张图（按同一套规则现算）
    tr_dir = root / "coco2017" / "images" / "train2017"
    shutil.copyfile(rs_dir / "rs_a.jpg", tr_dir / f"{3007:012d}.jpg")
    pop, _, _ = D.coco_population(val_doc, tiny=True)
    chosen = D.select_coco_splits(pop, D.COCO_EVAL_N_TINY)
    dup = chosen[D.XM_DECISION][0]
    shutil.copyfile(root / "coco2017" / "images" / "val2017" / f"{dup:012d}.jpg", tr_dir / f"{3008:012d}.jpg")
    tr_doc = _coco_doc([_image_entry(i, "train2017") for i in TRAIN_IDS], _train_anns())
    (ann_dir / "instances_train2017.json").write_text(json.dumps(tr_doc), encoding="utf-8")
    (root / "lvis").mkdir(parents=True)
    (root / "lvis" / "lvis_v1_val.json").write_text(json.dumps(_lvis_doc(VAL_IDS)), encoding="utf-8")
    return {"val_ids": VAL_IDS, "train_ids": TRAIN_IDS, "probe_ids": [139, 39769], "refcoco_in_train": [903, 920],
            "reasonseg_dup_train": 3007, "reasonseg_name_train": 3006, "split_dup_train": 3008, "split_dup_val": dup,
            "lvis_train_ids": [3001, 3002], "chosen": chosen}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("用法：python -m tests._xm_fakes <dir>", file=sys.stderr)
        return 2
    info = build_fake_root(argv[0])
    print(json.dumps({k: v for k, v in info.items() if k != "chosen"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
