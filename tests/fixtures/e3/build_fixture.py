"""生成 tests/fixtures/e3/root（DEV-0025）：E3b --tiny 彩排用的 RefCOCO 家族训练小数据，布局与 DEV-0022 的落地完全相同。

    conda run -n visint python tests/fixtures/e3/build_fixture.py

E2 的 fixture（tests/fixtures/e2/root/refcoco_train）只有 24×32 的图，切不出 32×32 的网格（codec.cell_edges 要求边长 ≥ 32），
也只有 RefCOCO 一个数据集；这里另造一份：
- refcoco_train：4 张 96×128 的 JPEG（平滑底色上画了几个色块，色块就是被指代的物体），RefCOCO（unc）、RefCOCO+（unc）、
  RefCOCOg（umd）三份 train 标注共用这些图（与真实数据相同：三者都来自 COCO train2014）。每张图 2–3 个物体（多边形 mask，
  "与邻近物体合并"有候选），其中一个物体的 mask 是 RLE（C3 的多边形目标退回框）；有一句指代句只有空白（加载后剔除并计数）。
- dedupe/train_exclusions.json：新格式（schema 2），refcoco_train/train 与 nyu_depth_train/train 都有键（不剔除），eval_sets 与 RUN-0026
  的剔除清单相同（13 个，含 configs/e3b_train.json 的 dedupe.required_eval_sets；DEV-0032）。E3b 的 --tiny 两个训练池都用这一份
  （src/e3/data.E3Paths.fixtures）。
NYU 深度的训练数据用 tests/fixtures/e2/root（480×640，已足够）。数值全部由固定种子生成；重跑逐字节相同。
DEV-0030 另造 eval_root/refcoco（E3b 评测 --tiny 的 RefCOCO val，E1 落地的布局；E1 的 fixture 只有 12×9 的图，同样切不出
32×32）与 eval_dev_subsets.json（它的开发子集清单），见 build_refcoco_eval；NYU 的官方测试图用 tests/fixtures/e1/root/nyuv2。
训练数据的随机数序列不变（评测数据用另一个种子），重跑后 root/ 逐字节不变。
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.data.dedupe import phash_bytes  # noqa: E402
from src.executors.raster import mask_to_rle, polygons_to_mask  # noqa: E402

ROOT = HERE / "root"
EVAL_ROOT = HERE / "eval_root"  # DEV-0030：E3b 评测的 RefCOCO val 小数据（E1 落地的布局）
EVAL_DEV = HERE / "eval_dev_subsets.json"  # 它的开发子集清单（E1 的 dev_subsets 格式）
H, W = 96, 128
# RUN-0026 剔除清单（experiments/P0/0026_a800_delta/results_0026/dedupe/train_exclusions.json）的 eval_sets，逐字照抄（tests 核对）
EVAL_SETS = ["ade20k/val", "blink_reldepth/val", "coco_val2017/e1sub500", "coco_val2017_full/val2017", "countbench/eval", "cvbench/test",
             "embspatial/test", "nyu_normals/test", "nyuv2/dev_train", "nyuv2/test", "odinw13/eval", "probe/D20", "refcoco/eval"]
PALETTE = [(220, 40, 40), (40, 180, 60), (50, 70, 220), (230, 200, 30)]


def _jpeg(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8), mode="RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class Src:
    def __init__(self, name: str, root: Path = ROOT) -> None:
        self.name = name
        self.dir = root / name
        self.files: dict[str, dict] = {}

    def put(self, rel: str, data: bytes, kind: str, image: bool = False) -> None:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rec = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "kind": kind}
        if image:
            rec["phash"] = phash_bytes(data)
        self.files[rel] = rec

    def put_json(self, rel: str, obj) -> None:
        self.put(rel, (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"), "metadata")

    def finish(self) -> None:
        doc = {"schema": 1, "source": self.name, "fixture": "tests/fixtures/e3/build_fixture.py（合成数据）",
               "files": dict(sorted(self.files.items()))}
        (self.dir / "MANIFEST.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def _shape(rng: np.random.Generator) -> list[float]:
    """随机的凸多边形（矩形、三角形或五边形），COCO 扁平坐标。"""
    cx, cy = float(rng.uniform(25, W - 25)), float(rng.uniform(20, H - 20))
    rx, ry = float(rng.uniform(10, 24)), float(rng.uniform(8, 18))
    k = int(rng.choice([3, 4, 5]))
    ang = float(rng.uniform(0, 2 * np.pi)) + 2 * np.pi * np.arange(k) / k  # 等角距：不会退化
    pts = []
    for a in ang:
        pts += [round(cx + rx * float(np.cos(a)), 2), round(cy + ry * float(np.sin(a)), 2)]
    return pts


def build_refcoco_train(rng: np.random.Generator) -> None:
    s = Src("refcoco_train")
    images, anns = {}, {}
    objects = []  # (image_id, ann_id, category)
    for k, iid in enumerate((2001, 2002, 2003, 2004)):
        fn = f"COCO_train2014_{iid:012d}.jpg"
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        img = np.stack([90 + 40 * np.sin(xx / W * 3 + k), 100 + 30 * np.cos(yy / H * 2 + k), 110 + 20 * np.sin((xx + yy) / 60)],
                       axis=-1)
        n_obj = 3 if k % 2 == 0 else 2
        for j in range(n_obj):
            aid = 30000 + 10 * k + j
            poly = _shape(rng)
            m = polygons_to_mask([poly], H, W)
            img[m] = PALETTE[j % len(PALETTE)]
            ys, xs = np.nonzero(m)
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
            seg = mask_to_rle(m) if (k, j) == (1, 1) else [poly]  # 一个 RLE 的物体
            anns[str(aid)] = {"id": aid, "image_id": iid, "category_id": 1 + j, "bbox": [x0, y0, x1 - x0, y1 - y0],
                              "area": float(m.sum()), "iscrowd": 0, "segmentation": seg}
            objects.append((iid, aid, 1 + j))
        s.put(f"images/{fn}", _jpeg(np.clip(img, 0, 255)), "image", image=True)
        images[str(iid)] = {"id": iid, "file_name": fn, "height": H, "width": W}
    sent = 70000
    for ds, split_by, n_sent in (("refcoco", "unc", 2), ("refcoco+", "unc", 1), ("refcocog", "umd", 1)):
        refs = []
        for ref_id, (iid, aid, cat) in enumerate(objects):
            sents = []
            for _ in range(n_sent):
                text = f"the {['red', 'green', 'blue'][cat - 1]} shape number {sent}"
                if ds == "refcoco+" and ref_id == 0:
                    text = "   "  # 只有空白的指代句：E3 的加载剔除并计数
                sents.append({"sent_id": sent, "sent": text, "raw": text})
                sent += 1
            refs.append({"ref_id": ref_id, "ann_id": aid, "image_id": iid, "category_id": cat, "split": "train",
                         "sentences": sents})
        doc = {"dataset": ds, "splitBy": split_by, "split": "train", "refs": refs, "anns": anns, "images": images,
               "categories": {"1": "person", "2": "dog", "3": "cat"}}
        s.put_json(f"{ds}_{split_by}_train.json", doc)
    s.finish()


def build_exclusions() -> None:
    d = ROOT / "dedupe"
    d.mkdir(parents=True, exist_ok=True)
    doc = {"schema": 2, "created_utc": "fixture", "max_hamming": 8,
           "rule": "fixture：E3 的 RefCOCO 家族小数据（真实清单由 scripts/data/dedupe_report.py 生成）",
           "train_sets": ["nyu_depth_train/train", "refcoco_train/train"], "eval_sets": EVAL_SETS,
           "inputs": [{"what": "fixture", "host": None, "root": None, "created_utc": "fixture", "git_commit": None, "sha256": None,
                       "sets_sha256": None, "sets": {}}],
           "sets": {"nyu_depth_train/train": [], "refcoco_train/train": []}}
    (d / "train_exclusions.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def build_refcoco_eval(rng: np.random.Generator) -> None:
    """DEV-0030：E3b 评测（scripts/eval_e3b.py --tiny）用的 RefCOCO（unc）val 小数据，布局与 E1 的 refcoco 落地相同
    （src/data/loaders.load_refcoco 读 refcoco/refcoco_unc.json 与 MANIFEST）。3 张 96×128 的图（与训练图不同的 image id），
    每张 2 个物体（person / dog，分割按人物分列有两组），其中一个物体的 mask 是 RLE；6 个 ref、7 句，其中 1 句在开发子集里
    （eval_dev_subsets.json 的 refcoco_unc_val，load_refcoco(subset=eval) 剔除），剩 6 句。"""
    s = Src("refcoco", root=EVAL_ROOT)
    images, anns, refs = {}, {}, []
    sent = 80000
    for k, iid in enumerate((3001, 3002, 3003)):
        fn = f"COCO_train2014_{iid:012d}.jpg"
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        img = np.stack([70 + 50 * np.cos(xx / W * 2 + k), 120 + 25 * np.sin(yy / H * 3 + k), 90 + 30 * np.cos((xx - yy) / 50)],
                       axis=-1)
        for j in range(2):
            aid = 40000 + 10 * k + j
            poly = _shape(rng)
            m = polygons_to_mask([poly], H, W)
            img[m] = PALETTE[(j + k) % len(PALETTE)]
            ys, xs = np.nonzero(m)
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
            seg = mask_to_rle(m) if (k, j) == (2, 0) else [poly]
            cat = 1 + j
            anns[str(aid)] = {"id": aid, "image_id": iid, "category_id": cat, "bbox": [x0, y0, x1 - x0, y1 - y0],
                              "area": float(m.sum()), "iscrowd": 0, "segmentation": seg}
            n_sent = 2 if (k, j) == (0, 0) else 1
            sents = []
            for _ in range(n_sent):
                text = f"the {['person', 'dog'][j]} in picture {k} number {sent}"
                sents.append({"sent_id": sent, "sent": text, "raw": text.capitalize()})
                sent += 1
            refs.append({"ref_id": 500 + len(refs), "ann_id": aid, "image_id": iid, "category_id": cat, "split": "val",
                         "sentences": sents})
        s.put(f"images/{fn}", _jpeg(np.clip(img, 0, 255)), "image", image=True)
        images[str(iid)] = {"id": iid, "file_name": fn, "height": H, "width": W}
    s.put_json("refcoco_unc.json", {"dataset": "refcoco", "splitBy": "unc", "eval_splits": ["val", "testA", "testB"],
                                    "refs": refs, "anns": anns, "images": images, "categories": {"1": "person", "2": "dog"}})
    s.finish()
    dev = {"version": 1, "salt": "fixture", "method": "fixture（tests/fixtures/e3/build_fixture.py）",
           "usage": "E3b 评测 --tiny：refcoco_unc_val 的开发子集（load_refcoco(subset=eval) 剔除）",
           "subsets": {"refcoco_unc_val": {"unit": "sentence(sent_id)", "rule": "fixture：第 1 句", "n_population": 7,
                                           "population_sha256": None, "population_provenance": "", "k": 1, "n_remaining": 6,
                                           "ids": ["80001"]}}}
    EVAL_DEV.write_text(json.dumps(dev, ensure_ascii=False, indent=1) + "\n")


def main() -> None:
    for d in (ROOT, EVAL_ROOT):
        if d.exists():
            shutil.rmtree(d)
    rng = np.random.default_rng(20261003)
    build_refcoco_train(rng)
    build_exclusions()
    build_refcoco_eval(np.random.default_rng(20261004))
    for d in (ROOT, EVAL_ROOT):
        total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        print(f"{d}：{total} B")


if __name__ == "__main__":
    main()
