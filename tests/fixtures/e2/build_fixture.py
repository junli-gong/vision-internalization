"""生成 tests/fixtures/e2/root（DEV-0023）：E2 --tiny 彩排用的合成小数据，布局与 DEV-0022 的落地完全相同。

    conda run -n visint python tests/fixtures/e2/build_fixture.py

- nyu_depth_train：5 个场景目录（4 个基础场景）× 2 帧，480×640 的 JPEG + 16 位深度 PNG（毫米，边框为 0 = 无效）；
- ade20k：train 6 张、val 3 张，48×64 的 JPEG + mode L 标注（0 与若干类别）；
- nyu_normals：train 5 张（idx 3–7）、test 2 张（idx 1、2），480×640 的 RGB PNG + 法向 PNG（四周无效）；
- refcoco_train：refcoco（unc）train 4 张 24×32 的图、5 个 ref、10 句；
- dedupe/train_exclusions.json：剔除 ade20k 的一张训练图与 nyu_normals 的 idx 3（与真实数据里"E1 开发子集按评测集剔除"同形）；
  新格式（DEV-0024）：四个训练集都有键，记 eval_sets 与 max_hamming。
数值全部由固定种子生成；重跑逐字节相同（PNG / JPEG 的编码参数固定）。
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

ROOT = HERE / "root"
H, W = 480, 640


def _jpeg(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8), mode="RGB").save(buf, format="JPEG", quality=60)
    return buf.getvalue()


def _png(a: np.ndarray, mode: str) -> bytes:
    im = Image.fromarray(a)  # mode 由 dtype 与形状决定（uint16 → I;16，uint8 二维 → L，uint8 三维 → RGB）
    if im.mode != mode:
        raise ValueError(f"PNG 的 mode {im.mode} ≠ {mode}")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=9)
    return buf.getvalue()


class Src:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = ROOT / name
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

    def put_jsonl(self, rel: str, rows) -> None:
        self.put(rel, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"), "metadata")

    def finish(self) -> None:
        doc = {"schema": 1, "source": self.name, "fixture": "tests/fixtures/e2/build_fixture.py（合成数据）",
               "files": dict(sorted(self.files.items()))}
        (self.dir / "MANIFEST.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def smooth_rgb(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.zeros((h, w, 3))
    for c in range(3):
        a, b, ph = rng.uniform(0.5, 3.0), rng.uniform(0.5, 3.0), rng.uniform(0, 6.28)
        out[..., c] = 127 + 100 * np.sin(a * xx / w * 6.28 + ph) * np.cos(b * yy / h * 6.28)
    return np.clip(out, 0, 255)


def depth_map(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    d = 1.0 + 2.5 * (yy / H) + 0.5 * rng.uniform() * (xx / W)
    mm = (np.round(d * 20) * 50).astype(np.uint16)  # 按 50 mm 量化，PNG 压得小
    mm[:12, :] = 0
    mm[:, :8] = 0
    return mm


def normal_map(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    n = np.stack([np.sin((xx / W - 0.5) * rng.uniform(1, 2)), np.sin((yy / H - 0.5) * rng.uniform(1, 2)),
                  -np.ones((H, W))], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgb = np.round((n + 1) / 2 * 255).astype(np.uint8)
    rgb = (rgb // 16) * 16 + 8  # 量化，PNG 压得小（解码后照样单位化）
    rgb[:40, :] = 0
    rgb[-40:, :] = 0
    rgb[:, :40] = 0
    return rgb


def build_nyu_depth_train(rng) -> None:
    s = Src("nyu_depth_train")
    rows = []
    # 基础场景都不在官方 test 场景（nyu_depth_train.TEST_SCENES）里：load_nyu_depth_train 复查 D11-2（DEV-0024）
    for scene, base in [("bedroom_0001a", "bedroom_0001"), ("bedroom_0001b", "bedroom_0001"), ("kitchen_0003a", "kitchen_0003"),
                        ("office_0003", "office_0003"), ("living_room_0004", "living_room_0004")]:
        for fr in ("00001", "00002"):
            rgb, dep = f"sync/{scene}/rgb_{fr}.jpg", f"sync/{scene}/sync_depth_{fr}.png"
            s.put(rgb, _jpeg(smooth_rgb(rng, H, W)), "image", image=True)
            s.put(dep, _png(depth_map(rng), "I;16"), "depth")
            rows.append({"id": f"{scene}/{fr}", "scene": scene, "base_scene": base, "frame": fr, "rgb": rgb, "depth": dep,
                         "focal": 518.8579, "rgb_sha256": s.files[rgb]["sha256"], "depth_sha256": s.files[dep]["sha256"]})
    s.put_jsonl("pairs_train.jsonl", rows)
    s.finish()


def build_ade(rng) -> None:
    s = Src("ade20k")
    h, w = 48, 64
    for sp, d, n in (("train", "training", 6), ("val", "validation", 3)):
        rows = []
        for i in range(1, n + 1):
            stem = f"ADE_{sp}_{i:08d}"
            lab = np.zeros((h, w), dtype=np.uint8)
            classes = [1, 5, 12, 150]
            for _ in range(5):
                r0, c0 = int(rng.integers(0, h - 8)), int(rng.integers(0, w - 8))
                lab[r0:r0 + int(rng.integers(8, 24)), c0:c0 + int(rng.integers(8, 32))] = classes[int(rng.integers(0, 4))]
            img, ann = f"images/{d}/{stem}.jpg", f"annotations/{d}/{stem}.png"
            s.put(img, _jpeg(smooth_rgb(rng, h, w)), "image", image=True)
            s.put(ann, _png(lab, "L"), "label")
            rows.append({"stem": stem, "split": sp, "image": img, "label": ann, "width": w, "height": h, "scene": "fixture",
                         "image_sha256": s.files[img]["sha256"], "label_sha256": s.files[ann]["sha256"]})
        s.put_jsonl(f"{sp}.jsonl", rows)
    s.finish()


def build_normals(rng) -> None:
    s = Src("nyu_normals")
    rows = []
    for sp, ids in (("train", (3, 4, 5, 6, 7)), ("test", (1, 2))):
        for idx in ids:
            img, nrm = f"{sp}/img/{idx:04d}.png", f"{sp}/norm/{idx:04d}.png"
            s.put(img, _png(((smooth_rgb(rng, H, W) // 32) * 32).astype(np.uint8), "RGB"), "image", image=True)
            s.put(nrm, _png(normal_map(rng), "RGB"), "normal")
            rows.append({"idx": idx, "split": sp, "img": img, "norm": nrm, "img_sha256": s.files[img]["sha256"],
                         "norm_sha256": s.files[nrm]["sha256"]})
    s.put_json("index.json", {"images": rows})
    s.finish()


def build_refcoco_train(rng) -> None:
    s = Src("refcoco_train")
    h, w = 24, 32
    images, anns, refs = {}, {}, []
    sent = 5000
    ref_id = 0
    for k, iid in enumerate((1001, 1002, 1003, 1004)):
        fn = f"COCO_train2014_{iid:012d}.jpg"
        s.put(f"images/{fn}", _jpeg(smooth_rgb(rng, h, w)), "image", image=True)
        images[str(iid)] = {"id": iid, "file_name": fn, "height": h, "width": w}
        for j in range(1 if k % 2 else 2):
            aid = 9000 + 10 * k + j
            x0, y0 = float(rng.integers(1, 12)), float(rng.integers(1, 10))
            x1, y1 = x0 + float(rng.integers(6, 18)), y0 + float(rng.integers(5, 12))
            anns[str(aid)] = {"id": aid, "image_id": iid, "category_id": 1 + j, "bbox": [x0, y0, x1 - x0, y1 - y0],
                              "area": (x1 - x0) * (y1 - y0), "iscrowd": 0,
                              "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]}
            sents = []
            for _ in range(2):
                sents.append({"sent_id": sent, "sent": f"the object number {sent}", "raw": f"The object number {sent}"})
                sent += 1
            refs.append({"ref_id": ref_id, "ann_id": aid, "image_id": iid, "category_id": 1 + j, "split": "train",
                         "sentences": sents})
            ref_id += 1
    doc = {"dataset": "refcoco", "splitBy": "unc", "split": "train", "refs": refs, "anns": anns, "images": images,
           "categories": {"1": "person", "2": "dog"}}
    s.put_json("refcoco_unc_train.json", doc)
    s.finish()


def build_exclusions() -> None:
    """新格式（train_dedupe.EXCLUSIONS_SCHEMA = 2）：四个训练集都有键（没剔除的是空列表），eval_sets 含 E2 四个任务的评测集
    （src/e2/data.eval_dedupe_sets，tests 核对）与两个内置评测集。"""
    d = ROOT / "dedupe"
    d.mkdir(parents=True, exist_ok=True)
    train_sets = ["ade20k/train", "nyu_depth_train/train", "nyu_normals/train", "refcoco_train/train"]
    eval_sets = ["ade20k/val", "coco_val2017_full/val2017", "nyu_normals/test", "nyuv2/test", "probe/D20", "refcoco/eval"]
    doc = {"schema": 2, "created_utc": "fixture", "max_hamming": 8,
           "rule": "fixture：演示加载器按清单剔除（真实清单由 scripts/data/dedupe_report.py 生成）",
           "train_sets": train_sets, "eval_sets": eval_sets,
           "inputs": [{"what": "fixture", "host": None, "root": None, "created_utc": "fixture", "git_commit": None, "sha256": None,
                       "sets_sha256": None, "sets": {}}],
           "sets": {"ade20k/train": [{"path": "images/training/ADE_train_00000006.jpg", "key": "ade20k:ADE_train_00000006",
                                       "reasons": ["fixture：演示剔除"]}],
                    "nyu_depth_train/train": [],
                    "nyu_normals/train": [{"path": "train/img/0003.png", "key": "nyuv2:3",
                                           "reasons": ["fixture：E1 开发子集按评测集剔除（D49-4 同形）"]}],
                    "refcoco_train/train": []}}
    (d / "train_exclusions.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = np.random.default_rng(20261002)
    build_nyu_depth_train(rng)
    build_ade(rng)
    build_normals(rng)
    build_refcoco_train(rng)
    build_exclusions()
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"{ROOT}：{total} B")


if __name__ == "__main__":
    main()
