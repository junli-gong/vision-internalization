"""生成 tests/fixtures/w2c/root（DEV-0060）：W2-c 训练数据 --tiny 用的 RefCOCO 家族训练小数据，布局与 DEV-0022 的落地相同。

    conda run -n visint python tests/fixtures/w2c/build_fixture.py

以 tests/fixtures/e3/root 为底（4 张 96×128 的图 coco:2001–2004，RefCOCO（unc）、RefCOCO+（unc）、RefCOCOg（umd）三份 train 标注，
一句只有空白），另加三张"要被去重"的图：
- coco:901（RefCOCO 2 句）与 coco:905（RefCOCOg 1 句）：与 W2 --tiny 划分的评测图（tests/fixtures/w2/root：901 在 devtune800、905 在
  devcheck2）同一个 COCO id。图与标注从 tests/fixtures/w2/root/refcoco 原样复制（48×36）。"按图像 id 排除 W2 评测图"那一步要去掉这 3 句。
- coco:2005（RefCOCO 2 句、RefCOCO+ 1 句）：图是 2001 的拷贝；dedupe/train_exclusions.json 的 refcoco_train/train 里有它（E3 训练池的
  剔除清单），训练加载器去掉这 3 句。
剔除清单其余与 E3 的 fixture 相同（schema 2，eval_sets 同 RUN-0026，含 configs/e3b_train.json 的 dedupe.required_eval_sets）。
全部由已有的 fixture 拷贝拼成，重跑逐字节相同。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E3_ROOT = REPO / "tests" / "fixtures" / "e3" / "root"
W2_REFCOCO = REPO / "tests" / "fixtures" / "w2" / "root" / "refcoco"
ROOT = HERE / "root"
SOURCE = "refcoco_train"
FILES = {"refcoco": "refcoco_unc_train.json", "refcoco+": "refcoco+_unc_train.json", "refcocog": "refcocog_umd_train.json"}
# (数据集, 图像 id, 标注 id, 句子 id 列表)；标注从哪里来见 _extra_anns
EXTRA_REFS = [("refcoco", 901, 7000, [70100, 70101]), ("refcocog", 905, 7005, [70110]),
              ("refcoco", 2005, 30100, [70120, 70121]), ("refcoco+", 2005, 30100, [70122])]
EXCLUDED = {"path": "images/COCO_train2014_000000002005.jpg", "key": "coco:2005",
            "reasons": [{"eval_set": "refcoco/eval", "by": "phash", "matched": "coco:3001", "hamming": 0}]}


def _fn(iid: int) -> str:
    return f"COCO_train2014_{iid:012d}.jpg"


def _extra_anns() -> tuple[dict[str, dict], dict[str, dict], dict[str, bytes], dict[str, str]]:
    """新加的标注、图像记录、图像字节、感知哈希。901、905 取 W2 fixture 的原样；2005 是 2001 的拷贝（标注 30100 = 30000 换图）。"""
    w2 = json.loads((W2_REFCOCO / "refcoco_unc.json").read_text())
    w2_man = json.loads((W2_REFCOCO / "MANIFEST.json").read_text())["files"]
    e3 = json.loads((E3_ROOT / SOURCE / FILES["refcoco"]).read_text())
    e3_man = json.loads((E3_ROOT / SOURCE / "MANIFEST.json").read_text())["files"]
    anns, images, data, phash = {}, {}, {}, {}
    for iid, aid in ((901, 7000), (905, 7005)):
        anns[str(aid)] = copy.deepcopy(w2["anns"][str(aid)])
        images[str(iid)] = copy.deepcopy(w2["images"][str(iid)])
        rel = f"images/{_fn(iid)}"
        data[rel] = (W2_REFCOCO / rel).read_bytes()
        phash[rel] = w2_man[rel]["phash"]
    a = copy.deepcopy(e3["anns"]["30000"])
    a.update(id=30100, image_id=2005)
    anns["30100"] = a
    images["2005"] = {**copy.deepcopy(e3["images"]["2001"]), "id": 2005, "file_name": _fn(2005)}
    src_rel = f"images/{_fn(2001)}"
    data[f"images/{_fn(2005)}"] = (E3_ROOT / SOURCE / src_rel).read_bytes()
    phash[f"images/{_fn(2005)}"] = e3_man[src_rel]["phash"]
    return anns, images, data, phash


def build() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    src = ROOT / SOURCE
    (src / "images").mkdir(parents=True)
    e3_man = json.loads((E3_ROOT / SOURCE / "MANIFEST.json").read_text())
    files: dict[str, dict] = {}
    for rel, rec in e3_man["files"].items():  # E3 的 4 张图原样
        if rec["kind"] == "image":
            (src / rel).write_bytes((E3_ROOT / SOURCE / rel).read_bytes())
            files[rel] = dict(rec)
    anns, images, data, phash = _extra_anns()
    for rel, b in data.items():
        (src / rel).write_bytes(b)
        files[rel] = {"path": rel, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "kind": "image", "phash": phash[rel]}
    for ds, fname in FILES.items():
        doc = json.loads((E3_ROOT / SOURCE / fname).read_text())
        doc["anns"].update(copy.deepcopy(anns))
        doc["images"].update(copy.deepcopy(images))
        for k, (dsx, iid, aid, sents) in enumerate(EXTRA_REFS):
            if dsx != ds:
                continue
            cat = int(anns[str(aid)]["category_id"])
            doc["refs"].append({"ref_id": 100 + k, "ann_id": aid, "image_id": iid, "category_id": cat, "split": "train",
                                "sentences": [{"sent_id": s, "sent": f"the thing on image {iid} number {s}",
                                               "raw": f"The thing on image {iid} number {s}"} for s in sents]})
        b = (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        (src / fname).write_bytes(b)
        files[fname] = {"path": fname, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "kind": "metadata"}
    man = {"schema": 1, "source": SOURCE, "fixture": "tests/fixtures/w2c/build_fixture.py（由 e3、w2 的 fixture 拼成）",
           "files": dict(sorted(files.items()))}
    (src / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    excl = json.loads((E3_ROOT / "dedupe" / "train_exclusions.json").read_text())
    excl["rule"] = "fixture：W2-c 的 RefCOCO 家族小数据（coco:2005 按剔除清单剔除；真实清单由 scripts/data/dedupe_report.py 生成）"
    excl["sets"]["refcoco_train/train"] = [EXCLUDED]
    (ROOT / "dedupe").mkdir()
    (ROOT / "dedupe" / "train_exclusions.json").write_text(json.dumps(excl, ensure_ascii=False, indent=1) + "\n")
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"{ROOT}：{total} B")


if __name__ == "__main__":
    build()
