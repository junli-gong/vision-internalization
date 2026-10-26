"""生成 tests/fixtures/explore/（DEV-0067）：devcheck3 与 explore_val 的 --tiny 用的 RefCOCO 小数据与 E1 评测清单。

    conda run -n visint python tests/fixtures/explore/build_fixture.py

以 tests/fixtures/w2/root/refcoco 为底（6 张 48×36 的图 coco:901–906；W2 的 --tiny 划分全在这 6 张图上：devtune800 在 901、902，
devcheck2 在 905、906，5 个 test 子集在 903、904），原样复制，再加：
- 16 张新的 val 图 coco:911–926：图是 901–906 依次循环的拷贝（字节与感知哈希相同；同 tests/fixtures/w2c 的做法），标注是源图上
  RefCOCO 第一个标注换成新的 id 与图；
- RefCOCO val：911–924 各一个 ref、两句（sent 120–147）；RefCOCO+ val：同样 911–924 各两句（sent 320–347；两者共用图，同真实数据）；
  RefCOCOg val：915–926 各两句（sent 420–443），另在 coco:903 上两句（sent 450、451）——903 是 W2 的 test_refcoco_unc_testA 子集的图，
  这两句要按图排除（"别的数据集的 test 子集的图也在 val 里"，真实数据里 refcocog 的 val 与 refcoco 的 testA、testB 共用一些图）。
  注意：903 在这里同时也是 refcocog 的 test 图（真实数据里 refcocog 自己的 val 与 test 按图不相交），只为造出上面那种情形。
- eval_subsets.json：tests/fixtures/e1/eval_subsets.json 原样，三个 val 总体的条数与 sha256 换成本 fixture 的（开发子集用
  tests/fixtures/e1/dev_subsets.json：101、300、403），poly 子集各加几句新句子（按句排除那一步要去掉它们）：
  poly:refcoco_unc_val + 120、121，poly:refcoco+_unc_val + 320，poly:refcocog_umd_val + 420，extra:refcoco_unc_val + 122。
全部由已有的 fixture 拷贝拼成，重跑逐字节相同。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.data.subsets import population_digest, sort_ids  # noqa: E402

W2_REFCOCO = REPO / "tests" / "fixtures" / "w2" / "root" / "refcoco"
E1_FIXTURE = REPO / "tests" / "fixtures" / "e1"
ROOT = HERE / "root"
FILES = {"refcoco": "refcoco_unc.json", "refcoco+": "refcoco+_unc.json", "refcocog": "refcocog_umd.json"}
POP_OF = {"refcoco": "refcoco_unc_val", "refcoco+": "refcoco+_unc_val", "refcocog": "refcocog_umd_val"}
NEW_IMAGES = list(range(911, 927))
SOURCE_OF = {iid: 901 + k % 6 for k, iid in enumerate(NEW_IMAGES)}  # 新图 → 拷贝自哪张图
# (数据集, 图像 id, 句子 id 列表)
NEW_REFS = ([("refcoco", iid, [120 + 2 * k, 121 + 2 * k]) for k, iid in enumerate(range(911, 925))]
            + [("refcoco+", iid, [320 + 2 * k, 321 + 2 * k]) for k, iid in enumerate(range(911, 925))]
            + [("refcocog", iid, [420 + 2 * k, 421 + 2 * k]) for k, iid in enumerate(range(915, 927))]
            + [("refcocog", 903, [450, 451])])
ANN_BASE = {"refcoco": 7100, "refcoco+": 8310, "refcocog": 8410}
EXTRA_E1_IDS = {"poly:refcoco_unc_val": ["120", "121"], "poly:refcoco+_unc_val": ["320"], "poly:refcocog_umd_val": ["420"],
                "extra:refcoco_unc_val": ["122"]}


def _fn(iid: int) -> str:
    return f"COCO_train2014_{iid:012d}.jpg"


def build() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    src = ROOT / "refcoco"
    (src / "images").mkdir(parents=True)
    w2_man = json.loads((W2_REFCOCO / "MANIFEST.json").read_text())
    files: dict[str, dict] = {}
    for rel, rec in w2_man["files"].items():
        if rec["kind"] == "image":
            (src / rel).write_bytes((W2_REFCOCO / rel).read_bytes())
            files[rel] = dict(rec)
    for iid in NEW_IMAGES:
        s_rel, rel = f"images/{_fn(SOURCE_OF[iid])}", f"images/{_fn(iid)}"
        b = (W2_REFCOCO / s_rel).read_bytes()
        (src / rel).write_bytes(b)
        files[rel] = {**w2_man["files"][s_rel], "path": rel, "image_key": f"coco:{iid}"}
    base = json.loads((W2_REFCOCO / FILES["refcoco"]).read_text())
    template = {}  # 图 → RefCOCO 在这张图上的第一个标注
    for a in sorted(base["anns"].values(), key=lambda a: a["id"]):
        template.setdefault(int(a["image_id"]), a)
    sents: dict[str, list[str]] = {ds: [] for ds in FILES}
    for ds, fname in FILES.items():
        doc = json.loads((W2_REFCOCO / fname).read_text())
        next_ref = max(r["ref_id"] for r in doc["refs"]) + 1
        k_ann = 0
        for dsx, iid, sids in NEW_REFS:
            if dsx != ds:
                continue
            src_iid = SOURCE_OF.get(iid, iid)
            aid = ANN_BASE[ds] + k_ann
            k_ann += 1
            a = copy.deepcopy(template[src_iid])
            a.update(id=aid, image_id=iid)
            if str(a["category_id"]) not in doc["categories"]:  # RefCOCO+、RefCOCOg 的 fixture 只有 person
                a["category_id"] = int(min(doc["categories"], key=int))
            doc["anns"][str(aid)] = a
            if str(iid) not in doc["images"]:
                doc["images"][str(iid)] = {**copy.deepcopy(base["images"][str(src_iid)]), "id": iid, "file_name": _fn(iid)}
            doc["refs"].append({"ref_id": next_ref, "ann_id": aid, "image_id": iid, "category_id": int(a["category_id"]), "split": "val",
                                "sentences": [{"sent_id": s, "sent": f"{ds} thing {s} on image {iid}", "raw": f"{ds} thing {s} on image {iid}"}
                                              for s in sids]})
            next_ref += 1
        sents[ds] = [str(s["sent_id"]) for r in doc["refs"] if r["split"] == "val" for s in r["sentences"]]
        b = (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        (src / fname).write_bytes(b)
        files[fname] = {"path": fname, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "kind": "metadata"}
    man = {"schema": 1, "source": "refcoco", "fixture": "tests/fixtures/explore/build_fixture.py（由 w2 的 fixture 拼成）",
           "files": dict(sorted(files.items()))}
    (src / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n")
    # E1 的评测清单：三个 val 总体换成本 fixture 的（剔除 E1 fixture 的开发子集），poly / extra 子集加几句新句子
    dev = json.loads((E1_FIXTURE / "dev_subsets.json").read_text())["subsets"]
    ev = json.loads((E1_FIXTURE / "eval_subsets.json").read_text())
    for ds, pop in POP_OF.items():
        ids = sort_ids(s for s in sents[ds] if s not in set(dev[pop]["ids"]))
        ev["populations"][pop].update(n=len(ids), sha256=population_digest(ids))
    for name, extra in EXTRA_E1_IDS.items():
        e = ev["subsets"][name]
        e["ids"] = sort_ids(set(e["ids"]) | set(extra))
        e["k"] = e["k_target"] = len(e["ids"])
    for e in ev["subsets"].values():
        p = ev["populations"].get(e["population"])
        if p is not None and e["population"] in POP_OF.values():
            e.update(n_population=p["n"], population_sha256=p["sha256"])
    ev["fixture"] = "tests/fixtures/explore/build_fixture.py（tests/fixtures/e1/eval_subsets.json 改三个 val 总体与 4 个子集）"
    (HERE / "eval_subsets.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1) + "\n")
    total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    print(f"{ROOT}：{total} B；{HERE / 'eval_subsets.json'}")


if __name__ == "__main__":
    build()
