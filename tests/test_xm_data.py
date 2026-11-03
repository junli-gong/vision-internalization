"""DEV-0079：X-M 的数据接口、三个评测划分与密封、LVIS 子集、ReasonSeg、去重清单（src/xm/data.py、src/xm/exclusions.py）。"""

from __future__ import annotations

import json
import shutil

import cv2
import numpy as np
import pytest

from src.data.subsets import population_digest
from src.executors.raster import rle_to_mask
from src.xm import data as D
from src.xm import exclusions as X
from tests._xm_fakes import REASONSEG, build_fake_root


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """假数据根 + 两份清单（与 scripts/xm/build_configs.py --tiny 同一条路径）。"""
    d = tmp_path_factory.mktemp("xm")
    info = build_fake_root(d / "root")
    doc = D.build_splits_doc(d / "root", tiny=True)
    sp = d / "xm_splits.json"
    sp.write_text(D.dump_config(doc), encoding="utf-8")
    ex = X.build_exclusions(d / "root", splits_path=sp, tiny=True, workers=2)
    ep = d / "xm_train_exclusions.json"
    ep.write_text(D.dump_config(ex), encoding="utf-8")
    return {"dir": d, "root": d / "root", "info": info, "doc": doc, "splits": sp, "ex": ex, "excl": ep}


def _val_doc(built):
    return D.read_json(built["root"] / D.COCO_ANN["val2017"])


# ---------------------------------------------------------------------------
# 建对
# ---------------------------------------------------------------------------


def test_coco_pairs_crowd_and_crowd_only(built):
    doc = _val_doc(built)
    pairs, st = D.coco_pairs(doc, coco_split="val2017", split="xm_tune", image_ids=[2001, 2002, 2003, 2004])
    ids = [p.id for p in pairs]
    assert ids == ["coco2017/val2017/2003/1", "coco2017/val2017/2004/18"]  # 2001 无标注、2002 只有 crowd、2003 的 car 只有 crowd
    assert st["n_crowd_only_skipped"] == 2 and st["n_images_without_pairs"] == 2
    person = pairs[0]
    assert len(person.instances) == 2 and len(person.crowd_rles) == 1 and st["n_crowd_regions"] == 1
    assert [i.ann_id for i in person.instances] == sorted(i.ann_id for i in person.instances)
    assert person.category == "person" and person.image == "coco2017/images/val2017/000000002003.jpg" and (person.width, person.height) == (64, 48)
    crowd = rle_to_mask(person.crowd_rles[0])
    assert crowd[10:44, 22:38].all() and crowd.sum() == 34 * 16
    for p in pairs:
        for inst in p.instances:
            m = rle_to_mask(inst.rle)
            assert inst.area == int(m.sum()) > 0 and m.shape == (48, 64)
            assert isinstance(inst.area, int) and len(inst.box_xyxy) == 4
    assert st["n_degenerate_polygons_dropped"] == 1  # 2004 的第二个多边形只有 4 个数


def test_coco_bbox_and_ids(built):
    doc = _val_doc(built)
    ann = next(a for a in doc["annotations"] if a["image_id"] == 2003 and not a["iscrowd"])
    pairs, _ = D.coco_pairs(doc, coco_split="val2017", split="train", image_ids=[2003])
    inst = next(i for i in pairs[0].instances if i.ann_id == ann["id"])
    x, y, w, h = ann["bbox"]
    assert inst.box_xyxy == (x, y, x + w, y + h)
    assert pairs[0].meta == {"image_id": 2003}
    with pytest.raises(D.XMDataError):
        D.coco_pairs(doc, coco_split="val2017", split="xm_tune", image_ids=[987654])


def test_segm_to_rle_variants():
    poly = [[2, 2, 10, 2, 10, 8, 2, 8]]
    r, dropped = D.segm_to_rle(poly, 12, 16)
    assert dropped == 0 and rle_to_mask(r).sum() == D.rle_area(r) > 0 and isinstance(r["counts"], str)
    r2, dropped2 = D.segm_to_rle([[1, 1, 3, 3]], 12, 16)  # 只有 4 个数：丢掉，空 mask
    assert dropped2 == 1 and D.rle_area(r2) == 0 and r2["size"] == [12, 16]
    m = np.zeros((12, 16), np.uint8)
    m[3:7, 4:9] = 1
    from pycocotools import mask as M

    comp = M.encode(np.asfortranarray(m))
    r3, _ = D.segm_to_rle({"size": [12, 16], "counts": comp["counts"].decode()}, 12, 16)
    assert (rle_to_mask(r3) == m.astype(bool)).all()
    flat = m.flatten(order="F")
    runs, cur, n = [], 0, 0
    for v in flat:
        if v != cur:
            runs.append(n)
            cur, n = v, 1
        else:
            n += 1
    runs.append(n)
    r4, _ = D.segm_to_rle({"size": [12, 16], "counts": runs}, 12, 16)
    assert (rle_to_mask(r4) == m.astype(bool)).all()


def test_pair_dict_roundtrip(built):
    pairs, _ = D.coco_pairs(_val_doc(built), coco_split="val2017", split="xm_tune", image_ids=[2003, 2005])
    for p in pairs:
        d = json.loads(json.dumps(D.pair_to_dict(p)))
        assert D.pair_from_dict(d) == p
    idx = D.pair_by_id(pairs)
    assert set(idx) == {p.id for p in pairs}
    with pytest.raises(D.XMDataError):
        D.pair_by_id(pairs + pairs[:1])


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def test_splits_deterministic_disjoint_and_stable(built):
    doc = built["doc"]
    again = D.build_splits_doc(built["root"], tiny=True)
    assert D.dump_config(again) == D.dump_config(doc)  # 确定
    assert D.check_splits_doc(doc) == []
    ids = {s: set(doc["splits"][s]["image_ids"]) for s in (*D.COCO_EVAL_SPLITS, D.LVIS_SUB)}
    names = list(ids)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not ids[names[a]] & ids[names[b]], (names[a], names[b])
    assert {s: len(ids[s]) for s in D.COCO_EVAL_SPLITS} == dict(D.COCO_EVAL_N_TINY)
    # D20 的探针图与 E1 的 COCO 子集都不在三个划分里（先剔除再抽样）
    allc = set().union(*(ids[s] for s in D.COCO_EVAL_SPLITS))
    assert not allc & {139, 39769}
    e1 = set(D.e1_coco_subset(_val_doc(built), tiny=True))
    assert len(e1) == D.E1_COCO_TINY_K and not allc & e1
    # 依次取：三份恰是同一个排序的前 2、接着 3、再接着 4 张
    pop, _, _ = D.coco_population(_val_doc(built), tiny=True)
    ranked = D.rank_images(pop, D.SPLITS_PURPOSE, D.SPLITS_SALT)
    assert sorted(ranked[:2]) == doc["splits"]["xm_tune"]["image_ids"]
    assert sorted(ranked[2:5]) == doc["splits"]["xm_decision"]["image_ids"]
    assert sorted(ranked[5:9]) == doc["splits"]["xm_final"]["image_ids"]
    # 指纹钉住（规则或假数据变了就会变；变了要先想清楚为什么）
    assert {s: doc["splits"][s]["image_ids"] for s in (*D.COCO_EVAL_SPLITS, D.LVIS_SUB)} == {
        "xm_tune": [2013, 2014], "xm_decision": [2008, 2016, 2018], "xm_final": [2007, 2009, 2010, 2022], "lvis_sub": [2004, 2015]}
    for s in D.COCO_EVAL_SPLITS:
        e = doc["splits"][s]
        assert e["sha256"] == population_digest(str(i) for i in e["image_ids"])


def test_hash_rank_salt_and_purpose():
    ids = list(range(100, 140))
    a = D.rank_images(ids, D.SPLITS_PURPOSE, D.SPLITS_SALT)
    import hashlib

    want = sorted(ids, key=lambda i: (hashlib.sha256(f"xm_splits_v1|coco2017_val2017|{i}".encode()).hexdigest(), str(i)))
    assert a == want
    assert D.rank_images(ids, D.LVIS_PURPOSE, D.LVIS_SALT) != a


def test_check_splits_doc_catches_tampering(built):
    doc = json.loads(json.dumps(built["doc"]))
    doc["splits"]["xm_final"]["image_ids"][0] = doc["splits"]["xm_decision"]["image_ids"][0]
    bad = D.check_splits_doc(doc)
    assert any("sha256" in b for b in bad) and any("shared_images:xm_decision|xm_final" in b for b in bad)
    doc2 = json.loads(json.dumps(built["doc"]))
    doc2["splits"]["xm_decision"]["sealed"] = False
    assert any("密封" in b for b in D.check_splits_doc(doc2))
    doc3 = json.loads(json.dumps(built["doc"]))
    doc3["tiny"] = False
    assert any("应是 300 张" in b for b in D.check_splits_doc(doc3))


def test_load_pairs_matches_doc_and_detects_changed_annotations(built, tmp_path):
    pairs = D.load_pairs("xm_tune", built["root"], splits_path=built["splits"])
    e = built["doc"]["splits"]["xm_tune"]
    assert len(pairs) == e["n_pairs"] and population_digest(p.id for p in pairs) == e["pairs_sha256"]
    assert all(p.split == "xm_tune" and p.source == "coco2017" for p in pairs)
    root2 = tmp_path / "root"
    shutil.copytree(built["root"], root2)
    p = root2 / D.COCO_ANN["val2017"]
    doc = json.loads(p.read_text())
    doc["annotations"] = doc["annotations"][:-1]
    p.write_text(json.dumps(doc))
    with pytest.raises(D.XMDataError, match="sha256"):
        D.load_pairs("xm_tune", root2, splits_path=built["splits"])
    with pytest.raises(D.XMDataError, match="不在"):
        D.load_pairs("xm_tune", built["root"], splits_path=tmp_path / "nope.json")
    with pytest.raises(D.XMDataError):
        D.load_pairs("test", built["root"], splits_path=built["splits"])


# ---------------------------------------------------------------------------
# 密封与 gate
# ---------------------------------------------------------------------------


def test_decision_sealed(built):
    with pytest.raises(D.SealedSplitError, match="G-XM"):
        D.load_pairs("xm_decision", built["root"], splits_path=built["splits"])
    with pytest.raises(D.SealedSplitError):
        D.load_pairs("xm_decision", built["root"], gate="G-EX", splits_path=built["splits"])
    pairs = D.load_pairs("xm_decision", built["root"], gate="G-XM", splits_path=built["splits"])
    assert len(pairs) == built["doc"]["splits"]["xm_decision"]["n_pairs"]
    rec = D.require_unsealed("xm_decision", "G-XM", now="2026-11-03T00:00:00+00:00")
    assert D.seal_identity(rec) == {"gate": "G-XM", "decision": D.SEAL_DECISION}
    assert D.require_unsealed("xm_tune", None) is None and D.require_unsealed("xm_final", None) is None


def _gate_file(tmp_path, ckpt, **over):
    ckpt.mkdir(exist_ok=True)
    (ckpt / "manifest.json").write_text('{"files": []}\n')
    doc = {"verdict": "通过", "rehearsal": False, "checks": {"not_tiny": True, "n_is_500": True},
           "candidate": {"manifest_sha256": D.file_sha256(ckpt / "manifest.json")}}
    doc.update(over)
    p = tmp_path / f"gate_{len(list(tmp_path.glob('gate_*.json')))}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False))
    return p


@pytest.mark.parametrize("split", ["xm_final", "lvis_sub"])
def test_final_gate(built, tmp_path, split):
    ck = tmp_path / "ckpt"
    with pytest.raises(D.XMGateError):
        D.load_pairs(split, built["root"], splits_path=built["splits"])
    with pytest.raises(D.XMGateError):
        D.load_pairs(split, built["root"], gate="G-XM", splits_path=built["splits"])  # 门名不够，要 gate 文件
    for over in ({"rehearsal": True}, {"verdict": "不通过"}, {"checks": {"not_tiny": False}}, {"checks": {}},
                 {"candidate": {"manifest_sha256": "0" * 64}}):
        with pytest.raises(D.XMGateError):
            D.check_xm_gate(_gate_file(tmp_path, ck, **over), checkpoint=ck)
    bad = tmp_path / "gate_missing.json"
    bad.write_text(json.dumps({"verdict": "通过"}))
    with pytest.raises(D.XMGateError, match="键"):
        D.check_xm_gate(bad, checkpoint=ck)
    ok = D.check_xm_gate(_gate_file(tmp_path, ck), checkpoint=ck)
    assert isinstance(ok, D.XMGatePass) and ok.identity()["gate"] == "G-XM"
    pairs = D.load_pairs(split, built["root"], xm_gate=ok, splits_path=built["splits"])
    assert len(pairs) == built["doc"]["splits"][split]["n_pairs"]
    assert D.require_xm_gate("xm_tune", None) is None and D.require_xm_gate("reasonseg_val", None) is None


# ---------------------------------------------------------------------------
# LVIS、ReasonSeg
# ---------------------------------------------------------------------------


def test_lvis_subset_val2017_only_and_disjoint(built, tmp_path):
    doc = built["doc"]
    lids = set(doc["splits"]["lvis_sub"]["image_ids"])
    val = set(built["info"]["val_ids"])
    assert lids <= val and not lids & {139, 39769} and not lids & set(built["info"]["lvis_train_ids"])
    assert not lids & set().union(*(set(doc["splits"][s]["image_ids"]) for s in D.COCO_EVAL_SPLITS))
    assert doc["splits"]["lvis_sub"]["population_steps"]["exclude_coco_splits"]["n_removed"] > 0
    lv = D.read_json(built["root"] / D.LVIS_ANN)
    pairs, st = D.lvis_pairs(lv, image_ids=sorted(lids))
    assert {p.source for p in pairs} == {"lvis_v1"} and all(p.image.startswith("coco2017/images/val2017/") for p in pairs)
    aero = [p for p in pairs if p.category_id == 1]
    assert aero and aero[0].category == "aerosol can" and aero[0].meta["lvis_name"] == "aerosol_can"
    for p in pairs:
        im = next(i for i in lv["images"] if i["id"] == p.meta["image_id"])
        assert p.meta["not_exhaustive"] == (p.category_id in im["not_exhaustive_category_ids"])
        assert p.meta["neg_category_ids"] == sorted(im["neg_category_ids"]) and p.crowd_rles == ()
        assert p.id == f"lvis_v1/val/{p.meta['image_id']}/{p.category_id}"
    with pytest.raises(D.XMDataError, match="val2017"):
        D.lvis_pairs(lv, image_ids=[3001])  # coco_url 是 train2017 的图不能进子集
    assert D.lvis_category_word("bow_(weapon)") == "bow (weapon)"


def _lisa_mask(doc, h, w):
    """LISA 的 get_mask_from_json 原样（只换掉读文件）。"""
    inform = doc["shapes"]
    area_list, valid = [], []
    for i in inform:
        if "flag" == i["label"].lower():
            continue
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(tmp, np.array([i["points"]], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp, np.array([i["points"]], dtype=np.int32), 1)
        area_list.append(tmp.sum())
        valid.append(i)
    sort_index = list(np.argsort(area_list)[::-1].astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    for s in sort_index:
        i = valid[s]
        v = 255 if "ignore" in i["label"].lower() else 1
        cv2.polylines(mask, np.array([i["points"]], dtype=np.int32), True, v, 1)
        cv2.fillPoly(mask, np.array([i["points"]], dtype=np.int32), v)
    return mask


def test_reasonseg_pairs(built):
    pairs = D.load_pairs("reasonseg_val", built["root"], splits_path=built["splits"])
    by = {p.meta["file"][:-4]: p for p in pairs}
    assert set(by) == set(REASONSEG)
    for stem, doc in REASONSEG.items():
        p = by[stem]
        assert p.id == f"reasonseg/val/{stem}/0" and p.category == doc["text"][0] and p.category_id == 0
        assert len(p.instances) == 1 and p.meta["is_sentence"] is doc["is_sentence"] and p.meta["n_texts"] == len(doc["text"])
        lisa = _lisa_mask(doc, p.height, p.width)
        assert (rle_to_mask(p.instances[0].rle) == (lisa == 1)).all()
        ign = p.meta["ignore_rle"]
        assert (ign is None) == (not (lisa == 255).any())
        if ign is not None:
            assert (rle_to_mask(ign) == (lisa == 255)).all()
        m = rle_to_mask(p.instances[0].rle)
        ys, xs = np.nonzero(m)
        assert p.instances[0].box_xyxy == (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    assert by["rs_b"].meta["n_flag"] == 1 and by["rs_b"].meta["n_ignore_polygons"] == 1
    assert by["rs_c"].meta["n_target_polygons"] == 2
    st = built["doc"]["splits"]["reasonseg_val"]["stats"]
    assert (st["n_sentence"], st["n_phrase"], st["n_multi_text"]) == (2, 2, 1)


def test_reasonseg_cp1252_fallback(tmp_path):
    p = tmp_path / "x.json"
    p.write_bytes(json.dumps({"text": ["café"], "is_sentence": False, "shapes": []}, ensure_ascii=False).encode("cp1252"))
    assert D.read_reasonseg_json(p)["text"] == ["café"]


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------


def test_dedupe_excludes_refcoco_eval_and_dups_from_train(built):
    ex = built["ex"]
    info = built["info"]
    want = sorted(info["refcoco_in_train"] + [info["reasonseg_dup_train"], info["reasonseg_name_train"], info["split_dup_train"]])
    assert ex["exclude_train2017_image_ids"] == want and X.check_exclusions_doc(ex) == []
    rc = ex["categories"]["refcoco_eval"]
    assert rc["n_in_train2017"] == 2 and set(rc["image_ids"]) >= {903, 920}
    cov = rc["subsets_covered"]
    for name in ("devtune800", "devcheck2", "test_refcoco_unc_testA", "test_refcocog_umd_test", "devcheck3_refcoco_unc_val",
                 "explore_val_refcocog_umd_val"):
        assert cov[name]["n_outside_eval_union"] == 0
    assert cov["devcheck2"]["image_of_checked"] and cov["devcheck3_refcoco_unc_val"]["image_of_checked"]
    assert all(cov[k]["explore_record_checked"] for k in cov if k.startswith("test_") or k in ("devtune800", "devcheck2"))
    rs = ex["categories"]["reasonseg_val"]["train2017_overlap"]
    assert rs["by_sha256"]["train_image_ids"] == [3007] and rs["by_file_name"]["train_image_ids"] == [3006]
    sp = ex["categories"]["coco_eval_splits"]["splits"]
    assert sp["xm_decision"]["train_image_ids_by_sha256"] == [3008]
    assert all(v["n_in_train2017_by_id"] == 0 for v in sp.values())
    assert {k: v["n_train2017_images"] for k, v in ex["by_category"].items()} == {"refcoco_eval": 2, "reasonseg_val": 2, "coco_eval_splits": 1}
    pairs = D.load_pairs("train", built["root"], splits_path=built["splits"], exclusions_path=built["excl"])
    imgs = {p.meta["image_id"] for p in pairs}
    assert imgs == {3001, 3002, 3003, 3004, 3005} and not imgs & set(want)
    assert all(p.split == "train" and p.id.startswith("coco2017/train2017/") for p in pairs)


def test_exclusions_bound_to_splits_file(built, tmp_path):
    sp2 = tmp_path / "xm_splits.json"
    sp2.write_text(built["splits"].read_text() + "\n")
    with pytest.raises(D.XMDataError):
        X.load_exclusions(built["excl"], splits_path=sp2)
    doc = json.loads(built["excl"].read_text())
    doc["exclude_train2017_image_ids"] = doc["exclude_train2017_image_ids"][1:]
    p = tmp_path / "ex.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(D.XMDataError, match="不合格"):
        X.load_exclusions(p, splits_path=built["splits"])


def test_subsets_covered_rejects_outside_and_image_of_mismatch():
    sents = {"refcoco": {"1": 901, "2": 902}}
    w2 = {"splits": {"devcheck2": {"kind": "refcoco", "dataset": "refcoco", "ids": ["1", "2"], "image_of": {"1": "coco:901", "2": "coco:902"}}}}
    ex = {"splits": {}}
    assert X.subsets_covered(w2, ex, sents, {901, 902})["devcheck2"]["n_images"] == 2
    with pytest.raises(D.XMDataError, match="并集"):
        X.subsets_covered(w2, ex, sents, {901})
    w2["splits"]["devcheck2"]["image_of"]["2"] = "coco:999"
    with pytest.raises(D.XMDataError, match="image_of"):
        X.subsets_covered(w2, ex, sents, {901, 902})


# ---------------------------------------------------------------------------
# 命令行：--out 的运行编号
# ---------------------------------------------------------------------------


def _script(name):
    import importlib.util

    p = D.REPO_ROOT / "scripts" / "xm" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"xm_{name}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("name,extra", [("build_configs", ["--data-root", "r", "--refcoco-root", "rc"]),
                                        ("build_targets", ["--data-root", "r"]),
                                        ("data_stats", ["--data-root", "r", "--targets", "t"])])
def test_out_needs_run_number(tmp_path, name, extra):
    m = _script(name)
    with pytest.raises(SystemExit) as e:
        m.parse_args([*extra, "--out", str(tmp_path / "xm_configs")])
    assert e.value.code == 2
    assert not (tmp_path / "xm_configs").exists()
    a = m.parse_args([*extra, "--out", str(tmp_path / "0103_xm_configs")])
    assert a.out.endswith("0103_xm_configs")


def test_build_configs_tiny_end_to_end(tmp_path, built):
    m = _script("build_configs")
    out = tmp_path / "cfg"
    assert m.main(["--tiny", "--out", str(out), "--workers", "1"]) == 0
    assert (out / "xm_splits.json").read_text() == built["splits"].read_text()
    ex = json.loads((out / "xm_train_exclusions.json").read_text())
    assert ex["exclude_train2017_image_ids"] == built["ex"]["exclude_train2017_image_ids"]
    assert m.main(["--tiny", "--out", str(out)]) == 2  # 已存在，不覆盖
