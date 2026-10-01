"""统一的样本加载器（src/data/loaders.py，DEV-0012 任务书 §4）。

tests/fixtures/e1/ 下是每个源几条的小 fixture（图像为合成小图，格式与落地输出相同）；dev_subsets.json 是 fixture 版的开发子集
（每个子集手工放 1 条）。另有一组测试直接在落地脚本对假上游的输出上跑加载器，防止两边格式分叉。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import loaders as L
from src.data.dedupe import REASON_DEV, REASON_PROBE, Exclusion
from src.data.records import Sample
from src.data.subsets import sort_ids

FX = Path(__file__).resolve().parent / "fixtures" / "e1"
ROOT = FX / "root"
DEV = FX / "dev_subsets.json"


def check_common(res, task: str) -> None:
    ids = [s.id for s in res.records]
    assert ids == sort_ids(ids) and len(set(ids)) == len(ids)  # id 的自然顺序
    for s in res.records:
        assert isinstance(s, Sample) and s.task == task
        assert all(Path(p).exists() for p in s.images)
        assert len(s.image_keys) == len(s.images) == len(s.phash)
        assert all(h and len(h) == 16 for h in s.phash)
    st = res.stats
    # 每条样本都有去向：保留、被剔除（逐条记录）、或缺图
    assert st["n_kept"] == len(res.records)
    assert st["exclusion"]["n_in"] == st["n_kept"] + len(st["exclusion"]["excluded"])
    assert sum(st["exclusion"]["excluded_by_reason"].values()) == len(st["exclusion"]["excluded"])


def test_refcoco_val_eval_excludes_dev_and_converts_boxes():
    res = L.load_refcoco(ROOT, "refcoco", "val", dev_config=DEV)
    check_common(res, "refexp")
    assert res.stats["n_total"] == 6 and res.stats["n_dev_excluded"] == 1 and res.stats["n_kept"] == 5
    assert "refcoco_unc/val/101" not in [s.id for s in res.records]
    s = next(x for x in res.records if x.fields["sent_id"] == 100)
    assert s.gt["box"] == [1.0, 1.0, 9.0, 7.0]  # xywh [1,1,8,6] → xyxy
    assert s.gt["mask"] == {"polygons": [[1.0, 1.0, 9.0, 1.0, 9.0, 7.0, 1.0, 7.0]], "height": 9, "width": 12}
    assert s.fields["expression"] == "the thing 100" and s.fields["category"] == "person" and s.image_keys == ("coco:901",)
    two = next(x for x in res.records if x.fields["ref_id"] == 2)
    assert len(two.gt["mask"]["polygons"]) == 2
    excl = res.stats["exclusion"]["excluded"]
    assert excl == [{"id": "refcoco_unc/val/101", "reason": REASON_DEV, "by": "sample_id"}]


def test_refcoco_dev_and_test_splits():
    dev = L.load_refcoco(ROOT, "refcoco", "val", subset="dev", dev_config=DEV)
    assert [s.id for s in dev.records] == ["refcoco_unc/val/101"] and dev.stats["n_not_in_dev"] == 5
    a = L.load_refcoco(ROOT, "refcoco", "testA", dev_config=DEV)
    assert len(a.records) == 2 and a.stats["n_dev_excluded"] == 0
    with pytest.raises(ValueError, match="开发子集只在 val"):
        L.load_refcoco(ROOT, "refcoco", "testA", subset="dev", dev_config=DEV)
    with pytest.raises(ValueError, match="没有 test"):
        L.load_refcoco(ROOT, "refcoco", "test")
    allv = L.load_refcoco(ROOT, "refcoco", "val", subset="all", dev_config=DEV)
    assert len(allv.records) == 6


def test_coco_ovd_excludes_probe_image_and_keeps_crowd_flag():
    res = L.load_coco_ovd(ROOT)
    check_common(res, "ovd")
    assert [s.fields["image_id"] for s in res.records] == [5, 17]
    assert res.stats["n_probe_excluded"] == 1
    assert res.stats["exclusion"]["excluded"][0] == {"id": "coco_val2017/val2017/39769", "reason": REASON_PROBE, "by": "image_key", "key": "coco:39769"}
    s5 = res.records[0]
    assert s5.fields["categories"] == ["person", "car"] and s5.fields["category_ids"] == [1, 3]
    assert s5.gt["boxes"] == [
        {"box": [1.0, 2.0, 4.0, 6.0], "category_id": 1, "category": "person", "iscrowd": 0, "ann_id": 1},
        {"box": [0.5, 0.5, 2.5, 2.5], "category_id": 3, "category": "car", "iscrowd": 1, "ann_id": 2},
    ]
    assert res.records[1].gt["boxes"] == []
    assert Path(res.stats["coco_json"]).exists()


def test_odinw_glip_and_raw_category_names():
    g = L.load_odinw(ROOT, "Fake_sub")
    check_common(g, "ovd")
    assert g.records[0].fields["categories"] == ["rabbit", "pistol"] and g.records[0].split == "val"
    assert g.records[0].gt["boxes"][0]["category"] == "pistol"
    r = L.load_odinw(ROOT, "Fake_sub", category_names="raw")
    assert r.records[0].fields["categories"] == ["Cottontail-Rabbit", "pistol-x"] and r.records[0].gt["boxes"][0]["category"] == "pistol-x"
    assert r.records[0].image_keys == ("odinw:Fake_sub:a.jpg",)


def test_countbench_skips_rows_without_image_and_dev():
    res = L.load_countbench(ROOT, dev_config=DEV)
    check_common(res, "count")
    assert [s.fields["row"] for s in res.records] == [0, 3]
    assert res.stats["n_rows_without_image_in_hf"] == 1 and res.stats["n_dev_excluded"] == 1
    assert [s.gt["count"] for s in res.records] == [2, 5]
    assert [s.fields["row"] for s in L.load_countbench(ROOT, subset="dev", dev_config=DEV).records] == [1]


def test_blink_and_cvbench_and_embspatial():
    b = L.load_blink_reldepth(ROOT, dev_config=DEV)
    check_common(b, "rel_depth")
    assert [s.fields["idx"] for s in b.records] == ["val_Relative_Depth_1", "val_Relative_Depth_2"] and [s.gt["answer"] for s in b.records] == ["A", "B"]
    assert b.stats["n_dev_excluded"] == 1

    c = L.load_cvbench(ROOT, dev_config=DEV)
    check_common(c, "spatial_qa")
    # idx 0 的源图是 D20 探针图（coco:139），idx 1 是开发子集：都剔除并逐条记录
    assert [s.fields["idx"] for s in c.records] == [2]
    assert c.stats["exclusion"]["excluded_by_reason"] == {REASON_PROBE: 1, REASON_DEV: 1}
    assert c.records[0].image_keys == ("nyuv2:21",) and c.records[0].gt["answer"] == "C"

    e = L.load_embspatial(ROOT, dev_config=DEV)
    check_common(e, "spatial_qa")
    assert [s.fields["question_id"] for s in e.records] == ["ai2thor_0", "scannet_0"]
    assert [(s.gt["answer"], s.gt["answer_index"]) for s in e.records] == [("B", 1), ("C", 2)]
    assert e.records[0].fields["choices"] == ["a", "b", "c", "d"]


def test_nyu_pairs_render_on_demand_and_verify(tmp_path):
    cache = tmp_path / "render"
    res = L.load_nyu_pairs(ROOT, pairs_config=FX / "nyu_pairs.json", cache_dir=cache)
    check_common(res, "rel_depth")
    cfg = json.loads((FX / "nyu_pairs.json").read_text())
    test_pairs = [p for p in cfg["pairs"] if p["split"] == "test"]
    assert [s.fields["points"] for s in res.records] == [{"A": p["A"], "B": p["B"]} for p in test_pairs]
    assert [s.gt["closer"] for s in res.records] == [p["closer"] for p in test_pairs]
    for s in res.records:
        assert (s.gt["closer"] == "A") == (s.gt["depth_A"] < s.gt["depth_B"])
        assert Path(s.images[0]).parent == cache and Path(s.fields["raw_image"]).exists()
        assert s.image_keys == ("nyuv2:1",)
    dev = L.load_nyu_pairs(ROOT, subset="dev", pairs_config=FX / "nyu_pairs.json", cache_dir=cache)
    assert {s.fields["idx"] for s in dev.records} == {3} and len(dev.records) == 5 and dev.stats["subset"] == "dev"
    # 已渲染的图被改坏：重新渲染
    first = Path(res.records[0].images[0])
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(first)
    L.load_nyu_pairs(ROOT, pairs_config=FX / "nyu_pairs.json", cache_dir=cache)
    assert np.asarray(Image.open(first)).any()


def test_nyu_pairs_detects_render_drift(tmp_path):
    cfg = json.loads((FX / "nyu_pairs.json").read_text())
    cfg["pairs"][0]["render_sha256"] = "0" * 64
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="render_sha256"):
        L.load_nyu_pairs(ROOT, pairs_config=p, cache_dir=tmp_path / "r")


def test_strict_missing_image(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(ROOT / "blink_reldepth", root / "blink_reldepth")
    (root / "blink_reldepth/images/val_Relative_Depth_1.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="落地不全"):
        L.load_blink_reldepth(root, dev_config=DEV)
    res = L.load_blink_reldepth(root, dev_config=DEV, strict=False)
    assert res.stats["n_missing_image"] == 1 and res.stats["n_total"] == 3 and len(res.records) == 1


def test_extra_exclusion_by_phash_and_key():
    man = json.loads((ROOT / "embspatial/MANIFEST.json").read_text())
    h = man["files"]["images/scannet_0.jpg"]["phash"]
    ex = Exclusion().add_phashes({"train:some_image": h}, "训练集图像").add_keys(["embspatial:ai2thor_0"], "训练集图像")
    res = L.load_embspatial(ROOT, dev_config=DEV, extra_exclusion=ex)
    assert res.records == []
    hits = {x["id"]: x for x in res.stats["exclusion"]["excluded"]}
    assert hits["embspatial/test/scannet_0"]["by"] == "phash" and hits["embspatial/test/scannet_0"]["hamming"] == 0
    assert hits["embspatial/test/ai2thor_0"]["by"] == "image_key"
    assert res.stats["exclusion"]["excluded_by_reason"] == {REASON_DEV: 1, "训练集图像": 2}


def test_bad_subset_and_missing_dev_config(tmp_path):
    with pytest.raises(ValueError):
        L.load_cvbench(ROOT, subset="test", dev_config=DEV)
    cfg = json.loads(DEV.read_text())
    del cfg["subsets"]["cvbench"]
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(KeyError, match="cvbench"):
        L.load_cvbench(ROOT, dev_config=p)  # 配置里没有就报错，不当成空集


def test_populations_from_fixture_root():
    pops, prov = L.populations(ROOT, ["cvbench", "embspatial", "blink_reldepth", "countbench", "refcoco_unc_val"])
    assert pops["cvbench"] == ["0", "1", "2"] and pops["countbench"] == ["0", "1", "3"]
    assert pops["refcoco_unc_val"] == ["100", "101", "102", "103", "104", "105"]
    assert set(prov) == set(pops)


# ---------------------------------------------------------------------------
# 落地输出 → 加载器（防止格式分叉）
# ---------------------------------------------------------------------------


def test_loaders_read_lander_outputs(tmp_path):
    from tests.test_data_land import (
        FakeServer,
        coco_upstream,
        countbench_upstream,
        cvbench_upstream,
        make_ctx,
        refcoco_upstream,
    )
    from src.data.land import coco, hfbench, refcoco

    s = FakeServer()
    try:
        ctx = make_ctx(tmp_path, options={"refcoco_datasets": ["refcoco"]})
        coco.land(ctx, coco_upstream(s))
        spec, _ = refcoco_upstream(s)
        refcoco.land(ctx, spec)
        cvbench_upstream(s, "c" * 40)
        hfbench.land_cvbench(ctx, {"rev": "c" * 40, "hf_base": s.base})
        hfbench.land_countbench(ctx, countbench_upstream(s, "k" * 40))
    finally:
        s.close()
    dev = tmp_path / "dev.json"
    cfg = json.loads(DEV.read_text())
    cfg["subsets"]["refcoco_unc_val"]["ids"] = ["2"]
    cfg["subsets"]["cvbench"]["ids"] = ["4"]
    cfg["subsets"]["countbench"]["ids"] = ["5"]
    dev.write_text(json.dumps(cfg))
    c = L.load_coco_ovd(ctx.root)
    assert len(c.records) == 8 and c.stats["n_probe_excluded"] == 0  # 落地时已先剔除探针图
    r = L.load_refcoco(ctx.root, "refcoco", "val", dev_config=dev)
    assert [s.fields["sent_id"] for s in r.records] == [3, 4, 5] and r.stats["n_dev_excluded"] == 1
    v = L.load_cvbench(ctx.root, dev_config=dev)
    assert [s.fields["idx"] for s in v.records] == [1, 2, 3]  # idx 0 是 coco:139（D20），idx 4 是开发子集
    n = L.load_countbench(ctx.root, dev_config=dev)
    assert [s.fields["row"] for s in n.records] == [0, 1, 3]
