"""DEV-0017 §6：检测开发图的完整 COCO 标注（src/data/land/coco_train2014_dev.py，scripts/data/land_e1.py coco_train2014_e1dev）
与 E1 读取（src/e1/tasks.load_ovd_dev_annotations、select("ovd", "dev")）。上游 zip 由本地假服务器提供，不联网。"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from src.data import loaders as L
from src.data.land import coco_train2014_dev as CD
from src.data.land.common import DryRun, LandContext
from src.data.land.manifest import read_manifest
from src.e1.tasks import FIXTURE_DIR, OVD_DEV_SOURCE, DataPaths, load_ovd_dev_annotations, select
from tests.test_data_net import FakeServer

REPO = Path(__file__).resolve().parents[1]
DEV = FIXTURE_DIR / "dev_subsets.json"
# fixture 的开发句：refcoco 101（ref 0 → ann 7000、图 901）、refcoco+ 300（ann 8300、图 901）、refcocog 403（ann 8401、图 902）
REFERRED = {7000: (901, 1, [1.0, 1.0, 8.0, 6.0]), 8300: (901, 1, [2.0, 1.0, 5.0, 4.0]), 8401: (902, 1, [2.0, 1.0, 5.0, 4.0])}


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def instances(**override) -> dict:
    anns = [{"id": aid, "image_id": iid, "category_id": cid, "bbox": bb, "area": bb[2] * bb[3], "iscrowd": 0}
            for aid, (iid, cid, bb) in REFERRED.items()]
    anns += [{"id": 9100, "image_id": 902, "category_id": 3, "bbox": [0.0, 0.0, 3.0, 3.0], "area": 9.0, "iscrowd": 1},
             {"id": 9200, "image_id": 903, "category_id": 2, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 4.0, "iscrowd": 0}]
    cats = [{"id": k, "name": "person" if k == 1 else f"c{k}", "supercategory": "s"} for k in range(1, CD.N_COCO_CATEGORIES + 1)]
    doc = {"info": {"description": "fake"}, "licenses": [],
           "images": [{"id": i, "file_name": f"COCO_train2014_{i:012d}.jpg", "width": 12, "height": 9} for i in (901, 902, 903, 904)],
           "annotations": anns, "categories": cats}
    doc.update(override)
    return doc


def publish(server: FakeServer, inst: dict) -> dict:
    raw = json.dumps(inst).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CD.SPEC["ann_member"], raw)
        zf.writestr("annotations/captions_train2014.json", b"{}" * 3000)
    data = buf.getvalue()
    server.files["/annotations/annotations_trainval2014.zip"] = data
    info = zipfile.ZipFile(io.BytesIO(data)).getinfo(CD.SPEC["ann_member"])
    return {"ann_zip_url": server.url("/annotations/annotations_trainval2014.zip"), "zip_bytes": len(data),
            "member_bytes": info.file_size, "member_compress_bytes": info.compress_size, "member_crc32": f"{info.CRC:08x}"}


def ctx_for(tmp_path: Path, *, with_refcoco: bool = True, **kw) -> LandContext:
    root = tmp_path / "root"
    if with_refcoco:
        shutil.copytree(FIXTURE_DIR / "root" / "refcoco", root / "refcoco")
    return LandContext(root=root, tmpdir=tmp_path / "tmp", retries=2, backoff=0.0, log=lambda s: None, **kw)


def test_land_writes_dev_image_annotations_and_manifest(server, tmp_path):
    spec = publish(server, instances())
    ctx = ctx_for(tmp_path)
    CD.land(ctx, spec, dev_config=DEV)
    src = ctx.root / CD.SOURCE
    man = read_manifest(src)
    assert all(v["ok"] for v in man["verification"]) and not man["failures"], man["verification"]
    sub = json.loads((src / CD.SUBSET_FILE).read_text())
    assert [im["id"] for im in sub["images"]] == [901, 902]
    assert [a["id"] for a in sub["annotations"]] == [7000, 8300, 8401, 9100]  # 全部标注（含未被指代的、iscrowd 的），不含别的图
    assert len(sub["categories"]) == CD.N_COCO_CATEGORIES
    assert man["selection"]["image_ids"] == [901, 902] and man["selection"]["n_union"] == 2
    assert man["selection"]["per_dataset"]["refcocog"] == {"dev_subset": "refcocog_umd_val", "n_sentences": 1, "n_images": 1}
    assert man["stats"] == {"n_images": 2, "n_images_without_annotations": 0, "n_annotations": 4, "n_crowd": 1,
                            "n_categories_present": 2, "n_referred_checked": 3}
    up = man["upstream"][0]
    assert up["member_crc32"] == spec["member_crc32"] and up["range_bytes_fetched"] < spec["zip_bytes"] + 1
    assert any("coco_val2017 还没有落地" in n for n in man["notes"])
    assert not list(ctx.tmpdir.rglob("instances_train2014.json"))  # 临时文件已删
    got = load_ovd_dev_annotations(ctx.root)
    assert got is not None and got[1]["sha256"] == man["files"][CD.SUBSET_FILE]["sha256"]


def test_land_flags_referred_annotation_mismatch_and_categories(server, tmp_path):
    inst = instances()
    inst["annotations"][1]["bbox"] = [2.0, 1.0, 5.0, 4.5]  # ann 8300 与 refcoco+ 里的框不同
    spec = publish(server, inst)
    ctx = ctx_for(tmp_path)
    shutil.copytree(FIXTURE_DIR / "root" / "coco_val2017", ctx.root / "coco_val2017")  # 只有 2 类：与 80 类不同
    CD.land(ctx, spec, dev_config=DEV)
    v = {x["what"]: x for x in read_manifest(ctx.root / CD.SOURCE)["verification"]}
    bad = v["被指代的标注（refcoco 文件）在 instances_train2014 里，image_id、category_id、bbox 逐位相同"]
    assert bad["ok"] is False and bad["problems_head"][0]["ann_id"] == 8300
    assert v["类别表（id 与名字）= coco_val2017 的类别表（E1 检测提示用的那张）"]["ok"] is False
    assert v["每张开发图都在 instances_train2014 里"]["ok"] is True


def test_land_rejects_central_directory_mismatch_and_missing_refcoco(server, tmp_path):
    spec = publish(server, instances())
    with pytest.raises(ValueError, match="与 SPEC 不符"):
        CD.land(ctx_for(tmp_path / "a"), {**spec, "member_crc32": "00000000"}, dev_config=DEV)
    assert not (tmp_path / "a" / "root" / CD.SOURCE / CD.SUBSET_FILE).exists()
    with pytest.raises(FileNotFoundError, match="先落地 refcoco"):
        CD.land(ctx_for(tmp_path / "b", with_refcoco=False), spec, dev_config=DEV)


def test_land_dry_run_before_any_request_and_limit(server, tmp_path):
    spec = publish(server, instances())
    with pytest.raises(DryRun):
        CD.land(ctx_for(tmp_path / "d", dry_run=True), spec, dev_config=DEV)
    assert server.log == []  # 预算检查在任何网络请求之前
    ctx = ctx_for(tmp_path / "l", limit=1)
    CD.land(ctx, spec, dev_config=DEV)
    sub = json.loads((ctx.root / CD.SOURCE / CD.SUBSET_FILE).read_text())
    assert [im["id"] for im in sub["images"]] == [901]


def test_dev_images_follow_loader_conventions():
    assert CD.REFCOCO_DATASETS == tuple(L.REFCOCO_FILES)
    dev = CD.dev_images(FIXTURE_DIR / "root", DEV)
    assert dev["image_ids"] == [901, 902] and set(dev["referred"]) == set(REFERRED)
    assert all((dev["referred"][a]["image_id"], dev["referred"][a]["category_id"], dev["referred"][a]["bbox"]) == REFERRED[a]
               for a in REFERRED)


def test_spec_pins_central_directory_of_official_zip():
    """开发机 2026-10-01 用 Range 读官方 zip 的中央目录得到的值（只取约 35 KB）。"""
    s = CD.SPEC
    assert s["ann_zip_url"] == "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    assert (s["zip_bytes"], s["member_bytes"], s["member_compress_bytes"], s["member_crc32"]) == (
        252_872_794, 332_556_225, 107_297_879, "3cd50ca9")


def test_land_cli_lists_source_outside_all():
    spec = importlib.util.spec_from_file_location("_land_e1_cd", REPO / "scripts/data/land_e1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.OPTIONAL_LANDERS["coco_train2014_e1dev"] is CD.land and "coco_train2014_e1dev" not in mod.LANDERS
    assert mod.parse_args(["coco_train2014_e1dev", "--root", "/tmp/x"]).source == "coco_train2014_e1dev"


def test_fixture_manifest_matches_fixture_file_and_select_uses_it(tmp_path):
    import hashlib

    src = FIXTURE_DIR / "root" / CD.SOURCE
    data = (src / CD.SUBSET_FILE).read_bytes()
    assert read_manifest(src)["files"][CD.SUBSET_FILE]["sha256"] == hashlib.sha256(data).hexdigest()
    sel = select("ovd", "dev", DataPaths.fixtures(nyu_cache=tmp_path))
    assert sel.has_gt and {s.id for s in sel.samples} == {f"{OVD_DEV_SOURCE}/901", f"{OVD_DEV_SOURCE}/902"}
