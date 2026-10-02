"""训练集与评测集的去重（硬约束 7；DEV-0022 任务书 §2、§3）：构造的重合能被查出、阈值边界、跨机器导出再比较、
内置评测集（D20 探针图、COCO val2017 全部 id）、已知重合的标定、训练剔除清单被加载器读到并自动排除。不联网。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import train_dedupe as td
from src.data import train_loaders as tl
from src.data.constants import PHASH_MAX_HAMMING, PROBE_IMAGE_KEYS
from src.data.dedupe import phash_file
from src.data.land.manifest import Manifest

REPO = Path(__file__).resolve().parents[1]


def flip(h: str, n: int) -> str:
    """把 64 位哈希的低 n 位取反（汉明距离恰为 n）。"""
    return f"{int(h, 16) ^ ((1 << n) - 1):016x}"


def S(role_name: str, items):
    return {"source": role_name.split("/")[0], "role": td.role_of(role_name), "items": [list(it) for it in items]}


def make_source(root: Path, source: str, files: dict[str, dict], stats: dict | None = None) -> Path:
    d = root / source
    m = Manifest(d, source, root=root, tmpdir=root, min_free_gb=0, log=lambda s: None)
    for rel, rec in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(rel.encode())
        m.add_file(rel, sha256="0" * 64, nbytes=p.stat().st_size, **rec)
    if stats:
        m.data["stats"] = stats
    m.write()
    return d


H0 = "f030474e4a5b59f9"
H1 = "0123456789abcdef"


def test_threshold_boundary_and_same_key():
    sets = {
        "ade20k/train": S("ade20k/train", [("ade20k:A", H0, "a.jpg"), ("ade20k:B", H1, "b.jpg"), ("ade20k:C", "ffffffffffffffff", "c.jpg")]),
        "refcoco_train/train": S("refcoco_train/train", [("coco:5", None, "images/5.jpg"), ("coco:6", None, "images/6.jpg")]),
        "cvbench/test": S("cvbench/test", [("x:1", flip(H0, 8), "p1"), ("coco:5", None, "p5")]),
        "embspatial/test": S("embspatial/test", [("y:1", flip(H1, 9), "q1")]),
    }
    r = td.compare(sets, max_hamming=8)
    ex = {t: {e["path"]: e for e in v} for t, v in r["exclusions"].items()}
    assert set(ex["ade20k/train"]) == {"a.jpg"} and ex["ade20k/train"]["a.jpg"]["reasons"][0] == {"eval_set": "cvbench/test", "by": "phash", "matched": "x:1", "matched_path": "p1", "hamming": 8}
    assert set(ex["refcoco_train/train"]) == {"images/5.jpg"} and ex["refcoco_train/train"]["images/5.jpg"]["reasons"] == [{"eval_set": "cvbench/test", "by": "image_key", "matched": "coco:5"}]
    p = {(x["train"], x["eval"]): x for x in r["pairs"]}
    assert p[("ade20k/train", "cvbench/test")]["n_train_items_phash_le"] == {"0": 0, "2": 0, "4": 0, "6": 0, "8": 1, "10": 1, "12": 1}
    assert p[("ade20k/train", "embspatial/test")]["n_train_items_phash_le"]["8"] == 0 and p[("ade20k/train", "embspatial/test")]["n_train_items_phash_le"]["10"] == 1
    assert p[("refcoco_train/train", "cvbench/test")]["n_same_key"] == 1 and p[("refcoco_train/train", "cvbench/test")]["same_key_examples"] == ["coco:5"]
    r9 = td.compare(sets, max_hamming=9)
    assert {e["path"] for e in r9["exclusions"]["ade20k/train"]} == {"a.jpg", "b.jpg"}
    assert PHASH_MAX_HAMMING == 8  # 报告与剔除用的默认阈值（暂定，见 src/data/constants.py）


def test_hamming_matrix_matches_python():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2**63, 7, dtype=np.uint64)
    b = rng.integers(0, 2**63, 5, dtype=np.uint64)
    d = td.hamming_matrix(a, b)
    assert d.shape == (7, 5) and all(d[i, j] == bin(int(a[i]) ^ int(b[j])).count("1") for i in range(7) for j in range(5))


def test_builtin_sets():
    b = td.builtin_sets()
    assert [it[0] for it in b[td.PROBE_SET]["items"]] == list(PROBE_IMAGE_KEYS) and all(it[1] for it in b[td.PROBE_SET]["items"])
    coco = b[td.COCO_VAL2017_SET]["items"]
    assert len(coco) == 5000 and coco[0][0].startswith("coco:") and all(it[1] is None for it in coco) and b[td.COCO_VAL2017_SET]["role"] == "eval"
    assert "coco:139" in {it[0] for it in coco} and "coco:39769" in {it[0] for it in coco}  # D20 的两张探针图都是 val2017


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    tr, ev = tmp_path / "a800", tmp_path / "b200"
    make_source(
        tr,
        "refcoco_train",
        {"images/COCO_train2014_000000000009.jpg": {"kind": "image", "image_key": "coco:9", "phash": H1, "split": "train"}, "images/COCO_train2014_000000000777.jpg": {"kind": "image", "image_key": "coco:777", "phash": "aaaaaaaaaaaaaaaa", "split": "train"}, "d11_1.json": {"kind": "metadata"}},
        stats={"d11_1": {"n_train_union": 5, "n_eval_union": 4, "n_removed": 2, "n_kept": 3, "cross": {}}},
    )
    make_source(
        tr,
        "nyu_normals",
        {"train/img/0003.png": {"kind": "image", "image_key": "nyuv2:3", "phash": "1111222233334444", "split": "train"}, "test/img/0001.png": {"kind": "image", "image_key": "nyuv2:1", "phash": flip("5555666677778888", 2), "split": "test"}, "test/norm/0001.png": {"kind": "normal", "split": "test"}},
    )
    make_source(tr, "nyu_depth_train", {"sync/a_0001a/rgb_00001.jpg": {"kind": "image", "image_key": "nyuraw:a_0001a/00001", "phash": flip("5555666677778888", 5), "split": "train"}}, stats={"d11_2": {"n_pairs": 6, "n_removed": 3, "n_kept": 3, "removed_dirs": {"b": 3}}})
    make_source(ev, "coco_val2017", {"images/000000000777.jpg": {"kind": "image", "image_key": "coco:777", "phash": "bbbbbbbbbbbbbbbb"}})
    nd = make_source(ev, "nyuv2", {"rgb/0001.png": {"kind": "image", "image_key": "nyuv2:1", "phash": "5555666677778888"}, "rgb/0003.png": {"kind": "image", "image_key": "nyuv2:3", "phash": "1111222233334444"}, "depth/0001.npz": {"kind": "depth"}})
    (nd / "index.json").write_text(json.dumps({"images": [{"idx": 1, "split": "test", "rgb": "rgb/0001.png"}, {"idx": 3, "split": "train", "dev": True, "rgb": "rgb/0003.png"}]}))
    make_source(ev, "cvbench", {"images/img/2D/count/coco_1.jpg": {"kind": "image", "image_key": "coco:777", "phash": flip("bbbbbbbbbbbbbbbb", 3)}, "images/img/3D/depth/sun_1.jpg": {"kind": "image", "image_key": "nyuv2:1", "phash": flip("5555666677778888", 1)}})
    return tr, ev


def test_export_sets_and_roles(tmp_path):
    tr, ev = _roots(tmp_path)
    e = td.export_root(ev)
    assert e["kind"] == td.EXPORT_KIND and sorted(e["sets"]) == ["coco_val2017/e1sub500", "cvbench/test", "nyuv2/dev_train", "nyuv2/test"]
    assert all(s["role"] == "eval" for s in e["sets"].values())
    t = td.export_root(tr)
    assert {n: s["role"] for n, s in t["sets"].items()} == {"nyu_depth_train/train": "train", "nyu_normals/test": "eval", "nyu_normals/train": "train", "refcoco_train/train": "train"}
    assert t["sets"]["nyu_normals/test"]["items"] == [["nyuv2:1", flip("5555666677778888", 2), "test/img/0001.png"]]
    # 单个 MANIFEST.json 也能当输入（nyuv2 的 index.json 在旁边）
    one = td.load_input(ev / "nyuv2" / "MANIFEST.json")
    assert sorted(one["sets"]) == ["nyuv2/dev_train", "nyuv2/test"]


def test_cross_machine_report_and_loader_exclusion(tmp_path):
    tr, ev = _roots(tmp_path)
    mod = _cli()
    eval_json, train_json = tmp_path / "eval_hashes_b200.json", tmp_path / "train_hashes_a800.json"
    assert mod.main(["--export-hashes", str(ev), "--out", str(eval_json)]) == 0
    assert mod.main(["--export-hashes", str(tr), "--out", str(train_json)]) == 0
    assert json.loads(eval_json.read_text())["schema"] == td.EXPORT_SCHEMA
    out1, out2 = tmp_path / "rep_files", tr / "dedupe"
    assert mod.main(["--train-hashes", str(train_json), "--eval-manifest", str(eval_json), "--out", str(out1)]) == 0
    assert mod.main(["--train-root", str(tr), "--eval-manifest", str(ev), "--out", str(out2)]) == 0
    r1, r2 = (json.loads((o / "dedupe_report.json").read_text()) for o in (out1, out2))
    assert r1["pairs"] == r2["pairs"] and json.loads((out1 / "train_exclusions.json").read_text())["sets"] == json.loads((out2 / "train_exclusions.json").read_text())["sets"]

    ex = td.load_exclusions(out2 / "train_exclusions.json")
    # COCO train2014 对 COCO val2017：同键 coco:777 被查出（E1 的 500 子集、CV-Bench 的 COCO 图、内置的 val2017 全集）
    reasons = ex["refcoco_train/train"]["images/COCO_train2014_000000000777.jpg"]["reasons"]
    assert {(x["eval_set"], x["by"]) for x in reasons} == {("coco_val2017/e1sub500", "image_key"), ("cvbench/test", "image_key")}
    assert "images/COCO_train2014_000000000009.jpg" not in ex["refcoco_train/train"]
    # 法向 train 里的 E1 开发子集图（nyuv2:3）：同键 + 感知哈希距离 0
    assert {(x["eval_set"], x["by"]) for x in ex["nyu_normals/train"]["train/img/0003.png"]["reasons"]} == {("nyuv2/dev_train", "image_key"), ("nyuv2/dev_train", "phash")}
    # NYU raw 训练帧与 NYU test 图的感知哈希距离 5（≤ 8）也被查出；与法向 test（同为评测集，距离 3）同样
    assert {x["eval_set"] for x in ex["nyu_depth_train/train"]["sync/a_0001a/rgb_00001.jpg"]["reasons"]} == {"nyuv2/test", "cvbench/test", "nyu_normals/test"}
    # 标定：不同集合里同键的已知重合
    by = r2["calibration"]["known_duplicates"]["by_pair"]
    assert by["cvbench/test ~ nyuv2/test"]["max"] == 1 and by["nyu_normals/test ~ nyuv2/test"]["max"] == 2 and by["coco_val2017/e1sub500 ~ cvbench/test"]["max"] == 3
    assert r2["d11"]["d11_1"]["n_kept"] == 3 and r2["d11"]["d11_2"]["n_removed"] == 3
    md = (out2 / "dedupe_report.md").read_text()
    assert "D11-1（RefCOCO 训练图）" in md and "D11-2（NYU 深度训练集）" in md and "阈值标定" in md and "coco_val2017_full/val2017" in md

    # 训练加载器读 <root>/dedupe/train_exclusions.json 自动排除
    (tr / "nyu_depth_train" / "pairs_train.jsonl").write_text(json.dumps({"id": "a_0001a/00001", "scene": "a_0001a", "base_scene": "a_0001", "frame": "00001", "rgb": "sync/a_0001a/rgb_00001.jpg", "depth": "sync/a_0001a/sync_depth_00001.png", "focal": 518.8579}) + "\n")
    _patch_manifest_file(tr / "nyu_depth_train", "sync/a_0001a/sync_depth_00001.png", {"kind": "depth", "split": "train"})
    res = tl.load_nyu_depth_train(tr)
    assert len(res) == 0 and res.stats["n_dedupe_excluded"] == 1 and res.stats["dedupe_excluded"][0]["path"] == "sync/a_0001a/rgb_00001.jpg"
    assert res.stats["dedupe_file"] == str(tr / td.EXCLUSIONS_REL)


def _patch_manifest_file(src: Path, rel: str, rec: dict) -> None:
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    man = json.loads((src / "MANIFEST.json").read_text())
    man["files"][rel] = {"path": rel, "sha256": "0" * 64, "bytes": 1, **rec}
    (src / "MANIFEST.json").write_text(json.dumps(man))


def test_ade20k_train_loader_excludes_listed_and_keeps_rest(tmp_path):
    root = tmp_path / "root"
    src = root / "ade20k"
    rows, files = [], {}
    for k in (1, 2):
        stem = f"ADE_train_{k:08d}"
        img, lab = src / f"images/training/{stem}.jpg", src / f"annotations/training/{stem}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        lab.parent.mkdir(parents=True, exist_ok=True)
        a = np.random.default_rng(k).integers(0, 256, (6, 8, 3), dtype=np.uint8)  # 两张互不相似的图
        Image.fromarray(a).resize((48, 36), Image.Resampling.NEAREST).save(img, format="PNG")
        Image.fromarray(np.full((6, 8), k, np.uint8)).save(lab, format="PNG")
        files[f"images/training/{stem}.jpg"] = {"kind": "image", "image_key": f"ade20k:{stem}", "phash": phash_file(img), "split": "train"}
        files[f"annotations/training/{stem}.png"] = {"kind": "label", "split": "train"}
        rows.append({"stem": stem, "split": "train", "image": f"images/training/{stem}.jpg", "label": f"annotations/training/{stem}.png", "width": 8, "height": 6, "scene": "x"})
    make_source(root, "ade20k", files)
    (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ev = {"cvbench/test": S("cvbench/test", [("ade20k:ADE_val_00000009", files["images/training/ADE_train_00000002.jpg"]["phash"], "img/x.jpg")])}
    report, excl = td.build_report([("train", td.export_root(root)), ("eval", {"kind": td.EXPORT_KIND, "schema": 1, "sets": ev})])
    td.write_outputs(report, excl, root / "dedupe")
    res = tl.load_ade20k(root, "train")
    assert [s.fields["stem"] for s in res.records] == ["ADE_train_00000001"] and res.stats["n_dedupe_excluded"] == 1
    assert res.stats["dedupe_excluded"][0]["reasons"][0]["by"] == "phash" and res.stats["dedupe_excluded"][0]["reasons"][0]["hamming"] == 0
    # 指定 exclusions=None 且 require_dedupe=False：不排除
    assert len(tl.load_ade20k(root, "train", exclusions=None, require_dedupe=False)) == 2


def test_load_exclusions_rejects_wrong_schema(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": 99, "sets": {}}))
    with pytest.raises(ValueError, match="schema"):
        td.load_exclusions(p)
    with pytest.raises(ValueError, match="既不是"):
        (tmp_path / "y.json").write_text("{}")
        td.load_input(tmp_path / "y.json")


def test_calibrate_random_pairs_and_known():
    sets = {
        "a/test": S("a/test", [("k:1", H0, "1"), ("k:2", H1, "2")]),
        "b/test": S("b/test", [("k:1", flip(H0, 4), "1b"), ("k:3", "fedcba9876543210", "3")]),
    }
    c = td.calibrate(sets, n_random=5000)
    assert c["known_duplicates"]["n"] == 1 and c["known_duplicates"]["max"] == 4 and c["known_duplicates"]["frac_le"]["4"] == 1.0
    assert c["distinct_random_pairs"]["n"] > 0 and "已知重合 1 对" in c["verdict"]


def _cli():
    sp = importlib.util.spec_from_file_location("_dedupe_report", REPO / "scripts/data/dedupe_report.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def test_cli_arg_errors():
    mod = _cli()
    with pytest.raises(SystemExit):
        mod.parse_args(["--out", "/tmp/x"])
    with pytest.raises(SystemExit):
        mod.parse_args(["--train-root", "/tmp/a", "--out", "/tmp/x"])
