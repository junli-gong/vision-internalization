"""落地脚本（src/data/land/，DEV-0012 任务书 §1、§5）：清单与哈希、断点续传与幂等、预算拒绝、磁盘红线、逐源端到端。

每个源都用本地假服务器（tests/test_data_net.FakeServer）提供合成的上游文件（zip、parquet、json、HDF5 .mat、图像），
走与服务器上相同的代码路径；不联网。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.land import coco, hfbench, manifest, nyuv2, odinw, refcoco
from src.data.land.common import DryRun, LandContext, image_ext
from src.data.land.manifest import GB, BudgetExceeded, DiskRedLine, Manifest, check_budget, read_manifest
from tests.test_data_net import FakeServer

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_sha1(b: bytes) -> str:
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()


def img_bytes(w: int, h: int, seed: int, fmt: str = "JPEG") -> bytes:
    yy, xx = np.mgrid[:h, :w]
    a = np.stack([(xx * 7 + seed * 13) % 256, (yy * 5 + seed * 29) % 256, ((xx + yy) * 3 + seed) % 256], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format=fmt)
    return buf.getvalue()


def zip_bytes(members: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=method) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    return buf.getvalue()


def parquet_bytes(rows: list[dict], row_group_size: int = 2) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf, row_group_size=row_group_size)
    return buf.getvalue()


def hf_publish(server: FakeServer, repo: str, rev: str, files: dict[str, bytes], *, kind: str = "dataset", lfs: bool = True, non_lfs: tuple[str, ...] = ()) -> None:
    """在假服务器上模拟 HF：tree API（按目录）与 resolve 直链。"""
    prefix = "/datasets/" if kind == "dataset" else "/"
    plural = "datasets" if kind == "dataset" else "models"
    dirs: dict[str, list] = {}
    for p, data in files.items():
        d = p.rsplit("/", 1)[0] if "/" in p else ""
        e = {"type": "file", "path": p, "size": len(data)}
        if lfs and p not in non_lfs:
            e.update(oid="0" * 40, lfs={"oid": sha(data), "size": len(data), "pointerSize": 133})
        else:
            e["oid"] = git_sha1(data)
        dirs.setdefault(d, []).append(e)
        server.files[f"{prefix}{repo}/resolve/{rev}/{p}"] = data
    for d, entries in dirs.items():
        server.files[f"/api/{plural}/{repo}/tree/{rev}/{d}"] = json.dumps(entries).encode()


def make_ctx(tmp_path: Path, **kw) -> LandContext:
    kw.setdefault("retries", 2)
    kw.setdefault("backoff", 0.0)
    return LandContext(root=tmp_path / "root", tmpdir=tmp_path / "tmp", log=lambda s: None, **kw)


def ok_all(man: dict) -> None:
    bad = [v for v in man["verification"] if not v["ok"]]
    assert not bad, bad
    assert not man["failures"], man["failures"]


# ---------------------------------------------------------------------------
# 清单、预算、红线
# ---------------------------------------------------------------------------


def test_budget_refuses_over_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "disk_free", lambda p: 100 * GB)
    with pytest.raises(BudgetExceeded, match="预算"):
        check_budget(int(10.5 * GB), budget_gb=10, root=tmp_path, min_free_gb=0)
    assert check_budget(int(9 * GB), budget_gb=10, root=tmp_path, min_free_gb=3)["estimate_bytes"] == 9 * GB


def test_budget_refuses_below_red_line(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "disk_free", lambda p: 5 * GB)
    with pytest.raises(BudgetExceeded, match="红线"):
        check_budget(int(2.5 * GB), budget_gb=10, root=tmp_path, min_free_gb=3)
    assert check_budget(int(1.5 * GB), budget_gb=10, root=tmp_path, min_free_gb=3)["root_free_bytes_before"] == 5 * GB


def test_manifest_journal_resume_and_write(tmp_path):
    m = Manifest(tmp_path / "src", "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None)
    m.add_file("b.jpg", sha256="bb", nbytes=2)
    m.add_file("a.jpg", sha256="aa", nbytes=1, phash="0" * 16)
    with open(tmp_path / "src" / manifest.JOURNAL_NAME, "a") as f:
        f.write('{"path": "half')  # 中断时写了半行
    m2 = Manifest(tmp_path / "src", "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None)
    assert m2.known_sha256("a.jpg") == "aa" and m2.known_sha256("b.jpg") == "bb" and m2.known_sha256("c") is None
    m.write()
    man = read_manifest(tmp_path / "src")
    assert list(man["files"]) == ["a.jpg", "b.jpg"] and man["total_bytes"] == 3 and man["n_files"] == 2
    assert man["script"]["git"]["commit"] is None or len(man["script"]["git"]["commit"]) == 40
    assert man["landed_at_utc"].endswith("+00:00")
    assert not (tmp_path / "src" / manifest.JOURNAL_NAME).exists()
    m3 = Manifest(tmp_path / "src", "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None)
    assert m3.known_sha256("a.jpg") == "aa"  # 从 MANIFEST.json 读回


def test_manifest_carries_over_untouched_files(tmp_path):
    d = tmp_path / "src"
    m = Manifest(d, "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None)
    for name in ("a.bin", "b.bin", "c.bin"):
        (d / name).write_bytes(name.encode())
        m.add_file(name, sha256=sha(name.encode()), nbytes=len(name))
    m.write()
    (d / "b.bin").unlink()  # 盘上已没有的不带
    (d / "c.bin").write_bytes(b"changed size")  # 字节数变了的不带
    m2 = Manifest(d, "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None)
    m2.add_file("new.bin", sha256="n", nbytes=1)
    m2.write()
    man = read_manifest(d)
    assert sorted(man["files"]) == ["a.bin", "new.bin"] and man["files"]["a.bin"]["carried_over"] and man["n_files_carried_over"] == 1


def test_df_red_line_stops(tmp_path, monkeypatch):
    m = Manifest(tmp_path / "src", "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=3, log=lambda s: None)
    monkeypatch.setattr(manifest, "disk_free", lambda p: int(2.9 * GB))
    with pytest.raises(DiskRedLine):
        m.df("某一步")
    assert m.data["df_log"][-1]["root_free"] == int(2.9 * GB)


def test_image_ext_by_magic():
    assert image_ext(img_bytes(4, 3, 0, "JPEG")) == ".jpg"
    assert image_ext(img_bytes(4, 3, 0, "PNG")) == ".png"
    with pytest.raises(ValueError):
        image_ext(b"not an image")


def test_cli_rejects_too_many_workers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_land_e1", REPO / "scripts/data/land_e1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.parse_args(["coco_val2017", "--root", "/tmp/x", "--workers", "9"])
    a = mod.parse_args(["all", "--root", "/tmp/x"])
    assert a.budget_gb == 10.0 and a.min_free_gb == 3.0 and a.workers == 8
    assert list(mod.LANDERS) == ["refcoco", "coco_val2017", "odinw13", "countbench", "nyuv2", "blink_reldepth", "cvbench", "embspatial"]


# ---------------------------------------------------------------------------
# COCO val2017：清单、幂等、续传、失败记录、预算拒绝
# ---------------------------------------------------------------------------

COCO_IDS = [139, 39769, 5, 17, 23, 42, 77, 101, 250, 333]


def coco_upstream(server: FakeServer) -> dict:
    images = []
    for i in COCO_IDS:
        fn = f"{i:012d}.jpg"
        w, h = 40 + i % 7, 30 + i % 5
        server.files[f"/val2017/{fn}"] = img_bytes(w, h, i)
        images.append({"id": i, "file_name": fn, "coco_url": server.url(f"/val2017/{fn}"), "width": w, "height": h, "license": 1, "flickr_url": "", "date_captured": ""})
    anns = [
        {"id": 1000 + k, "image_id": i, "category_id": 1 + k % 2, "bbox": [1.0, 2.0, 3.0, 4.0], "area": 12.0, "iscrowd": 0, "segmentation": [[1, 2, 4, 2, 4, 6]]}
        for k, i in enumerate(COCO_IDS[:7])
    ]
    inst = {"info": {"description": "fake"}, "licenses": [{"id": 1, "name": "x", "url": ""}], "images": images, "annotations": anns, "categories": [{"id": 1, "name": "cat", "supercategory": "a"}, {"id": 2, "name": "dog", "supercategory": "a"}]}
    raw = json.dumps(inst).encode()
    server.files["/annotations/annotations_trainval2017.zip"] = zip_bytes({"annotations/instances_val2017.json": raw, "annotations/captions_val2017.json": b"{}" * 5000})
    return {"ann_zip_url": server.url("/annotations/annotations_trainval2017.zip"), "ann_sha256": sha(raw), "ann_bytes": len(raw)}


def image_gets(server: FakeServer) -> list[str]:
    return [p for (m, p, _r) in server.log if m == "GET" and p.startswith("/val2017/")]


def test_coco_land_manifest_and_idempotent(server, tmp_path):
    from pycocotools.coco import COCO

    spec = coco_upstream(server)
    ctx = make_ctx(tmp_path)
    coco.land(ctx, spec)
    src = ctx.root / "coco_val2017"
    man = read_manifest(src)
    ok_all(man)
    chosen = sorted(set(COCO_IDS) - {139, 39769})
    assert man["selection"]["image_ids"] == chosen
    assert man["selection"]["probe_excluded_before_sampling"] == [139, 39769]
    for i in chosen:
        rel = f"images/{i:012d}.jpg"
        rec = man["files"][rel]
        assert rec["sha256"] == sha(server.files[f"/val2017/{i:012d}.jpg"]) == sha((src / rel).read_bytes())
        assert rec["bytes"] == (src / rel).stat().st_size and len(rec["phash"]) == 16 and rec["image_key"] == f"coco:{i}"
    sub = COCO(str(src / coco.SUBSET_FILE))
    assert sorted(sub.getImgIds()) == chosen and 139 not in sub.imgs
    assert man["estimate"]["estimate_bytes"] > 0 and man["df_log"][0]["step"] == "开跑前"
    assert man["upstream"][0]["member_sha256"] == spec["ann_sha256"]
    assert not list((ctx.tmpdir).rglob("instances_val2017.json"))  # 临时文件已删

    n_before = len(image_gets(server))
    coco.land(ctx, spec)  # 重跑：全部跳过
    assert len(image_gets(server)) == n_before
    assert {k: v["sha256"] for k, v in read_manifest(src)["files"].items()} == {k: v["sha256"] for k, v in man["files"].items()}

    # 删一张、改坏一张：重跑只补这两张
    (src / f"images/{chosen[0]:012d}.jpg").unlink()
    (src / f"images/{chosen[1]:012d}.jpg").write_bytes(b"corrupt")
    coco.land(ctx, spec)
    new = image_gets(server)[n_before:]
    assert sorted(new) == sorted([f"/val2017/{chosen[0]:012d}.jpg", f"/val2017/{chosen[1]:012d}.jpg"])
    assert sha((src / f"images/{chosen[1]:012d}.jpg").read_bytes()) == sha(server.files[f"/val2017/{chosen[1]:012d}.jpg"])


def test_coco_land_records_unreachable_urls(server, tmp_path):
    spec = coco_upstream(server)
    server.faults["/val2017/000000000017.jpg"] = ["404"] * 5
    ctx = make_ctx(tmp_path)
    coco.land(ctx, spec)
    man = read_manifest(ctx.root / "coco_val2017")
    assert len(man["failures"]) == 1 and man["failures"][0]["url"].endswith("000000000017.jpg") and "404" in man["failures"][0]["error"]
    assert "images/000000000017.jpg" not in man["files"]
    # 修好后重跑补齐
    server.faults.clear()
    coco.land(ctx, spec)
    man = read_manifest(ctx.root / "coco_val2017")
    assert not man["failures"] and "images/000000000017.jpg" in man["files"]


def test_coco_land_resumes_partial_image(server, tmp_path):
    spec = coco_upstream(server)
    ctx = make_ctx(tmp_path)
    data = server.files["/val2017/000000000042.jpg"]
    part = ctx.root / "coco_val2017/images/000000000042.jpg.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:100])
    coco.land(ctx, spec)
    ranges = [r for (m, p, r) in server.log if m == "GET" and p == "/val2017/000000000042.jpg"]
    assert ranges == ["bytes=100-"]
    assert (ctx.root / "coco_val2017/images/000000000042.jpg").read_bytes() == data


def test_coco_land_budget_refusal_and_dry_run(server, tmp_path):
    spec = coco_upstream(server)
    with pytest.raises(BudgetExceeded):
        coco.land(make_ctx(tmp_path, budget_gb=1e-6), spec)
    with pytest.raises(DryRun):
        coco.land(make_ctx(tmp_path, dry_run=True), spec)
    assert server.log == []  # 预算检查在任何网络请求之前
    assert not (tmp_path / "root/coco_val2017/images").exists()


def test_coco_land_limit_lands_subset_only(server, tmp_path):
    spec = coco_upstream(server)
    ctx = make_ctx(tmp_path, limit=3)
    coco.land(ctx, spec)
    man = read_manifest(ctx.root / "coco_val2017")
    assert len(man["selection"]["image_ids"]) == 8 and len([k for k in man["files"] if k.startswith("images/")]) == 3


# ---------------------------------------------------------------------------
# HF 上的四个基准
# ---------------------------------------------------------------------------


def test_blink_land(server, tmp_path):
    rows = []
    for i in (3, 1, 2, 10):
        rows.append(
            {
                "idx": f"val_Relative_Depth_{i}",
                "question": "Which point is closer to the camera?",
                "sub_task": "Relative Depth",
                "image_1": {"bytes": img_bytes(20 + i, 15, i), "path": None},
                "image_2": None,
                "image_3": None,
                "image_4": None,
                "choices": ["A is closer", "B is closer"],
                "answer": "(A)" if i % 2 else "(B)",
                "prompt": "p",
                "explanation": "",
            }
        )
    rev = "r" * 40
    path = "Relative_Depth/val-00000-of-00001.parquet"
    hf_publish(server, "BLINK-Benchmark/BLINK", rev, {path: parquet_bytes(rows)})
    ctx = make_ctx(tmp_path)
    hfbench.land_blink(ctx, {"rev": rev, "path": path, "hf_base": server.base})
    src = ctx.root / "blink_reldepth"
    man = read_manifest(src)
    ok_all(man)
    out = [json.loads(l) for l in (src / "val.jsonl").read_text().splitlines()]
    assert [r["idx"] for r in out] == [f"val_Relative_Depth_{i}" for i in (1, 2, 3, 10)]
    assert [r["answer_letter"] for r in out] == ["A", "B", "A", "B"]
    for r in out:
        assert r["image"].endswith(".jpg") and man["files"][r["image"]]["sha256"] == r["image_sha256"] == sha((src / r["image"]).read_bytes())
    assert not list(ctx.tmpdir.rglob("*.parquet"))


def cvbench_upstream(server: FakeServer, rev: str) -> None:
    rows2d = [
        {"type": "2D", "task": "Count", "question": "How many?", "choices": ["1", "2"], "answer": "(B)", "prompt": "p", "filename": "img/2D/count/coco_1.png", "source": "COCO", "source_dataset": "COCO 2017 Validation Set", "source_filename": "/coco/val2017/000000000139.jpg", "target_class": "cat", "target_size": 10, "bbox": None},
        {"type": "2D", "task": "Count", "question": "How many organs?", "choices": ["3", "2", "1", "0"], "answer": "(C)", "prompt": "p", "filename": "img/2D/count/ade20k_10.png", "source": "ADE20K", "source_dataset": "ADE20K Validation Set", "source_filename": "ADE_val_00000248.jpg", "target_class": None, "target_size": None, "bbox": None},
        {"type": "2D", "task": "Relation", "question": "Left?", "choices": ["yes", "no"], "answer": "(A)", "prompt": "p", "filename": "img/2D/relation/coco_2.png", "source": "COCO", "source_dataset": "COCO 2017 Validation Set", "source_filename": "/coco/val2017/000000000005.jpg", "target_class": None, "target_size": None, "bbox": None},
    ]
    rows3d = [
        {"type": "3D", "task": "Depth", "question": "Closer?", "choices": ["table", "bed"], "answer": "(A)", "prompt": "p", "filename": "img/3D/depth/omni3d_sunrgbd_0.jpg", "source": "Omni3D", "source_dataset": "Omni3D_SUNRGBD", "source_filename": "SUNRGBD/kv1/NYUdata/NYU0021//image/NYU0021.jpg", "target_class": None, "target_size": None, "bbox": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]},
        {"type": "3D", "task": "Distance", "question": "Which?", "choices": ["a", "b"], "answer": "(B)", "prompt": "p", "filename": "img/3D/distance/omni3d_hypersim_0.jpg", "source": "Omni3D", "source_dataset": "Omni3D_Hypersim", "source_filename": "hypersim/ai_037_009/images/scene_cam_00_final_preview/frame.0003.tonemap.jpg", "target_class": None, "target_size": None, "bbox": [[1.0, 2.0, 3.0, 4.0]]},
    ]

    def pq_rows(rows, start):
        return [{"idx": start + k, **{kk: vv for kk, vv in r.items() if kk not in ("filename",)}, "filename": r["filename"], "image": {"bytes": img_bytes(30, 20 + k, start + k), "path": None}} for k, r in enumerate(rows)]

    files = {
        "test_2d.jsonl": "".join(json.dumps(r) + "\n" for r in rows2d).encode(),
        "test_3d.jsonl": "".join(json.dumps(r) + "\n" for r in rows3d).encode(),
        "test_2d.parquet": parquet_bytes(pq_rows(rows2d, 0)),
        "test_3d.parquet": parquet_bytes(pq_rows(rows3d, len(rows2d))),
    }
    hf_publish(server, "nyu-visionx/CV-Bench", rev, files, non_lfs=("test_2d.jsonl", "test_3d.jsonl"))


@pytest.mark.parametrize("limit", [None, 1])
def test_cvbench_land(server, tmp_path, limit):
    rev = "c" * 40
    cvbench_upstream(server, rev)
    ctx = make_ctx(tmp_path, limit=limit)
    hfbench.land_cvbench(ctx, {"rev": rev, "hf_base": server.base})
    src = ctx.root / "cvbench"
    man = read_manifest(src)
    ok_all(man)
    out = [json.loads(l) for l in (src / "test.jsonl").read_text().splitlines()]
    assert [r["idx"] for r in out] == [0, 1, 2, 3, 4]
    assert [r["image_key"] for r in out] == ["coco:139", "ade20k:ADE_val_00000248", "coco:5", "nyuv2:21", "hypersim:hypersim/ai_037_009/images/scene_cam_00_final_preview/frame.0003.tonemap.jpg"]
    assert [r["answer_letter"] for r in out] == ["B", "C", "A", "A", "B"]
    with_img = [r["idx"] for r in out if "image" in r]
    assert with_img == ([0, 1, 2, 3, 4] if limit is None else [0, 3])
    df = man["dedupe_facts"]
    assert df["nyuv2_image_keys"] == ["nyuv2:21"] and df["coco_val2017_image_keys"] == ["coco:139", "coco:5"]
    assert df["unique_images_per_source_dataset"]["COCO 2017 Validation Set"] == 2
    if limit is None:
        assert any(v["what"].startswith("HF nyu-visionx/CV-Bench") and v["method"] == "git blob sha1" for v in man["verification"])
        assert not list(ctx.tmpdir.rglob("*.parquet"))
    else:
        gets = [r for (m, p, r) in server.log if m == "GET" and p.endswith(".parquet")]
        assert gets and all(r is not None for r in gets)  # 只用 Range 读


@pytest.mark.parametrize("limit", [None, 2])
def test_embspatial_land(server, tmp_path, limit):
    rows = []
    for src_name, n in (("mp3d", 2), ("ai2thor", 1), ("scannet", 1)):
        for i in range(n):
            rows.append({"data_source": src_name, "scene_id": f"s{i}", "question_id": f"{src_name}_{i}", "question": "q", "relation": "left", "image": base64.b64encode(img_bytes(16, 12, i + len(rows))).decode(), "answer_options": ["a", "b", "c", "d"], "answer": i % 4, "objects": [{"name": "a", "bbox": [1, 2, 3, 4]}]})
    raw = json.dumps(rows).encode()
    rev = "e" * 40
    hf_publish(server, "Phineas476/EmbSpatial-Bench", rev, {"embspatial_bench.json": raw})
    ctx = make_ctx(tmp_path, limit=limit)
    hfbench.land_embspatial(ctx, {"rev": rev, "hf_base": server.base})
    src = ctx.root / "embspatial"
    man = read_manifest(src)
    ok_all(man)
    out = [json.loads(l) for l in (src / "test.jsonl").read_text().splitlines()]
    n = len(rows) if limit is None else limit
    assert [r["question_id"] for r in out] == [r["question_id"] for r in rows][:n]
    assert [r["answer_letter"] for r in out] == ["A", "B", "A", "A"][:n]
    for r in out:
        assert (src / r["image"]).read_bytes() == base64.b64decode(rows[[x["question_id"] for x in rows].index(r["question_id"])]["image"])
        assert "image" in r and "image_base64" not in r


def test_iter_json_array_streams_small_chunks():
    data = json.dumps([{"a": i, "s": "é" * (i * 7)} for i in range(20)]).encode()
    got = list(hfbench.iter_json_array(io.BytesIO(data), None, chunk=5))
    assert got == json.loads(data)
    assert list(hfbench.iter_json_array(io.BytesIO(data), 3, chunk=7)) == json.loads(data)[:3]


def countbench_upstream(server: FakeServer, rev: str) -> dict:
    rows = []
    for i in range(6):
        has = i not in (2, 4)
        rows.append({"image_url": f"http://x/{i}.jpg", "text": f"{i + 2} cats", "number": i + 2, "image": {"bytes": img_bytes(10, 8, i), "path": None} if has else None})
    hf_publish(server, "nielsr/countbench", rev, {"data/train-00000-of-00001-x.parquet": parquet_bytes(rows)})
    official = json.dumps([{"image_url": r["image_url"], "text": r["text"], "number": r["number"]} for r in rows]).encode()
    server.files["/CountBench.json"] = official
    return {"rev": rev, "hf_base": server.base, "official_json_url": server.url("/CountBench.json"), "official_json_sha256": sha(official)}


def test_countbench_land(server, tmp_path):
    spec = countbench_upstream(server, "k" * 40)
    ctx = make_ctx(tmp_path)
    hfbench.land_countbench(ctx, spec)
    src = ctx.root / "countbench"
    man = read_manifest(src)
    ok_all(man)
    out = [json.loads(l) for l in (src / "countbench.jsonl").read_text().splitlines()]
    assert [r["has_image"] for r in out] == [True, True, False, True, False, True]
    assert [r["row"] for r in out if "image" in r] == [0, 1, 3, 5]
    assert man["stats"]["n_with_image"] == 4


def test_countbench_land_flags_official_mismatch(server, tmp_path):
    spec = countbench_upstream(server, "k" * 40)
    bad = json.loads(server.files["/CountBench.json"])
    bad[1]["number"] = 99
    server.files["/CountBench.json"] = json.dumps(bad).encode()
    spec["official_json_sha256"] = sha(server.files["/CountBench.json"])
    ctx = make_ctx(tmp_path)
    hfbench.land_countbench(ctx, spec)
    v = {x["what"]: x["ok"] for x in read_manifest(ctx.root / "countbench")["verification"]}
    assert v["HF 打包版与官方 CountBench.json 逐条同序一致（image_url, text, number）"] is False


# ---------------------------------------------------------------------------
# NYUv2：合成的 MATLAB v7.3（HDF5）文件与 splits.mat
# ---------------------------------------------------------------------------

N = nyuv2.N_LABELED


def nyu_upstream(tmp_path: Path) -> tuple[bytes, bytes, dict[int, dict]]:
    """合成的 labeled .mat：形状与官方相同（未写的块不占空间），只写小样本会读的三张（test 前两张、开发子集第一张）。"""
    import h5py
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    perm = rng.permutation(np.arange(1, N + 1))
    test = np.sort(perm[:654]).astype(np.uint16)
    train = np.sort(perm[654:]).astype(np.uint16)
    sp = tmp_path / "splits.mat"
    savemat(sp, {"trainNdxs": train.reshape(-1, 1), "testNdxs": test.reshape(-1, 1)})
    dev = nyuv2.dev_train_ids([int(x) for x in train], tmp_path / "no_config.json")
    content: dict[int, dict] = {}
    p = tmp_path / "labeled.mat"
    yy, xx = np.mgrid[:480, :640]
    written = {}
    with h5py.File(p, "w") as f:
        im = f.create_dataset("images", shape=(N, 3, 640, 480), dtype="uint8", chunks=(1, 3, 42, 480), compression="gzip")
        dp = f.create_dataset("depths", shape=(N, 640, 480), dtype="float32", chunks=(1, 33, 480), compression="gzip")
        rd = f.create_dataset("rawDepths", shape=(N, 640, 480), dtype="float32", chunks=(1, 33, 480), compression="gzip")
        refs = f.create_group("#refs#")
        scenes = f.create_dataset("scenes", shape=(1, N), dtype=h5py.ref_dtype)
        types = f.create_dataset("sceneTypes", shape=(1, N), dtype=h5py.ref_dtype)
        for k, (name, typ) in enumerate((("kitchen_0004", "kitchen"), ("office_0004", "office"))):
            s = refs.create_dataset(f"s{k}", data=np.array([ord(c) for c in name], dtype=np.uint16).reshape(-1, 1))
            t = refs.create_dataset(f"t{k}", data=np.array([ord(c) for c in typ], dtype=np.uint16).reshape(-1, 1))
            content[k] = {"s": s.ref, "t": t.ref, "name": name, "type": typ}
        sc = np.empty((1, N), dtype=object)
        ty = np.empty((1, N), dtype=object)
        for i in range(N):
            sc[0, i] = content[i % 2]["s"]
            ty[0, i] = content[i % 2]["t"]
        scenes[...] = sc
        types[...] = ty
        for idx in (int(test[0]), int(test[1]), dev[0]):
            rgb = np.stack([(xx + idx) % 256, (yy * 2 + idx) % 256, (xx + yy) % 256], -1).astype(np.uint8)
            dfill = (0.7 + (xx + yy) / 200.0 + idx * 0.001).astype(np.float32)
            draw = dfill.copy()
            draw[:, :20] = 0.0
            im[idx - 1] = rgb.transpose(2, 1, 0)
            dp[idx - 1] = dfill.T
            rd[idx - 1] = draw.T
            written[idx] = {"rgb": rgb, "depth_filled": dfill, "depth_raw": draw, "scene": content[(idx - 1) % 2]["name"], "scene_type": content[(idx - 1) % 2]["type"]}
    return p.read_bytes(), sp.read_bytes(), written


def nyu_spec(server: FakeServer, mat: bytes, splits: bytes) -> dict:
    server.files["/nyu/labeled.mat"] = mat
    server.files["/nyu/splits.mat"] = splits
    return {"mat_url": server.url("/nyu/labeled.mat"), "mat_bytes": len(mat), "splits_url": server.url("/nyu/splits.mat"), "splits_sha256": sha(splits), "local_dir": "/nonexistent"}


def check_nyu_output(root: Path, written: dict[int, dict], want: list[int]) -> None:
    src = root / "nyuv2"
    man = read_manifest(src)
    ok_all(man)
    index = json.loads((src / "index.json").read_text())
    assert [r["idx"] for r in index["images"]] == sorted(want)
    for r in index["images"]:
        w = written[r["idx"]]
        assert np.array_equal(np.asarray(Image.open(src / r["rgb"])), w["rgb"])
        with np.load(src / r["depth"]) as d:
            assert np.array_equal(d["depth_raw"], w["depth_raw"]) and np.array_equal(d["depth_filled"], w["depth_filled"])
            assert d["depth_raw"].dtype == np.float32
        assert r["scene"] == w["scene"] and r["scene_type"] == w["scene_type"]
        assert man["files"][r["rgb"]]["image_key"] == f"nyuv2:{r['idx']}"
    assert "BTS" in index["eigen_crop"]["ours"] and "427×561" in index["eigen_crop"]["matlab_toolbox"]


def nyu_expected(splits: bytes, tmp_path: Path) -> tuple[list[int], list[int]]:
    p = tmp_path / "s.mat"
    p.write_bytes(splits)
    train, test = nyuv2.read_splits(p)
    dev = nyuv2.dev_train_ids(train, tmp_path / "no_config.json")
    return test, dev


def test_nyuv2_land_download_path(server, tmp_path):
    mat, splits, written = nyu_upstream(tmp_path)
    spec = nyu_spec(server, mat, splits)
    test, dev = nyu_expected(splits, tmp_path)
    ctx = make_ctx(tmp_path, limit=3, options={"dev_config": str(tmp_path / "no_config.json")})
    nyuv2.land(ctx, spec)
    check_nyu_output(ctx.root, written, [test[0], test[1], dev[0]])
    man = read_manifest(ctx.root / "nyuv2")
    assert man["upstream"][0]["sha256"] == sha(mat) and "TMPDIR" in man["upstream"][0]["official_check"]
    assert not list(ctx.tmpdir.rglob("*.mat"))  # TMPDIR 里的 .mat 用完即删


def test_nyuv2_land_local_copy_verified_by_streaming(server, tmp_path):
    mat, splits, written = nyu_upstream(tmp_path)
    spec = nyu_spec(server, mat, splits)
    local = tmp_path / "other_project"
    local.mkdir()
    (local / "nyu_depth_v2_labeled.mat").write_bytes(mat)
    test, dev = nyu_expected(splits, tmp_path)
    ctx = make_ctx(tmp_path, limit=3, options={"dev_config": str(tmp_path / "no_config.json"), "nyu_local_dir": str(local)})
    nyuv2.land(ctx, spec)
    check_nyu_output(ctx.root, written, [test[0], test[1], dev[0]])
    up = read_manifest(ctx.root / "nyuv2")["upstream"][0]
    assert up["local_path"].endswith("nyu_depth_v2_labeled.mat") and up["sha256"] == up["local_sha256"] == sha(mat)
    gets = [r for (m, p, r) in server.log if m == "GET" and p == "/nyu/labeled.mat"]
    assert gets == [None]  # 只流式读了一遍官方文件，没有落盘下载
    assert (local / "nyu_depth_v2_labeled.mat").read_bytes() == mat  # 别人的目录没被改


def test_nyuv2_land_local_mismatch_falls_back_to_download(server, tmp_path):
    mat, splits, written = nyu_upstream(tmp_path)
    spec = nyu_spec(server, mat, splits)
    local = tmp_path / "other_project"
    local.mkdir()
    bad = bytearray(mat)
    bad[-1] ^= 1
    (local / "x.mat").write_bytes(bytes(bad))
    ctx = make_ctx(tmp_path, limit=2, options={"dev_config": str(tmp_path / "no_config.json"), "nyu_local_dir": str(local)})
    nyuv2.land(ctx, spec)
    man = read_manifest(ctx.root / "nyuv2")
    v = [x for x in man["verification"] if x["what"].startswith("本地 .mat")]
    assert v and v[0]["ok"] is False
    assert man["upstream"][0]["sha256"] == sha(mat) and "TMPDIR" in man["upstream"][0]["official_check"]


def test_nyuv2_land_remote_range_mode(server, tmp_path):
    mat, splits, written = nyu_upstream(tmp_path)
    spec = nyu_spec(server, mat, splits)
    test, dev = nyu_expected(splits, tmp_path)
    ctx = make_ctx(tmp_path, limit=3, options={"dev_config": str(tmp_path / "no_config.json"), "nyu_remote": True})
    nyuv2.land(ctx, spec)
    check_nyu_output(ctx.root, written, [test[0], test[1], dev[0]])
    gets = [r for (m, p, r) in server.log if m == "GET" and p == "/nyu/labeled.mat"]
    assert gets and all(r is not None for r in gets)


def test_nyuv2_dev_ids_must_match_committed_config(tmp_path):
    train = list(range(2, 1449, 2))
    cfg_path = tmp_path / "dev.json"
    cfg_path.write_text(json.dumps({"subsets": {"nyuv2_train": {"ids": ["2", "4"]}}}))
    with pytest.raises(ValueError, match="不一致"):
        nyuv2.dev_train_ids(train, cfg_path)


def test_npz_bytes_deterministic():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b1 = nyuv2.npz_bytes(depth_raw=a, depth_filled=a + 1)
    b2 = nyuv2.npz_bytes(depth_filled=a + 1, depth_raw=a)
    assert b1 == b2
    with np.load(io.BytesIO(b1)) as d:
        assert np.array_equal(d["depth_raw"], a) and sorted(d.files) == ["depth_filled", "depth_raw"]


# ---------------------------------------------------------------------------
# ODinW13：表与 GLIP yaml 一致；端到端
# ---------------------------------------------------------------------------


def test_odinw_table_matches_glip_yaml():
    import yaml

    ev = REPO / "experiments/P0/0001_env_probe/raw/p6/C"
    members = json.loads((ev / "odinw13_members.json").read_text())
    assert sorted(odinw.CONFIGS) == sorted(p.stem for p in (ev / "odinw_glip_configs13").glob("*.yaml"))
    for name, cfg in odinw.CONFIGS.items():
        y = yaml.safe_load((ev / "odinw_glip_configs13" / f"{name}.yaml").read_text())["DATASETS"]
        test = y["TEST"].strip("()\",").strip('"')
        reg = y["REGISTER"][test]
        assert cfg["test"] == test
        assert "odinw/" + cfg["ann"] == reg["ann_file"] and "odinw/" + cfg["img_dir"] == reg["img_dir"]
        assert cfg["override"] == [c["name"] for c in sorted(json.loads(y["OVERRIDE_CATEGORY"]), key=lambda c: c["id"])]
        assert cfg["zip"] == members[name]["zip"]
    assert odinw.GLIP_REV == "66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1"


def test_odinw_land(server, tmp_path):
    imgs, anns, members = [], [], {}
    for i in range(60):
        fn = f"img_{i}.jpg"
        data = img_bytes(12 + i % 3, 9, i)
        members[f"Fake/sub/valid/{fn}"] = data
        imgs.append({"id": i, "file_name": fn, "width": 12 + i % 3, "height": 9, "license": 1, "date_captured": ""})
        anns.append({"id": i, "image_id": i, "category_id": 1 + i % 2, "bbox": [1, 1, 2, 2], "area": 4, "segmentation": [], "iscrowd": 0})
    ann = {"info": {"x": 1}, "licenses": [{"id": 1, "name": "CC BY 4.0"}], "categories": [{"id": 2, "name": "B"}, {"id": 1, "name": "A"}], "images": imgs, "annotations": anns}
    members["Fake/sub/valid/annotations_without_background.json"] = json.dumps(ann).encode()
    members["Fake/sub/README.dataset.txt"] = b"License: CC BY 4.0"
    rev = "o" * 40
    hf_publish(server, "GLIPModel/GLIP", rev, {"odinw_35/Fake.zip": zip_bytes(members, zipfile.ZIP_STORED)}, kind="model")
    configs = {"Fake_sub": {"zip": "Fake", "test": "val", "ann": "Fake/sub/valid/annotations_without_background.json", "img_dir": "Fake/sub/valid", "override": ["alpha", "beta"]}}
    ctx = make_ctx(tmp_path, limit=3)
    odinw.land(ctx, {"rev": rev, "hf_base": server.base, "configs": configs})
    src = ctx.root / "odinw13"
    man = read_manifest(src)
    ok_all(man)
    sel = man["selection"]["per_dataset"]["Fake_sub"]
    assert sel["n_population"] == 60 and len(sel["image_ids"]) == 50 and sel["landed"] == 3
    sub = json.loads((src / "Fake_sub/annotations_e1sub.json").read_text())
    assert [im["id"] for im in sub["images"]] == sel["image_ids"] and sub["glip"]["override_category_names"] == ["alpha", "beta"]
    landed = [k for k in man["files"] if "/images/" in k]
    assert len(landed) == 3
    for k in landed:
        assert (src / k).read_bytes() == members["Fake/sub/valid/" + k.rsplit("/", 1)[1]] and man["files"][k]["crc32"]
    assert man["license"]["per_dataset"]["Fake_sub"]["readme"] == {"Fake/sub/README.dataset.txt": "License: CC BY 4.0"}
    assert man["upstream"][0]["zip_lfs_sha256"] == sha(server.files["/GLIPModel/GLIP/resolve/" + rev + "/odinw_35/Fake.zip"])
    # 只用 Range 读：没有整包 GET
    gets = [r for (m, p, r) in server.log if m == "GET" and p.endswith("Fake.zip")]
    assert all(r is not None for r in gets)


# ---------------------------------------------------------------------------
# RefCOCO：Wayback zip 成员 + 镜像逐条比对 + 评测划分图像
# ---------------------------------------------------------------------------


def refcoco_upstream(server: FakeServer, *, corrupt_mirror: bool = False) -> tuple[dict, dict]:
    refs, images, anns = [], [], []
    plan = [("train", 900), ("val", 901), ("val", 902), ("testA", 903), ("testB", 904)]
    sid = 0
    for rid, (split, img_id) in enumerate(plan):
        sents = []
        for k in range(2):
            sents.append({"tokens": ["the", "cat"], "raw": f"The cat {sid}", "sent_id": sid, "sent": f"the cat {sid}"})
            sid += 1
        refs.append({"sent_ids": [s["sent_id"] for s in sents], "file_name": f"COCO_train2014_{img_id:012d}_1.jpg", "ann_id": 5000 + rid, "ref_id": rid, "image_id": img_id, "split": split, "sentences": sents, "category_id": 1})
        images.append({"id": img_id, "file_name": f"COCO_train2014_{img_id:012d}.jpg", "height": 9, "width": 12})
        anns.append({"id": 5000 + rid, "image_id": img_id, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "area": 12.0, "iscrowd": 0, "segmentation": [[1, 2, 4, 2, 4, 6]]})
        server.files[f"/train2014/COCO_train2014_{img_id:012d}.jpg"] = img_bytes(12, 9, img_id)
    refs_raw = pickle.dumps(refs, protocol=2)
    inst = json.dumps({"info": {}, "images": images, "annotations": anns, "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}]}).encode()
    z = zip_bytes({"refcoco/instances.json": inst, "refcoco/refs(unc).p": refs_raw, "refcoco/refs(google).p": b"x" * 100})
    server.files["/wb/refcoco.zip"] = z
    mirror_files = {}
    for prefix, split in (("validation", "val"), ("test", "testA"), ("testB", "testB")):
        rows = []
        for r in refs:
            if r["split"] != split:
                continue
            sents = [dict(s) for s in r["sentences"]]
            if corrupt_mirror and split == "val":
                sents[0]["sent"] = "something else"
            a = anns[[x["id"] for x in anns].index(r["ann_id"])]
            rows.append({"ref_id": r["ref_id"], "ann_id": r["ann_id"], "image_id": r["image_id"], "category_id": r["category_id"], "split": r["split"], "sent_ids": r["sent_ids"], "sentences": sents, "raw_anns": json.dumps(a)})
        mirror_files[f"data/{prefix}-00000-of-00001-abc.parquet"] = parquet_bytes(rows)
    mrev = "m" * 40
    hf_publish(server, "jxu124/refcoco", mrev, mirror_files)
    ds = {
        "zip_url": server.url("/wb/refcoco.zip"),
        "zip_bytes": len(z),
        "split_by": "unc",
        "refs_member": "refcoco/refs(unc).p",
        "refs_sha256": sha(refs_raw),
        "instances_member": "refcoco/instances.json",
        "eval_splits": ["val", "testA", "testB"],
        "expect": {"val": (2, 4, 2), "testA": (1, 2, 1), "testB": (1, 2, 1)},
        "mirror": {"repo": "jxu124/refcoco", "rev": mrev, "files": {"validation": "val", "test": "testA", "testB": "testB"}},
    }
    spec = {"datasets": {"refcoco": ds}, "image_url": server.base + "/train2014/COCO_train2014_{id:012d}.jpg", "hf_base": server.base, "eval_union_bytes": 10_000}
    return spec, {"refs": refs}


def test_refcoco_land(server, tmp_path):
    spec, up = refcoco_upstream(server)
    ctx = make_ctx(tmp_path, options={"refcoco_datasets": ["refcoco"]})
    refcoco.land(ctx, spec)
    src = ctx.root / "refcoco"
    man = read_manifest(src)
    ok_all(man)
    doc = json.loads((src / "refcoco_unc.json").read_text())
    assert [r["ref_id"] for r in doc["refs"]] == [1, 2, 3, 4]  # 训练划分不进评测文件
    assert sorted(doc["images"]) == ["901", "902", "903", "904"] and doc["categories"] == {"1": "cat"}
    landed = sorted(k for k in man["files"] if k.startswith("images/"))
    assert landed == [f"images/COCO_train2014_{i:012d}.jpg" for i in (901, 902, 903, 904)]
    assert not [p for (m, p, _r) in server.log if p.endswith("000000000900.jpg")]  # 训练图没下
    v = {x["what"]: x for x in man["verification"]}
    assert v["refcoco：与镜像逐条比对"]["n_mismatch"] == 0 and v["refcoco：与镜像逐条比对"]["n_mirror_rows"] == 4
    assert not list(ctx.tmpdir.rglob("*.p")) and not list(ctx.tmpdir.rglob("*instances.json"))


def test_refcoco_land_flags_mirror_mismatch(server, tmp_path):
    spec, _ = refcoco_upstream(server, corrupt_mirror=True)
    ctx = make_ctx(tmp_path, options={"refcoco_datasets": ["refcoco"]})
    refcoco.land(ctx, spec)
    v = {x["what"]: x for x in read_manifest(ctx.root / "refcoco")["verification"]}
    rec = v["refcoco：与镜像逐条比对"]
    assert rec["ok"] is False and rec["n_mismatch"] == 2 and rec["mismatches_head"][0]["fields"] == ["sentences"]


def test_refcoco_rejects_wrong_pickle_hash(server, tmp_path):
    spec, _ = refcoco_upstream(server)
    spec["datasets"]["refcoco"]["refs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        refcoco.land(make_ctx(tmp_path, options={"refcoco_datasets": ["refcoco"]}), spec)


def test_refs_unpickler_rejects_globals(tmp_path):
    p = tmp_path / "evil.p"
    p.write_bytes(pickle.dumps([{"a": {1, 2}}], protocol=2))  # 协议 2 里 set 要用 builtins.set 这个 global
    with pytest.raises(pickle.UnpicklingError, match="global"):
        refcoco.load_refs_pickle(p)
    p.write_bytes(pickle.dumps([{"a": [1, "x", 2.5]}], protocol=2))
    assert refcoco.load_refs_pickle(p) == [{"a": [1, "x", 2.5]}]


def test_pinned_upstream_hashes_match_p0_evidence():
    """SPEC 里钉死的 sha256 与 P0 存档的证据一致（防止手抄出错）。"""
    ev = REPO / "experiments/P0/0001_env_probe/raw/p6"
    stats = json.loads((ev / "A/refs_stats.json").read_text())
    for name, key in (("refcoco", "refcoco(unc)"), ("refcoco+", "refcoco+(unc)"), ("refcocog", "refcocog(umd)")):
        ds = refcoco.SPEC["datasets"][name]
        assert ds["refs_sha256"] == stats[key]["member_sha256"] and ds["zip_url"] == stats[key]["source_url"]
        assert ds["refs_member"] == stats[key]["member"]
        assert {k: tuple(v) for k, v in ds["expect"].items()} == {s: (v["refs"], v["sentences"], v["images"]) for s, v in stats[key]["per_split"].items() if s in ds["eval_splits"]}
    schema = json.loads((ev / "A/coco_val2017/coco_val2017_schema.json").read_text())
    assert coco.SPEC["ann_sha256"] == schema["sha256"] and coco.SPEC["ann_bytes"] == int(schema["bytes"])
    splits = json.loads((ev / "B/nyu_splits_mat.json").read_text())
    assert nyuv2.SPEC["splits_sha256"] == splits["sha256"]
    for f, mod_spec in (("C/hf/nielsr__countbench.json", hfbench.COUNTBENCH), ("C/hf/BLINK-Benchmark__BLINK.json", hfbench.BLINK), ("C/hf/nyu-visionx__CV-Bench.json", hfbench.CVBENCH), ("C/hf/Phineas476__EmbSpatial-Bench.json", hfbench.EMBSPATIAL)):
        assert json.loads((ev / f).read_text())["sha"] == mod_spec["rev"]
