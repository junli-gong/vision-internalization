"""E2/E3 训练数据落地（DEV-0022 任务书 §1、§3）：按成员偏移批量取 zip 成员、四个源在 fixture 上的落地 / 续传 / 核对、
D11-1 的减法与 D11-2 的场景过滤（构造的小例子与手算一致，另在 P0 存档的真实清单上复算钉住的数字）、加载器字段。

上游一律是本地假服务器（tests/test_data_net.FakeServer）上的合成 zip，与真实运行走同一条代码路径；不联网。
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import train_loaders as tl
from src.data.land import ade20k, common, net, nyu_depth_train, nyu_normals, refcoco, refcoco_train
from src.data.land.common import LocalZip, MemberJob, RemoteZip, fetch_zip_members, plan_spans, read_zip_index
from src.data.land.manifest import read_manifest
from tests.test_data_land import hf_publish, img_bytes, make_ctx, ok_all, sha
from tests.test_data_net import FakeServer

REPO = Path(__file__).resolve().parents[1]
P0 = REPO / "experiments/P0/0001_env_probe/raw/p6"


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def zip_of(members: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED, dirs: tuple[str, ...] = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=method) as zf:
        for d in dirs:
            zf.writestr(zipfile.ZipInfo(d), b"")
        for k, v in members.items():
            zf.writestr(k, v)
    return buf.getvalue()


def pins_of(data: bytes, tmp_path: Path) -> dict:
    p = tmp_path / f"pins_{sha(data)[:12]}.zip"
    p.write_bytes(data)
    return read_zip_index(LocalZip(p)).pins()


def png_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def member_gets(server: FakeServer, path: str, start_dir: int) -> list[str]:
    """对 zip 的 GET 里，读成员的那些（Range 整段落在中央目录之前、且不是 bytes=0-0 的探测；EOCD 与尾部的读取都到文件末尾）。"""
    out = []
    for m, p, r in server.log:
        if m == "GET" and p == path and r and r != "bytes=0-0":
            b = int(r.split("=")[1].split("-")[1])
            if b < start_dir:
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# 按成员偏移批量取 zip 成员
# ---------------------------------------------------------------------------


def test_read_zip_index_matches_zipfile_and_pins_tail(tmp_path):
    data = zip_of({f"d/{i}.bin": bytes([i]) * (100 + i) for i in range(20)}, dirs=("d/",))
    p = tmp_path / "a.zip"
    p.write_bytes(data)
    idx = read_zip_index(LocalZip(p))
    with zipfile.ZipFile(p) as zf:
        infos = {z.filename: z for z in zf.infolist() if not z.is_dir()}
        assert set(idx.members) == set(infos) and idx.n_entries == len(zf.infolist())
        for k, z in infos.items():
            mm = idx.members[k]
            assert (mm.header_offset, mm.compress_size, mm.file_size, mm.crc) == (z.header_offset, z.compress_size, z.file_size, z.CRC)
        assert idx.start_dir == zf.start_dir
    assert idx.tail_sha256 == hashlib.sha256(data[idx.start_dir :]).hexdigest() and idx.size == len(data)
    order = sorted(idx.members.values(), key=lambda z: z.header_offset)
    assert all(a.end_bound == b.header_offset for a, b in zip(order, order[1:])) and order[-1].end_bound == idx.start_dir


def test_read_zip_index_zip64_records(tmp_path, monkeypatch):
    """EOCD 里的条目数与偏移是 0xFFFF/0xFFFFFFFF 时走 zip64 记录（train2014.zip、nyu_depth_v2.zip 都是这样）。"""
    monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 1)  # 让 zipfile 写出 zip64 EOCD 记录与 locator
    data = bytearray(zip_of({"a.txt": b"x" * 50, "b.txt": b"y" * 70, "c.txt": b"z"}))
    monkeypatch.undo()
    i = data.rfind(b"PK\x05\x06")
    struct.pack_into("<HHLL", data, i + 8, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF)  # 只留 zip64 记录里的真值
    p = tmp_path / "z64.zip"
    p.write_bytes(bytes(data))
    idx = read_zip_index(LocalZip(p))
    assert sorted(idx.members) == ["a.txt", "b.txt", "c.txt"] and idx.n_entries == 3
    with zipfile.ZipFile(p) as zf:
        assert idx.start_dir == zf.start_dir
    buf = bytes(data)
    assert common.member_bytes(buf, 0, idx.members["b.txt"]) == b"y" * 70


def test_plan_spans_gap_and_max_span():
    M = common.ZipMember
    ms = [M(f"m{i}", o, 10, 10, 0, 0, o + 40) for i, o in enumerate((0, 40, 80, 200, 240, 1000))]
    assert [[m.name for m in s] for s in plan_spans(ms, gap=0, max_span=1 << 20)] == [["m0", "m1", "m2"], ["m3", "m4"], ["m5"]]
    assert [[m.name for m in s] for s in plan_spans(ms, gap=80, max_span=1 << 20)] == [["m0", "m1", "m2", "m3", "m4"], ["m5"]]
    assert [[m.name for m in s] for s in plan_spans(ms, gap=0, max_span=80)] == [["m0", "m1"], ["m2"], ["m3", "m4"], ["m5"]]
    assert plan_spans([], gap=0, max_span=10) == []


def test_fetch_members_remote_idempotent_resume_and_crc(server, tmp_path):
    members = {f"x/{i:03d}.bin": hashlib.sha256(bytes([i])).digest() * (50 + i) for i in range(12)}
    data = zip_of(members)
    server.files["/z.zip"] = data
    server.redirects["/z.zip"] = "/signed/z.zip"  # 像 HF 的 resolve → 签名 CDN 直链
    server.files["/signed/z.zip"] = data
    src = RemoteZip(server.url("/z.zip"), retries=1, backoff=0)
    idx = read_zip_index(src)
    out = tmp_path / "out"
    jobs = [MemberJob(k, out / k) for k in members]
    seen = []
    ok, fail = fetch_zip_members(src, idx, jobs, workers=4, gap=0, process=lambda j, p, r: {"n": p.stat().st_size}, on_done=lambda j, r: seen.append(j.member))
    assert not fail and sorted(seen) == sorted(members)
    for job, rec in ok:
        assert job.dest.read_bytes() == members[job.member] and rec["sha256"] == sha(members[job.member]) and rec["n"] == len(members[job.member]) and not rec["skipped"]
    n_member_reads = len(member_gets(server, "/signed/z.zip", idx.start_dir))
    assert n_member_reads == 1  # 12 个相邻成员合并成 1 个 Range 请求
    assert not [e for e in server.log if e[1] == "/z.zip" and e[2] not in (None, "bytes=0-0")]  # 之后都直接打签名直链

    # 重跑：全部跳过（大小 + CRC32 相符），不再读成员
    ok2, fail2 = fetch_zip_members(src, idx, jobs, workers=4)
    assert not fail2 and all(r["skipped"] for _, r in ok2)
    assert len(member_gets(server, "/signed/z.zip", idx.start_dir)) == n_member_reads
    # 删一个、改坏一个、留一个半截 .part：只补这两个
    (out / "x/003.bin").unlink()
    (out / "x/007.bin").write_bytes(b"corrupt" * 10)
    (out / "x/003.bin.part").write_bytes(b"half")
    ok3, fail3 = fetch_zip_members(src, idx, jobs, workers=4)
    assert not fail3 and sorted(j.member for j, r in ok3 if not r["skipped"]) == ["x/003.bin", "x/007.bin"]
    assert (out / "x/007.bin").read_bytes() == members["x/007.bin"] and not (out / "x/003.bin.part").exists()


def test_fetch_members_failures_are_recorded_not_skipped(server, tmp_path):
    members = {f"y/{i}.bin": bytes([65 + i]) * 3000 for i in range(4)}
    data = bytearray(zip_of(members, method=zipfile.ZIP_STORED))
    server.files["/s.zip"] = bytes(data)
    src = RemoteZip(server.url("/s.zip"), retries=1, backoff=0)
    idx = read_zip_index(src)
    # 改坏 y/1.bin 的数据（中央目录不动）：CRC32 不符只记它自己
    mm = idx.members["y/1.bin"]
    data[mm.header_offset + 30 + len(mm.name) + 10] ^= 0xFF
    server.files["/s.zip"] = bytes(data)
    jobs = [MemberJob(k, tmp_path / k) for k in members] + [MemberJob("y/missing.bin", tmp_path / "nope")]

    def process(job, path, rec):
        if job.member == "y/2.bin":
            raise ValueError("格式不对")
        return {}

    ok, fail = fetch_zip_members(src, idx, jobs, workers=2, gap=0, process=process)
    errs = {f["member"]: f["error"] for f in fail}
    assert sorted(j.member for j, _ in ok) == ["y/0.bin", "y/3.bin"]
    assert "CRC32" in errs["y/1.bin"] and "核对不通过" in errs["y/2.bin"] and "没有这个成员" in errs["y/missing.bin"]
    assert not (tmp_path / "y/1.bin").exists() and not (tmp_path / "y/2.bin").exists()
    # 整个 span 读失败（404）：span 里的成员都记失败
    server.faults["/s.zip"] = ["404"]
    ok2, fail2 = fetch_zip_members(src, idx, [MemberJob("y/0.bin", tmp_path / "again/0.bin")], workers=1)
    assert not ok2 and "404" in fail2[0]["error"]


def test_fetch_members_red_line_stops_remaining_spans(server, tmp_path):
    from src.data.land.manifest import DiskRedLine

    members = {f"r/{i}.bin": bytes([i]) * 500 for i in range(6)}
    data = zip_of(members)
    server.files["/r.zip"] = data
    src = RemoteZip(server.url("/r.zip"), retries=0, backoff=0)
    idx = read_zip_index(src)

    def on_done(job, rec):
        raise DiskRedLine("根盘剩余低于红线")

    with pytest.raises(DiskRedLine):
        fetch_zip_members(src, idx, [MemberJob(k, tmp_path / k) for k in members], workers=1, max_span=1, on_done=on_done)
    assert len(member_gets(server, "/r.zip", idx.start_dir)) == 1  # 第一个 span 之后就停了


def test_remote_zip_refreshes_signed_url_on_403(server, tmp_path, monkeypatch):
    data = zip_of({"a.bin": b"a" * 100})
    server.files["/z.zip"] = data
    server.redirects["/z.zip"] = "/signed/v1"
    server.files["/signed/v1"] = data
    src = RemoteZip(server.url("/z.zip"), retries=0, backoff=0)
    assert src._final.endswith("/signed/v1")
    server.redirects["/z.zip"] = "/signed/v2"
    server.files["/signed/v2"] = data
    real = net.range_get
    calls = []

    def flaky(url, *a, **kw):
        calls.append(url)
        if url.endswith("/signed/v1"):
            raise net.HTTPStatusError(url, 403)
        return real(url, *a, **kw)

    monkeypatch.setattr(net, "range_get", flaky)
    assert src.read(0, 4) == data[:4] and src.n_refresh == 1 and src._final.endswith("/signed/v2")
    assert [c.rsplit("/", 1)[1] for c in calls] == ["v1", "v2"]


def test_range_get_checks_total_length(server):
    server.files["/f"] = b"0123456789"
    assert net.range_get(server.url("/f"), 2, 4, expect_total=10, retries=0) == b"234"
    with pytest.raises(net.FetchError, match="总长"):
        net.range_get(server.url("/f"), 2, 4, expect_total=11, retries=0)
    with pytest.raises(net.HTTPStatusError) as e:
        net.range_get(server.url("/nope"), 0, 1, retries=0)
    assert e.value.status == 404
    assert net.resolve_final_url(server.url("/f"), retries=0) == (server.url("/f"), 10)


# ---------------------------------------------------------------------------
# D11-1：构造的小例子与手算一致；真实 P0 清单复算钉住的数字
# ---------------------------------------------------------------------------

# 手算：train 并集 {1,2,3,4,10}；评测并集 {2,10,11,12}；保留 {1,3,4}；剔除 2（refcocog/val）、10（refcoco/val、refcoco+/val）
SMALL_TRAIN = {"refcoco": {1, 2, 3}, "refcoco+": {1, 2}, "refcocog": {3, 4, 10}}
SMALL_EVAL = {"refcoco": {"val": {10}, "testA": {11}, "testB": set()}, "refcoco+": {"val": {10}, "testA": {11}, "testB": set()}, "refcocog": {"val": {2}, "test": {12}}}


def test_d11_1_split_small_example_by_hand():
    d = refcoco_train.d11_1_split(SMALL_TRAIN, SMALL_EVAL)
    assert d["kept"] == [1, 3, 4] and (d["n_train_union"], d["n_eval_union"], d["n_removed"], d["n_kept"]) == (5, 4, 2, 3)
    assert d["removed"] == [{"image_id": 2, "in_eval": ["refcocog/val"]}, {"image_id": 10, "in_eval": ["refcoco+/val", "refcoco/val"]}]
    assert d["cross"]["refcocog train ∩ refcoco eval"] == 1 and d["cross"]["refcoco train ∩ refcocog eval"] == 1 and d["cross"]["refcoco train ∩ refcoco eval"] == 0


def test_d11_1_on_p0_image_lists_reproduces_pinned_numbers():
    """P0 用官方 pickle 逐条统计的 per-split image id（refs_image_ids.json）上复算 D11-1，得到 SPEC 钉住的数字。"""
    d = json.loads((P0 / "A/refs_image_ids.json").read_text())["per_dataset_split"]
    names = {"refcoco": "refcoco(unc)", "refcoco+": "refcoco+(unc)", "refcocog": "refcocog(umd)"}
    train = {n: set(d[k]["train"]) for n, k in names.items()}
    ev = {n: {sp: set(d[k][sp]) for sp in refcoco.SPEC["datasets"][n]["eval_splits"]} for n, k in names.items()}
    got = refcoco_train.d11_1_split(train, ev)
    exp = refcoco_train.SPEC["expect_d11_1"]
    assert {k: got[k] for k in ("n_train_union", "n_eval_union", "n_removed", "n_kept", "kept_digest", "removed_digest")} == {k: exp[k] for k in ("n_train_union", "n_eval_union", "n_removed", "n_kept", "kept_digest", "removed_digest")}
    assert {k: got["cross"][k] for k in exp["cross"]} == exp["cross"]
    assert (exp["n_kept"], exp["n_removed"]) == (24_407, 3_751)  # data_sources A §1.5：28,158 − 3,751
    for n, (nr, ns, ni) in refcoco_train.SPEC["expect_train"].items():
        assert len(train[n]) == ni


# ---------------------------------------------------------------------------
# refcoco_train：fixture 上的落地
# ---------------------------------------------------------------------------


def _refs_and_instances(name: str, train: set[int], ev: dict[str, set[int]]):
    refs, anns, images = [], [], {}
    rid = {"refcoco": 0, "refcoco+": 1000, "refcocog": 2000}[name]
    for split, ids in [("train", train), *ev.items()]:
        for i in sorted(ids):
            rid += 1
            aid = 900_000 + rid
            refs.append({"ref_id": rid, "ann_id": aid, "image_id": i, "split": split, "category_id": 1 + i % 2, "file_name": f"COCO_train2014_{i:012d}_1.jpg", "sent_ids": [rid * 10, rid * 10 + 1], "sentences": [{"sent_id": rid * 10 + k, "sent": f"{name} thing {i} {k}", "raw": f"{name.upper()} THING {i} {k}", "tokens": ["thing"]} for k in range(2)]})
            anns.append({"id": aid, "image_id": i, "category_id": 1 + i % 2, "bbox": [1.0, 2.0, 3.0 + i % 3, 4.0], "area": 12.0, "iscrowd": 0, "segmentation": [[1, 2, 4, 2, 4, 6]]})
            images[i] = {"id": i, "file_name": f"COCO_train2014_{i:012d}.jpg", "height": 20 + i % 4, "width": 30 + i % 5}
    inst = {"info": {}, "images": list(images.values()), "annotations": anns, "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]}
    return refs, inst


def refcoco_train_upstream(server: FakeServer, tmp_path: Path) -> dict:
    datasets, refer_pins, inst_pins, expect_train = {}, {}, {}, {}
    dims = {}
    for name in refcoco_train.DATASETS:
        ds = dict(refcoco.SPEC["datasets"][name])
        refs, inst = _refs_and_instances(name, SMALL_TRAIN[name], SMALL_EVAL[name])
        for im in inst["images"]:
            dims[im["id"]] = (im["width"], im["height"])
        pk = pickle.dumps(refs)
        ij = json.dumps(inst).encode()
        z = zip_of({ds["refs_member"]: pk, ds["instances_member"]: ij, f"{name}/refs(google).p": b"unused" * 100})
        server.files[f"/refer/{name}.zip"] = z
        ds.update(zip_url=server.url(f"/refer/{name}.zip"), refs_sha256=sha(pk))
        datasets[name] = ds
        refer_pins[name] = pins_of(z, tmp_path)
        inst_pins[name] = (len(ij), f"{zlib.crc32(ij) & 0xFFFFFFFF:08x}")
        tr = [r for r in refs if r["split"] == "train"]
        expect_train[name] = (len(tr), sum(len(r["sentences"]) for r in tr), len({r["image_id"] for r in tr}))
    imgs = {f"train2014/COCO_train2014_{i:012d}.jpg": img_bytes(w, h, i) for i, (w, h) in sorted(dims.items())}
    coco = zip_of(imgs, method=zipfile.ZIP_STORED, dirs=("train2014/",))
    server.files["/zips/train2014.zip"] = coco
    d = refcoco_train.d11_1_split(SMALL_TRAIN, SMALL_EVAL)
    return {
        "refcoco_spec": {"datasets": datasets},
        "refer_zip_pins": refer_pins,
        "instances_pins": inst_pins,
        "expect_train": expect_train,
        "expect_d11_1": {**{k: d[k] for k in ("n_train_union", "n_eval_union", "n_removed", "n_kept", "kept_digest", "removed_digest")}, "cross": {"refcocog train ∩ refcoco eval": 1, "refcoco train ∩ refcocog eval": 1}},
        "coco_zip_url": server.url("/zips/train2014.zip"),
        "coco_zip_pins": pins_of(coco, tmp_path),
        "_images": imgs,
    }


def test_refcoco_train_land_d11_1_resume_and_loader(server, tmp_path):
    spec = refcoco_train_upstream(server, tmp_path)
    imgs = spec.pop("_images")
    ctx = make_ctx(tmp_path)
    refcoco_train.land(ctx, spec)
    src = ctx.root / "refcoco_train"
    man = read_manifest(src)
    ok_all(man)
    assert man["script"]["path"] == "scripts/data/land_train.py"
    landed = sorted(k for k, v in man["files"].items() if v.get("kind") == "image")
    assert landed == [f"images/COCO_train2014_{i:012d}.jpg" for i in (1, 3, 4)]  # 手算的保留集
    for rel in landed:
        rec = man["files"][rel]
        i = int(rel[-16:-4])
        assert rec["image_key"] == f"coco:{i}" and rec["split"] == "train" and len(rec["phash"]) == 16 and rec["crc32"]
        assert (src / rel).read_bytes() == imgs[f"train2014/{Path(rel).name}"] and rec["sha256"] == sha(imgs[f"train2014/{Path(rel).name}"])
    d = json.loads((src / "d11_1.json").read_text())
    assert d["kept"] == [1, 3, 4] and [r["image_id"] for r in d["removed"]] == [2, 10]
    assert man["stats"]["d11_1"]["n_kept"] == 3 and man["stats"]["per_dataset"]["refcocog"]["train_after_d11_1"]["images"] == 2
    doc = json.loads((src / "refcocog_umd_train.json").read_text())
    assert doc["splits"] == ["train"] and sorted({r["image_id"] for r in doc["refs"]}) == [3, 4]  # refcocog train 里的 10 被剔除
    assert not list(ctx.tmpdir.rglob("*_instances.json")) and not list(ctx.tmpdir.rglob("*_refs.p"))

    # 重跑：图像全部跳过，不读 train2014.zip 的成员
    idx = read_zip_index(LocalZip(_tmpzip(tmp_path, server.files["/zips/train2014.zip"])))
    n0 = len(member_gets(server, "/zips/train2014.zip", idx.start_dir))
    refcoco_train.land(ctx, spec)
    assert len(member_gets(server, "/zips/train2014.zip", idx.start_dir)) == n0
    # 删一张：只补这一张
    (src / landed[1]).unlink()
    refcoco_train.land(ctx, spec)
    assert len(member_gets(server, "/zips/train2014.zip", idx.start_dir)) == n0 + 1
    ok_all(read_manifest(src))

    # 加载器：一句一条，框 xyxy、多边形、感知哈希
    res = tl.load_refcoco_train(ctx.root, "refcoco", require_dedupe=False)
    assert len(res) == 2 * 2 and {s.fields["image_id"] for s in res.records} == {1, 3}
    s = res.records[0]
    assert s.task == "refexp" and s.split == "train" and s.image_keys == (f"coco:{s.fields['image_id']}",) and len(s.phash[0]) == 16
    assert s.gt["box"][2] > s.gt["box"][0] and s.gt["mask"]["polygons"] == [[1, 2, 4, 2, 4, 6]] and "height" in s.gt["mask"]
    with pytest.raises(FileNotFoundError, match="硬约束 7"):
        tl.load_refcoco_train(ctx.root, "refcoco")


def _tmpzip(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / f"t_{sha(data)[:10]}.zip"
    p.write_bytes(data)
    return p


def test_refcoco_train_rejects_changed_upstream(server, tmp_path):
    spec = refcoco_train_upstream(server, tmp_path)
    spec.pop("_images")
    spec["coco_zip_pins"] = {**spec["coco_zip_pins"], "tail_sha256": "0" * 64}
    with pytest.raises(zipfile.BadZipFile, match="钉住"):
        refcoco_train.land(make_ctx(tmp_path), spec)
    spec2 = refcoco_train_upstream(server, tmp_path)
    spec2.pop("_images")
    spec2["expect_d11_1"] = {**spec2["expect_d11_1"], "n_kept": 4}
    with pytest.raises(ValueError, match="D11-1"):
        refcoco_train.land(make_ctx(tmp_path / "b"), spec2)


def test_refcoco_train_local_zips_and_limit(server, tmp_path):
    spec = refcoco_train_upstream(server, tmp_path)
    spec.pop("_images")
    refer_dir = tmp_path / "refer"
    refer_dir.mkdir()
    for name in refcoco_train.DATASETS:
        (refer_dir / f"{name}.zip").write_bytes(server.files[f"/refer/{name}.zip"])
    local = _tmpzip(tmp_path, server.files["/zips/train2014.zip"])
    server.files.clear()  # 断网：只能用本地副本
    ctx = make_ctx(tmp_path, limit=2)
    ctx.options.update(refer_zip_dir=refer_dir, local_zip=local)
    refcoco_train.land(ctx, spec)
    man = read_manifest(ctx.root / "refcoco_train")
    ok_all(man)
    assert sorted(k for k, v in man["files"].items() if v.get("kind") == "image") == [f"images/COCO_train2014_{i:012d}.jpg" for i in (1, 3)]
    assert all(u["access"] == "local" for u in man["upstream"])
    res = tl.load_refcoco_train(ctx.root, "refcocog", require_dedupe=False, strict=False)
    assert res.stats["n_missing"] == 2 and {s.fields["image_id"] for s in res.records} == {3}


# ---------------------------------------------------------------------------
# nyu_depth_train：D11-2 的场景过滤与落地
# ---------------------------------------------------------------------------

NYU_HW = (12, 16)
NYU_PAIRS = [("kitchen_0001a", "00001"), ("kitchen_0001a", "00002"), ("bookstore_0001d", "00003"), ("nyu_office_0", "00004"), ("office_0002b", "00005"), ("bookstore_0001j", "00006")]
NYU_TEST_SCENES = ("bookstore_0001", "office_0002")


def test_base_scene_and_scene_filter_by_hand():
    assert nyu_depth_train.base_scene("bookstore_0001d") == "bookstore_0001"
    assert nyu_depth_train.base_scene("nyu_office_0") == "nyu_office_0" and nyu_depth_train.base_scene("bathroom_0001") == "bathroom_0001"
    assert nyu_depth_train.base_scene("study_room_0004b") == "study_room_0004"
    pairs = [{"scene": s, "frame": f} for s, f in NYU_PAIRS]
    kept, st = nyu_depth_train.scene_filter(pairs, NYU_TEST_SCENES)
    assert [(p["scene"], p["frame"]) for p in kept] == [("kitchen_0001a", "00001"), ("kitchen_0001a", "00002"), ("nyu_office_0", "00004")]
    assert (st["n_pairs"], st["n_removed"], st["n_kept"]) == (6, 3, 3) and st["removed_dirs"] == {"bookstore_0001d": 1, "bookstore_0001j": 1, "office_0002b": 1}


def test_d11_2_on_real_bts_list_reproduces_pinned_numbers():
    """仓库里的 BTS 清单（sha256 = 固定 commit）× 钉住的 215 个 test 场景：去掉 1,383 帧（bookstore_0001d–j），剩 22,848。"""
    spec = nyu_depth_train.SPEC
    p = REPO / spec["bts"]["repo_copy"]
    assert net.sha256_file(p) == spec["bts"]["sha256"] and p.stat().st_size == spec["bts"]["bytes"]
    assert net.git_blob_sha1(p) == spec["bts"]["git_blob_sha1"]
    pairs = nyu_depth_train.parse_bts_list(p.read_text())
    kept, st = nyu_depth_train.scene_filter(pairs, nyu_depth_train.TEST_SCENES)
    assert {k: st[k] for k in ("n_pairs", "n_removed", "n_kept", "removed_dirs")} == spec["expect"]
    assert st["removed_base_scenes"] == ["bookstore_0001"] and len(pairs) == 24_231


def test_test_scenes_equal_p0_record():
    rec = json.loads((P0 / "B/nyu_labeled_scenes_names.json").read_text())
    assert list(nyu_depth_train.TEST_SCENES) == rec["test_scenes_sorted"] and len(nyu_depth_train.TEST_SCENES) == rec["n_test_scenes"] == 215
    assert hashlib.sha256("\n".join(nyu_depth_train.TEST_SCENES).encode()).hexdigest() == nyu_depth_train.SPEC["test_scenes_sha256"]
    assert not set(nyu_depth_train.TEST_SCENES) & set(rec["train_scenes_sorted"])


def test_parse_bts_list_rejects_bad_lines():
    with pytest.raises(ValueError, match="第 1 行"):
        nyu_depth_train.parse_bts_list("/a_0001/rgb_00001.jpg /b_0001/sync_depth_00001.png 518.8579\n")


def nyu_upstream(server: FakeServer, tmp_path: Path, *, bad_depth: str | None = None) -> tuple[dict, dict]:
    H, W = NYU_HW
    text = "".join(f"/{s}/rgb_{f}.jpg /{s}/sync_depth_{f}.png 518.8579\n" for s, f in NYU_PAIRS)
    members, depth = {}, {}
    for k, (s, f) in enumerate(NYU_PAIRS):
        members[f"nyu_depth_v2/sync/{s}/rgb_{f}.jpg"] = img_bytes(W, H, k)
        a = (np.arange(H * W, dtype=np.uint16).reshape(H, W) * 7 + 500 + k) % 10000
        a[0, 0] = 0
        depth[f"{s}/{f}"] = a
        members[f"nyu_depth_v2/sync/{s}/sync_depth_{f}.png"] = png_bytes(a.astype(np.uint8) if f"{s}/{f}" == bad_depth else a)
    members["nyu_depth_v2/official_splits/test/kitchen/rgb_00000.jpg"] = img_bytes(W, H, 99)
    z = zip_of(members, dirs=("nyu_depth_v2/", "nyu_depth_v2/sync/"))
    rev = "f" * 40
    hf_publish(server, "aradhye/nyu_depth_v2", rev, {"nyu_depth_v2.zip": z})
    bts = tmp_path / "bts.txt"
    bts.write_text(text)
    _, st = nyu_depth_train.scene_filter([{"scene": s, "frame": f} for s, f in NYU_PAIRS], NYU_TEST_SCENES)
    spec = {
        "bts": {**nyu_depth_train.SPEC["bts"], "sha256": sha(text.encode()), "bytes": len(text.encode()), "n_pairs": len(NYU_PAIRS), "url": server.url("/bts.txt"), "repo_copy": "does/not/exist.txt"},
        "test_scenes": NYU_TEST_SCENES,
        "test_scenes_sha256": hashlib.sha256("\n".join(NYU_TEST_SCENES).encode()).hexdigest(),
        "hf": {**nyu_depth_train.SPEC["hf"], "rev": rev, "lfs_sha256": sha(z)},
        "zip_pins": pins_of(z, tmp_path),
        "expect": {k: st[k] for k in ("n_pairs", "n_removed", "n_kept", "removed_dirs")},
        "hw": NYU_HW,
    }
    server.files["/bts.txt"] = text.encode()
    return spec, {"zip": z, "depth": depth, "members": members, "bts": bts, "rev": rev}


def test_nyu_depth_train_land_resume_and_loader(server, tmp_path):
    spec, up = nyu_upstream(server, tmp_path)
    ctx = make_ctx(tmp_path)
    ctx.options["hf_base"] = server.base
    nyu_depth_train.land(ctx, spec)
    src = ctx.root / "nyu_depth_train"
    man = read_manifest(src)
    ok_all(man)
    assert man["stats"]["d11_2"]["n_removed"] == 3 and man["stats"]["d11_2"]["n_kept"] == 3
    assert man["upstream"][0]["read_from"] == "GitHub 固定 commit"  # 仓库副本不存在、没给 --bts-list → 下载
    rows = [json.loads(l) for l in (src / "pairs_train.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["kitchen_0001a/00001", "kitchen_0001a/00002", "nyu_office_0/00004"]
    assert not list(src.rglob("*bookstore*")) and not list(src.rglob("*office_0002b*")) and not (src / "official_splits").exists()
    r0 = man["files"]["sync/kitchen_0001a/rgb_00001.jpg"]
    assert r0["image_key"] == "nyuraw:kitchen_0001a/00001" and r0["split"] == "train" and len(r0["phash"]) == 16
    assert man["files"]["sync/kitchen_0001a/sync_depth_00001.png"]["png_bit_depth"] == 16
    path = f"/datasets/aradhye/nyu_depth_v2/resolve/{up['rev']}/nyu_depth_v2.zip"
    idx = read_zip_index(LocalZip(_tmpzip(tmp_path, up["zip"])))
    n0 = len(member_gets(server, path, idx.start_dir))
    nyu_depth_train.land(ctx, spec)  # 重跑：不读成员
    assert len(member_gets(server, path, idx.start_dir)) == n0

    res = tl.load_nyu_depth_train(ctx.root, require_dedupe=False)
    assert len(res) == 3
    s = res.records[0]
    d = tl.read_depth_m(s.gt["depth"])
    want = up["depth"]["kitchen_0001a/00001"].astype(np.float32) / 1000.0
    assert d.dtype == np.float32 and d.shape == NYU_HW and np.array_equal(d, want) and d[0, 0] == 0.0
    assert s.task == "depth" and s.gt["depth_units"] == "meters" and s.fields["focal"] == pytest.approx(518.8579)


def test_nyu_depth_train_local_bts_and_zip_and_bad_depth(server, tmp_path):
    spec, up = nyu_upstream(server, tmp_path, bad_depth="nyu_office_0/00004")
    local = _tmpzip(tmp_path, up["zip"])
    server.files.clear()
    ctx = make_ctx(tmp_path)
    ctx.options.update(local_zip=local, bts_list=up["bts"])
    nyu_depth_train.land(ctx, spec)
    man = read_manifest(ctx.root / "nyu_depth_train")
    assert man["upstream"][0]["read_from"].startswith("--bts-list")
    assert [v for v in man["verification"] if v["what"].startswith("本地 zip 的整文件 sha256")][0]["ok"]
    assert len(man["failures"]) == 1 and "16 位" in man["failures"][0]["error"]
    bad = [v for v in man["verification"] if not v["ok"]]
    assert len(bad) == 1 and "齐全" in bad[0]["what"] and bad[0]["n_incomplete"] == 1


# ---------------------------------------------------------------------------
# nyu_normals
# ---------------------------------------------------------------------------

NN_HW = (8, 10)
NN_IDS = {"train": [3, 5], "test": [1, 2, 4]}


def normals_upstream(server: FakeServer, tmp_path: Path) -> dict:
    H, W = NN_HW
    members = {}
    for sp, ids in NN_IDS.items():
        for i in ids:
            rgba = np.zeros((H, W, 4), np.uint8)
            rgba[..., :3] = np.asarray(Image.open(io.BytesIO(img_bytes(W, H, i))).convert("RGB"))
            rgba[..., 3] = 255
            members[f"nyu/{sp}/img/{i - 1:06d}.png"] = png_bytes(rgba)
            nm = np.zeros((H, W, 4), np.uint8)
            nm[..., 0], nm[..., 1], nm[..., 2], nm[..., 3] = 128, 128, 255, 255  # ≈ (0, 0, 1)
            nm[:2] = (0, 0, 0, 255)  # 无效
            members[f"nyu/{sp}/norm/{i - 1:06d}.png"] = png_bytes(nm)
    z = zip_of(members, dirs=("nyu/", "nyu/train/", "nyu/test/"))
    server.files["/drive/nyu_dataset.zip"] = z
    return {
        "zip_url": server.url("/drive/nyu_dataset.zip"),
        "zip_pins": pins_of(z, tmp_path),
        "expect": {sp: (len(ids), nyu_normals.index_digest(ids)) for sp, ids in NN_IDS.items()},
        "hw": NN_HW,
        "splits_url": server.url("/no/splits.mat"),  # 404：只按钉住的指纹核对，记一条 note
    }


def test_nyu_normals_land_and_loader(server, tmp_path):
    spec = normals_upstream(server, tmp_path)
    ctx = make_ctx(tmp_path)
    nyu_normals.land(ctx, spec)
    src = ctx.root / "nyu_normals"
    man = read_manifest(src)
    ok_all(man)
    assert any("splits.mat 取不到" in n for n in man["notes"])
    idx = json.loads((src / "index.json").read_text())["images"]
    assert [(r["split"], r["idx"]) for r in idx] == [("test", 1), ("test", 2), ("test", 4), ("train", 3), ("train", 5)]
    rec = man["files"]["test/img/0001.png"]
    assert rec["image_key"] == "nyuv2:1" and rec["split"] == "test" and rec["png_color_type"] == 6 and len(rec["phash"]) == 16
    assert man["files"]["train/norm/0005.png"]["kind"] == "normal"
    st = read_zip_index(LocalZip(_tmpzip(tmp_path, server.files["/drive/nyu_dataset.zip"]))).start_dir
    n0 = len(member_gets(server, "/drive/nyu_dataset.zip", st))
    nyu_normals.land(ctx, spec)  # 重跑：不读成员
    assert len(member_gets(server, "/drive/nyu_dataset.zip", st)) == n0
    ok_all(read_manifest(src))
    res = tl.load_nyu_normals(ctx.root, "test")  # 评测划分不读剔除清单
    assert [s.fields["idx"] for s in res.records] == [1, 2, 4] and res.stats["role"] == "eval"
    n, v = tl.read_normals(res.records[0].gt["normal"])
    assert n.shape == (*NN_HW, 3) and n.dtype == np.float32 and not v[:2].any() and v[2:].all()
    assert np.allclose(np.linalg.norm(n[v], axis=-1), 1.0, atol=1e-6) and np.all(n[~v] == 0)
    assert np.allclose(n[v].mean(0), [0.0, 0.0, 1.0], atol=0.01)
    with pytest.raises(FileNotFoundError):
        tl.load_nyu_normals(ctx.root, "train")
    assert len(tl.load_nyu_normals(ctx.root, "train", require_dedupe=False)) == 2


def test_nyu_normals_split_mismatch_is_flagged(server, tmp_path):
    spec = normals_upstream(server, tmp_path)
    spec["expect"] = {**spec["expect"], "test": (3, "0" * 64)}
    nyu_normals.land(make_ctx(tmp_path), spec)
    bad = [v for v in read_manifest(tmp_path / "root/nyu_normals")["verification"] if not v["ok"]]
    assert len(bad) == 1 and bad[0]["what"].startswith("test：")


# ---------------------------------------------------------------------------
# ade20k
# ---------------------------------------------------------------------------


def ade_upstream(server: FakeServer, tmp_path: Path, *, bad_label: bool = False) -> dict:
    members = {"ADEChallengeData2016/objectInfo150.txt": b"Idx\tRatio\tTrain\tVal\tName\n1\t0.1\t1\t1\twall\n", "ADEChallengeData2016/sceneCategories.txt": b""}
    scenes = []
    for sp, d, n in (("train", "training", 3), ("val", "validation", 2)):
        for k in range(1, n + 1):
            stem = f"ADE_{sp}_{k:08d}"
            w, h = 12 + k, 9 + k
            members[f"ADEChallengeData2016/images/{d}/{stem}.jpg"] = img_bytes(w, h, k + (10 if sp == "val" else 0))
            lab = (np.arange(w * h).reshape(h, w) % 151).astype(np.uint8)
            if bad_label and stem == "ADE_train_00000002":
                lab[0, 0] = 200
            members[f"ADEChallengeData2016/annotations/{d}/{stem}.png"] = png_bytes(lab)
            scenes.append(f"{stem} scene_{k}")
    members["ADEChallengeData2016/sceneCategories.txt"] = ("\n".join(scenes) + "\n").encode()
    z = zip_of(members, dirs=("ADEChallengeData2016/", "ADEChallengeData2016/images/"))
    server.files["/ade/ADEChallengeData2016.zip"] = z
    return {"zip_url": server.url("/ade/ADEChallengeData2016.zip"), "zip_pins": pins_of(z, tmp_path), "expect": {"train": 3, "val": 2}, "_zip": z}


def test_ade20k_land_resume_and_loader(server, tmp_path):
    spec = ade_upstream(server, tmp_path)
    z = spec.pop("_zip")
    ctx = make_ctx(tmp_path)
    ade20k.land(ctx, spec)
    src = ctx.root / "ade20k"
    man = read_manifest(src)
    ok_all(man)
    assert man["files"]["images/training/ADE_train_00000002.jpg"]["image_key"] == "ade20k:ADE_train_00000002"
    assert man["files"]["annotations/validation/ADE_val_00000001.png"]["kind"] == "label"
    assert man["files"]["annotations/training/ADE_train_00000001.png"]["max_label"] <= 150
    tr = [json.loads(l) for l in (src / "train.jsonl").read_text().splitlines()]
    assert [r["stem"] for r in tr] == ["ADE_train_00000001", "ADE_train_00000002", "ADE_train_00000003"] and tr[0]["scene"] == "scene_1"
    idx = read_zip_index(LocalZip(_tmpzip(tmp_path, z)))
    n0 = len(member_gets(server, "/ade/ADEChallengeData2016.zip", idx.start_dir))
    assert n0 == 1  # 全部成员相邻，合并成一个 Range 请求
    (src / "annotations/training/ADE_train_00000003.png").write_bytes(b"bad")
    ade20k.land(ctx, spec)
    assert len(member_gets(server, "/ade/ADEChallengeData2016.zip", idx.start_dir)) == n0 + 1
    ok_all(read_manifest(src))
    res = tl.load_ade20k(ctx.root, "val")
    lab = tl.read_ade_label(res.records[0].gt["label"])
    assert lab.dtype == np.uint8 and lab.max() <= 150 and res.records[0].task == "semseg" and res.records[0].gt["ignore_index"] == 0
    assert len(tl.load_ade20k(ctx.root, "train", require_dedupe=False)) == 3


def test_ade20k_network_failure_then_resume(server, tmp_path, monkeypatch):
    """读成员的请求一直 503：成员逐条记进 failures（不静默跳过）；网络恢复后重跑补齐。"""
    spec = ade_upstream(server, tmp_path)
    z = spec.pop("_zip")
    st = read_zip_index(LocalZip(_tmpzip(tmp_path, z))).start_dir
    ctx = make_ctx(tmp_path)
    real_read = common.RemoteZip.read

    def flaky(self, start, end):
        if end <= st:
            raise net.FetchError(self.url, "重试 2 次后仍失败：HTTP 503")
        return real_read(self, start, end)

    monkeypatch.setattr(common.RemoteZip, "read", flaky)
    ade20k.land(ctx, spec)
    monkeypatch.undo()
    man = read_manifest(ctx.root / "ade20k")
    assert len(man["failures"]) == 12 and all("503" in f["error"] for f in man["failures"])
    assert (ctx.root / "ade20k/train.jsonl").read_text() == ""
    ade20k.land(ctx, spec)
    man = read_manifest(ctx.root / "ade20k")
    ok_all(man)
    assert len((ctx.root / "ade20k/train.jsonl").read_text().splitlines()) == 3


def test_ade20k_label_out_of_range_fails(server, tmp_path):
    spec = ade_upstream(server, tmp_path, bad_label=True)
    spec.pop("_zip")
    ade20k.land(make_ctx(tmp_path, limit=3), spec)
    man = read_manifest(tmp_path / "root/ade20k")
    assert len(man["failures"]) == 1 and "200 > 150" in man["failures"][0]["error"]
    assert sorted(json.loads(l)["stem"] for l in (tmp_path / "root/ade20k/train.jsonl").read_text().splitlines()) == ["ADE_train_00000001"]


# ---------------------------------------------------------------------------
# 预算、dry-run、CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", [refcoco_train, nyu_depth_train, nyu_normals, ade20k])
def test_dry_run_and_budget_before_any_network(mod, server, tmp_path):
    from src.data.land.common import DryRun
    from src.data.land.manifest import BudgetExceeded

    spec = {"coco_zip_url": server.url("/x"), "zip_url": server.url("/x"), "hf_base": server.base}
    with pytest.raises(DryRun):
        mod.land(make_ctx(tmp_path, dry_run=True), spec)
    with pytest.raises(BudgetExceeded):
        mod.land(make_ctx(tmp_path, budget_gb=1e-6), spec)
    assert server.log == []


def test_full_estimates_cover_pinned_bytes():
    assert refcoco_train.SPEC["kept_bytes"] == 4_146_591_678 and nyu_depth_train.SPEC["kept_bytes"] == 3_961_165_376
    assert ade20k.SPEC["zip_pins"]["size"] == 967_382_037 and nyu_normals.SPEC["zip_pins"]["size"] == 1_008_950_467


def test_cli_args():
    import importlib.util

    sp = importlib.util.spec_from_file_location("_land_train", REPO / "scripts/data/land_train.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    assert list(mod.LANDERS) == ["refcoco_train", "nyu_depth_train", "nyu_normals", "ade20k"]
    with pytest.raises(SystemExit):
        mod.parse_args(["ade20k", "--root", "/tmp/x", "--workers", "9"])
    with pytest.raises(SystemExit):
        mod.parse_args(["all", "--root", "/tmp/x", "--local-zip", "/tmp/a.zip"])
    a = mod.parse_args(["nyu_depth_train", "--root", "/tmp/x", "--hf-base", "https://hf-mirror.com"])
    assert a.budget_gb == 10.0 and a.min_free_gb == 3.0 and a.workers == 8 and a.hf_base == "https://hf-mirror.com"
