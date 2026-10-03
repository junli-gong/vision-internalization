"""DEV-0026：E1 的 refcoco 读本地 refer zip（--refer-zip-dir）、与训练源共用的读 zip 代码、download_many 在磁盘红线处
叫停、Manifest 的 script 参数（DEV-0029：ade20k、nyu_depth_train、nyu_normals 也改为传参）、scripts/data/refer_zip_sha256.py。

上游一律是本地假服务器（tests/test_data_net.FakeServer）上的合成文件：三个 refer zip 的成员名、划分与真实的一致，内容是
造的；COCO 评测图与 jxu124 镜像也在假服务器上。download_many 的红线测试用假网络（替换 net._open）与假磁盘统计。不联网。
"""

from __future__ import annotations

import importlib.util
import io
import json
import pickle
import re
import struct
import threading
import time
import zipfile
import zlib
from pathlib import Path

import pytest

from src.data.land import ade20k, manifest, net, nyu_depth_train, nyu_normals, refcoco, refcoco_train
from src.data.land.common import LocalZip, read_zip_index
from src.data.land.manifest import DEFAULT_SCRIPT, GB, DiskRedLine, Manifest, read_manifest
from tests.test_data_land import hf_publish, img_bytes, make_ctx, ok_all, parquet_bytes, sha
from tests.test_data_net import FakeServer

REPO = Path(__file__).resolve().parents[1]
B200_MANIFEST = REPO / "experiments/P0/0010_e1_data/manifests/refcoco.MANIFEST.json"  # B200 上 RUN-0010 落地的 E1 refcoco 清单
NAMES = ("refcoco", "refcoco+", "refcocog")
MREV = "m" * 40


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def _load(rel: str):
    sp = importlib.util.spec_from_file_location("_dev0026_" + Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def _zip(members: dict[str, bytes], dirs: tuple[str, ...] = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in dirs:
            zf.writestr(zipfile.ZipInfo(d), b"")
        for k, v in members.items():
            zf.writestr(k, v)
    return buf.getvalue()


def _dims(image_id: int) -> tuple[int, int]:
    return 12 + image_id % 5, 9 + image_id % 3


def build_refer(server: FakeServer, tmp_path: Path) -> tuple[dict, dict[str, bytes], dict[str, list[int]]]:
    """三个数据集的合成 refer zip 放到假服务器的 /wb/<name>.zip（代替 Wayback），jxu124 镜像与评测图也放上去。

    返回 (spec 覆盖，{name: zip 字节}，{name: val 划分按 ref_id 排序的 image_id})。refcoco 与 refcoco+ 共用一部分图。"""
    datasets, zips, pins, ipins, fulls, val_imgs = {}, {}, {}, {}, {}, {}
    pin_dir = tmp_path / "pins"
    pin_dir.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(NAMES):
        real = refcoco.SPEC["datasets"][name]
        base = 0 if name != "refcocog" else 500  # refcoco 与 refcoco+ 的图有重合（与真实情况一样）
        refs, anns, images = [], [], {}
        rid, sid = 100 * k, 10_000 * k
        for split in ["train", *real["eval_splits"]]:
            n = 4 if split == "val" else 1
            for j in range(n):
                img = base + {"train": 1, "val": 10, "testA": 30, "testB": 40, "test": 50}[split] + j + (k if split == "val" and j >= 2 else 0)
                rid += 1
                sents = [{"tokens": ["a"], "raw": f"{name} Thing {rid} {q}", "sent_id": sid + q, "sent": f"{name} thing {rid} {q}"} for q in range(2)]
                sid += 2
                refs.append({"sent_ids": [s["sent_id"] for s in sents], "file_name": f"COCO_train2014_{img:012d}_1.jpg", "ann_id": 900_000 + rid, "ref_id": rid, "image_id": img, "split": split, "sentences": sents, "category_id": 1 + img % 2})
                anns.append({"id": 900_000 + rid, "image_id": img, "category_id": 1 + img % 2, "bbox": [1.0, 2.0, 3.0 + img % 3, 4.0], "area": 12.0, "iscrowd": 0, "segmentation": [[1, 2, 4, 2, 4, 6]]})
                w, h = _dims(img)
                images[img] = {"id": img, "file_name": f"COCO_train2014_{img:012d}.jpg", "height": h, "width": w}
                server.files[f"/train2014/COCO_train2014_{img:012d}.jpg"] = img_bytes(w, h, img)
        pk = pickle.dumps(refs, protocol=2)
        ij = json.dumps({"info": {}, "images": list(images.values()), "annotations": anns, "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]}).encode()
        z = _zip({real["instances_member"]: ij, f"{name}/refs(google).p": b"g" * 300, real["refs_member"]: pk}, dirs=(f"{name}/",) if name == "refcocog" else ())
        zips[name] = z
        server.files[f"/wb/{name}.zip"] = z
        (pin_dir / f"{name}.zip").write_bytes(z)
        pins[name] = read_zip_index(LocalZip(pin_dir / f"{name}.zip")).pins()
        ipins[name] = (len(ij), f"{zlib.crc32(ij) & 0xFFFFFFFF:08x}")
        fulls[name] = sha(z)
        expect = {}
        for sp in real["eval_splits"]:
            rs = [r for r in refs if r["split"] == sp]
            expect[sp] = (len(rs), sum(len(r["sentences"]) for r in rs), len({r["image_id"] for r in rs}))
        val_imgs[name] = [r["image_id"] for r in sorted(refs, key=lambda r: r["ref_id"]) if r["split"] == "val"]
        mirror = {}
        for prefix, sp in real["mirror"]["files"].items():
            rows = [{"ref_id": r["ref_id"], "ann_id": r["ann_id"], "image_id": r["image_id"], "category_id": r["category_id"], "split": r["split"], "sent_ids": r["sent_ids"], "sentences": r["sentences"], "raw_anns": json.dumps(anns[[a["id"] for a in anns].index(r["ann_id"])])} for r in refs if r["split"] == sp]
            mirror[f"data/{prefix}-00000-of-00001-x{k}.parquet"] = parquet_bytes(rows)
        hf_publish(server, real["mirror"]["repo"], MREV, mirror)
        datasets[name] = {**real, "zip_url": server.url(f"/wb/{name}.zip"), "zip_bytes": len(z), "refs_sha256": sha(pk), "expect": expect, "mirror": {**real["mirror"], "rev": MREV}}
    spec = {
        "datasets": datasets,
        "refer_zip_pins": pins,
        "instances_pins": ipins,
        "refer_zip_sha256": fulls,
        "image_url": server.base + "/train2014/COCO_train2014_{id:012d}.jpg",
        "hf_base": server.base,
        "eval_union_bytes": 10_000,
    }
    return spec, zips, val_imgs


def local_copies(tmp_path: Path, zips: dict[str, bytes]) -> Path:
    d = tmp_path / "refer_zips"
    d.mkdir(parents=True, exist_ok=True)
    for name, z in zips.items():
        (d / f"{name}.zip").write_bytes(z)
    return d


def corrupt_member(data: bytes, member: str) -> bytes:
    """改坏一个成员的压缩数据（中央目录与尾部不变，所以尾部 sha256 的钉住值照样相符，只有 CRC32 / 解压能查出来）。"""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        off = zf.getinfo(member).header_offset
    n_name, n_extra = struct.unpack("<HH", data[off + 26 : off + 30])
    b = bytearray(data)
    b[off + 30 + n_name + n_extra + 7] ^= 0xFF
    return bytes(b)


def wb_gets(server: FakeServer, since: int = 0) -> list:
    return [e for e in server.log[since:] if e[1].startswith("/wb/")]


# ---------------------------------------------------------------------------
# E1 refcoco：--refer-zip-dir
# ---------------------------------------------------------------------------


def test_e1_refcoco_local_zips_land_same_as_remote(server, tmp_path, monkeypatch):
    spec, zips, _ = build_refer(server, tmp_path)
    # 不给 --refer-zip-dir：走原路径，不碰共用的本地 zip 代码
    with monkeypatch.context() as mp:
        mp.setattr(refcoco, "fetch_refer_members", lambda *a, **k: pytest.fail("远端路径不应调用 fetch_refer_members"))
        remote = make_ctx(tmp_path / "remote")
        refcoco.land(remote, spec)
    man_r = read_manifest(remote.root / "refcoco")
    ok_all(man_r)
    # 原路径的清单格式与 B200 已落地的那份相同（顶层键、upstream 每条的键、无 notes）
    b200 = json.loads(B200_MANIFEST.read_text())
    assert list(man_r) == list(b200) and [list(u) for u in man_r["upstream"]] == [list(u) for u in b200["upstream"]] and man_r["notes"] == []
    assert [v["what"] for v in man_r["verification"] if not v["what"].startswith("HF ")] == [v["what"] for v in b200["verification"] if not v["what"].startswith("HF ")]

    refer_dir = local_copies(tmp_path, zips)
    for name in NAMES:
        del server.files[f"/wb/{name}.zip"]  # 断开"Wayback"：只能读本地副本
    n0 = len(server.log)
    local = make_ctx(tmp_path / "local", options={"refer_zip_dir": refer_dir})
    refcoco.land(local, spec)
    assert wb_gets(server, n0) == []
    man = read_manifest(local.root / "refcoco")
    ok_all(man)
    # 产物与远端落地逐字节相同（三份评测标注、全部评测图）
    assert {k: v["sha256"] for k, v in man["files"].items()} == {k: v["sha256"] for k, v in man_r["files"].items()}
    # 清单记下用的是本地 zip，以及每个 zip 的完整 sha256
    assert [u["dataset"] for u in man["upstream"]] == list(NAMES)
    for u in man["upstream"]:
        n = u["dataset"]
        assert u["access"] == "local" and u["where"] == str(refer_dir / f"{n}.zip") and u["copy_of"] == spec["datasets"][n]["zip_url"]
        assert u["zip_sha256"] == sha(zips[n]) and u["zip"] == spec["refer_zip_pins"][n]
        assert set(u["members"]) == {spec["datasets"][n]["refs_member"], spec["datasets"][n]["instances_member"]}
    v = {x["what"]: x for x in man["verification"]}
    for n in NAMES:
        assert v[f"refer {n}.zip：zip 大小、条目数与中央目录 sha256 = SPEC 钉住的值"]["ok"]
        assert v[f"refer {n}.zip：整个文件的 sha256 = SPEC 钉住的值"]["ok"]
        assert v[f"{n}：instances.json 的字节数与 CRC32 = 钉住的值"]["ok"] and v[f"{n}：与镜像逐条比对"]["n_mismatch"] == 0
    assert any("--refer-zip-dir" in s for s in man["notes"])
    assert not list(local.tmpdir.rglob("*.p")) and not list(local.tmpdir.rglob("*instances.json"))


def test_e1_refcoco_local_zip_unpinned_full_sha_is_recorded_not_checked(server, tmp_path):
    spec, zips, _ = build_refer(server, tmp_path)
    spec["refer_zip_sha256"] = dict.fromkeys(NAMES)
    ctx = make_ctx(tmp_path, options={"refer_zip_dir": local_copies(tmp_path, zips), "refcoco_datasets": ["refcocog"]})
    refcoco.land(ctx, spec)
    man = read_manifest(ctx.root / "refcoco")
    ok_all(man)
    assert man["upstream"][0]["zip_sha256"] == sha(zips["refcocog"])
    assert not [x for x in man["verification"] if "整个文件" in x["what"]]
    assert any(sha(zips["refcocog"]) in s and "还没钉住" in s for s in man["notes"])


@pytest.mark.parametrize("case", ["tail", "full_sha", "instances_pin", "member_crc", "missing"])
def test_e1_refcoco_local_zip_mismatch_refuses_to_land(case, server, tmp_path):
    spec, zips, _ = build_refer(server, tmp_path)
    refer_dir = local_copies(tmp_path, zips)
    if case == "tail":
        spec["refer_zip_pins"]["refcoco"] = {**spec["refer_zip_pins"]["refcoco"], "tail_sha256": "0" * 64}
        err, match = zipfile.BadZipFile, "钉住"
    elif case == "full_sha":
        spec["refer_zip_sha256"]["refcoco"] = "0" * 64
        err, match = zipfile.BadZipFile, "整个文件的 sha256"
    elif case == "instances_pin":
        spec["instances_pins"]["refcoco"] = (1, "00000000")
        err, match = zipfile.BadZipFile, "instances.json"
    elif case == "member_crc":  # 尾部完好、成员数据坏了；整个文件的 sha256 不钉，确认 CRC32 那一关自己就能拦住
        spec["refer_zip_sha256"]["refcoco"] = None
        (refer_dir / "refcoco.zip").write_bytes(corrupt_member(zips["refcoco"], "refcoco/instances.json"))
        err, match = net.FetchError, "成员取回失败"
    else:
        (refer_dir / "refcoco+.zip").unlink()
        err, match = FileNotFoundError, re.escape(f"缺 ['{refer_dir / 'refcoco+.zip'}']")
    ctx = make_ctx(tmp_path, options={"refer_zip_dir": refer_dir})
    with pytest.raises(err, match=match):
        refcoco.land(ctx, spec)
    src = ctx.root / "refcoco"
    assert not (src / "MANIFEST.json").exists() and not list(src.glob("*.json")) and not (src / "images").exists()
    assert wb_gets(server) == []


def test_refer_pins_defined_once_and_consistent():
    """钉住值只在 refcoco.SPEC 定义一份，训练源引用同一个对象；尾部钉住的 size 就是 E1 的 zip_bytes。"""
    for key in ("refer_zip_pins", "instances_pins", "refer_zip_sha256"):
        assert refcoco_train.SPEC[key] is refcoco.SPEC[key]
    for n in NAMES:
        assert refcoco.SPEC["refer_zip_pins"][n]["size"] == refcoco.SPEC["datasets"][n]["zip_bytes"]
        assert re.fullmatch(r"[0-9a-f]{64}", refcoco.SPEC["refer_zip_sha256"][n])  # D57 起三个都钉住了
    assert len(set(refcoco.SPEC["refer_zip_sha256"].values())) == 3
    assert list(refcoco.SPEC["refer_zip_pins"]) == list(NAMES) == list(refcoco.SPEC["datasets"])
    # 与 P0 的中央目录清单一致（zip_list_*.txt 里的成员数，含 refcocog 的目录条目）
    p0 = REPO / "experiments/P0/0001_env_probe/raw/p6/A"
    for n in NAMES:
        lines = [ln for ln in (p0 / f"zip_list_{n}.txt").read_text().splitlines()[1:] if ln.strip()]
        assert refcoco.SPEC["refer_zip_pins"][n]["n_entries"] == len(lines)


def test_refcoco_train_reads_refer_zips_through_shared_code(server, tmp_path, monkeypatch):
    """训练源的 refer 成员走同一份 refcoco.fetch_refer_members；本地 zip 时清单同样记下完整 sha256；script 仍是 land_train.py。"""
    from tests.test_data_train_land import refcoco_train_upstream

    spec = refcoco_train_upstream(server, tmp_path)
    spec.pop("_images")
    spec["refer_zip_sha256"] = dict.fromkeys(NAMES)  # 这里走"没钉住：只记 note"的路径（钉住的路径见 test_data_train_land）
    refer_dir = tmp_path / "refer"
    refer_dir.mkdir()
    for n in NAMES:
        (refer_dir / f"{n}.zip").write_bytes(server.files[f"/refer/{n}.zip"])
    calls = []
    real = refcoco.fetch_refer_members
    monkeypatch.setattr(refcoco, "fetch_refer_members", lambda *a, **k: calls.append(k["local"]) or real(*a, **k))
    ctx = make_ctx(tmp_path, limit=1)
    ctx.options.update(refer_zip_dir=refer_dir)
    refcoco_train.land(ctx, spec)
    assert calls == [refer_dir / f"{n}.zip" for n in NAMES]
    man = read_manifest(ctx.root / "refcoco_train")
    ok_all(man)
    assert man["script"]["path"] == refcoco_train.SCRIPT == "scripts/data/land_train.py"
    refer_up = [u for u in man["upstream"] if u.get("dataset")]
    assert [u["zip_sha256"] for u in refer_up] == [sha(server.files[f"/refer/{n}.zip"]) for n in NAMES]
    assert not [x for x in man["verification"] if "整个文件" in x["what"]]
    assert sum("还没钉住" in s for s in man["notes"]) == 3


@pytest.mark.parametrize("which", ["e1", "train"])
def test_real_full_sha256_pin_rejects_other_zip(which, server, tmp_path):
    """真实钉住的完整 sha256（D57）：尾部与成员都换成 fixture 的、能过，只有整个文件的 sha256 用 SPEC 里的真实值 →
    fixture zip 必然不符，拒绝落地（E1 与训练源都一样）。"""
    real = refcoco.SPEC["refer_zip_sha256"]
    assert all(isinstance(real[n], str) for n in NAMES)
    if which == "e1":
        spec, zips, _ = build_refer(server, tmp_path)
        refer_dir = local_copies(tmp_path, zips)
        land, src_name = refcoco.land, "refcoco"
    else:
        from tests.test_data_train_land import refcoco_train_upstream

        spec = refcoco_train_upstream(server, tmp_path)
        spec.pop("_images")
        refer_dir = tmp_path / "refer"
        refer_dir.mkdir()
        for n in NAMES:
            (refer_dir / f"{n}.zip").write_bytes(server.files[f"/refer/{n}.zip"])
        land, src_name = refcoco_train.land, "refcoco_train"
    spec["refer_zip_sha256"] = real
    ctx = make_ctx(tmp_path / "land", options={"refer_zip_dir": refer_dir})
    with pytest.raises(zipfile.BadZipFile, match=f"refcoco.zip：整个文件的 sha256 [0-9a-f]{{64}} ≠ 钉住的 {real['refcoco']}"):
        land(ctx, spec)
    src = ctx.root / src_name
    assert not (src / "MANIFEST.json").exists() and not list(src.glob("*.json")) and not (src / "images").exists()
    assert not list(ctx.tmpdir.rglob("*.p")) and not list(ctx.tmpdir.rglob("*instances.json"))  # 成员都还没取
    # 完整性工具同样拒绝
    tool = _load("scripts/data/refer_zip_sha256.py")
    tool_spec = {"refer_zip_pins": spec["refer_zip_pins"], "refer_zip_sha256": real}
    assert tool.main([str(refer_dir)], spec=tool_spec) == 1


# ---------------------------------------------------------------------------
# 验收：land_e1.py refcoco --refer-zip-dir … --limit 3（fixture 的小 zip；CLI 全路径）
# ---------------------------------------------------------------------------


def test_cli_land_e1_refcoco_refer_zip_dir_limit3(server, tmp_path, monkeypatch):
    spec, zips, val_imgs = build_refer(server, tmp_path)
    refer_dir = local_copies(tmp_path, zips)
    for name in NAMES:
        del server.files[f"/wb/{name}.zip"]
    for k in ("datasets", "refer_zip_pins", "instances_pins", "refer_zip_sha256", "image_url"):
        monkeypatch.setitem(refcoco.SPEC, k, spec[k])
    monkeypatch.setitem(refcoco.SPEC, "hf_base", "http://127.0.0.1:9")  # SPEC 的 HF 根地址不通：镜像比对必须走 --mirror-hf-base
    cli = _load("scripts/data/land_e1.py")
    a = cli.parse_args(["refcoco", "--root", "/tmp/x"])
    assert a.refer_zip_dir is None and a.mirror_hf_base is None
    root = tmp_path / "root"
    argv = ["refcoco", "--root", str(root), "--tmpdir", str(tmp_path / "tmp"), "--refer-zip-dir", str(refer_dir), "--limit", "3", "--mirror-hf-base", server.base, "--retries", "1"]
    assert cli.main(argv) == 0
    assert wb_gets(server) == []
    man = read_manifest(root / "refcoco")
    ok_all(man)
    want = sorted({i for n in NAMES for i in val_imgs[n][:3]})
    assert sorted(k for k in man["files"] if k.startswith("images/")) == [f"images/{refcoco.image_file_name(i)}" for i in want]
    assert sorted(k for k in man["files"] if not k.startswith("images/")) == ["refcoco+_unc.json", "refcoco_unc.json", "refcocog_umd.json"]
    assert [u["access"] for u in man["upstream"]] == ["local"] * 3 and man["script"]["path"] == "scripts/data/land_e1.py"
    hf = [x for x in man["verification"] if x["what"].startswith("HF ")]
    assert len(hf) == 3 + 3 + 2 and all(x["url"].startswith(server.base) for x in hf)
    # 同一批 fixture zip 配真实的钉住值：拒绝落地（不是真 zip）
    monkeypatch.undo()
    with pytest.raises(zipfile.BadZipFile, match="钉住"):
        cli.main(["refcoco", "--root", str(tmp_path / "root2"), "--tmpdir", str(tmp_path / "tmp"), "--refer-zip-dir", str(refer_dir), "--limit", "3"])
    assert not (tmp_path / "root2" / "refcoco" / "MANIFEST.json").exists()


# ---------------------------------------------------------------------------
# scripts/data/refer_zip_sha256.py
# ---------------------------------------------------------------------------


def test_refer_zip_sha256_tool(server, tmp_path, capsys):
    spec, zips, _ = build_refer(server, tmp_path)
    d = local_copies(tmp_path, zips)
    tool = _load("scripts/data/refer_zip_sha256.py")
    assert tool.main([str(d)], spec=spec) == 0
    out = capsys.readouterr().out
    assert json.loads(out.strip().splitlines()[-1]) == {f"{n}.zip": {"bytes": len(zips[n]), "sha256": sha(zips[n])} for n in NAMES}
    assert out.count("[相符]") == 3 and "= 钉住的值" in out
    # 整个文件的 sha256 还没钉住：照样通过，提示交管理员补钉
    assert tool.main([str(d)], spec={**spec, "refer_zip_sha256": dict.fromkeys(NAMES)}) == 0
    assert "未钉住" in capsys.readouterr().out
    # 成员数据坏了（尾部不变）：CRC32 / 解压查出来
    (d / "refcocog.zip").write_bytes(corrupt_member(zips["refcocog"], "refcocog/instances.json"))
    assert tool.main([str(d)], spec={**spec, "refer_zip_sha256": dict.fromkeys(NAMES)}) == 1
    out = capsys.readouterr().out
    assert "[不符] refcocog.zip" in out and ("CRC32" in out or "解压成员失败" in out)
    # 缺文件
    (d / "refcoco+.zip").unlink()
    assert tool.main([str(d)], spec=spec) == 1
    assert "[缺失]" in capsys.readouterr().out
    # fixture 配真实的钉住值：字节数与中央目录都不符
    assert tool.main([str(d)]) == 1
    assert "中央目录与钉住的不符" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# download_many：磁盘红线处叫停（D49-5 遗留 1）
# ---------------------------------------------------------------------------

PIECE = 1000


class _Resp:
    """假的 HTTP 响应：正文分块返回。gate 不为 None 时，第二块要等 gate 被 set（即叫停）之后才回来。"""

    def __init__(self, body: bytes, gate: threading.Event | None, inflight: threading.Event, finish_after: threading.Event | None) -> None:
        self.body, self.gate, self.inflight, self.finish_after, self.pos = body, gate, inflight, finish_after, 0
        self.status = 200
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n: int = -1) -> bytes:
        if self.pos >= len(self.body):
            return b""
        if self.pos and self.gate is not None:
            self.inflight.set()
            self.gate.wait(5)
        if self.pos + PIECE >= len(self.body) and self.finish_after is not None:
            self.finish_after.wait(5)  # 先完成的几个等到确实有下载在进行中，才交出最后一块
        piece = self.body[self.pos : self.pos + PIECE]
        self.pos += len(piece)
        return piece


def test_download_many_stops_writing_after_disk_red_line(tmp_path, monkeypatch):
    n_jobs, workers, before = 30, 4, 3
    cancel, inflight = threading.Event(), threading.Event()
    opened: list[int] = []
    lock = threading.Lock()

    def fake_open(url, **kw):
        i = int(url.rsplit("/", 1)[1])
        with lock:
            opened.append(i)
        early = i < before
        return _Resp(bytes([i]) * (3 * PIECE), None if early else cancel, inflight, inflight if early else None)

    monkeypatch.setattr(net, "_open", fake_open)
    src = tmp_path / "src"
    m = Manifest(src, "x", root=tmp_path, tmpdir=tmp_path, min_free_gb=3, log=lambda s: None)
    n_done = [0]
    monkeypatch.setattr(manifest, "disk_free", lambda p: (100 if n_done[0] < before else 1) * GB)  # 第 3 个文件落下后跌破红线

    def on_done(job, res):  # 与 common.download_images 一样：登记清单、查红线
        m.add_file(job.dest.name, sha256=res.sha256, nbytes=res.bytes)
        n_done[0] += 1
        m.df(f"已完成 {n_done[0]}")

    jobs = [net.DownloadJob(f"http://fake/{i}", src / f"{i}.bin") for i in range(n_jobs)]
    t0 = time.monotonic()
    with pytest.raises(DiskRedLine):
        net.download_many(jobs, workers=workers, retries=0, backoff=0.0, on_done=on_done, cancel=cancel)
    assert time.monotonic() - t0 < 4  # 进行中的下载是被叫停的，不是等到 gate 超时
    assert cancel.is_set() and n_done[0] == before  # 红线之后不再调用 on_done
    assert sorted(p.name for p in src.glob("*.bin")) == [f"{i}.bin" for i in range(before)]  # 红线之后没有新文件写出
    parts = list(src.glob("*.part"))
    # 进行中的停在下一块：.part 不再增长（刚开始的可能还是空的），留着供续传
    assert any(p.stat().st_size == PIECE for p in parts) and all(p.stat().st_size <= PIECE for p in parts)
    assert len(opened) <= before + workers - 1  # 没开始的下载不再联网
    assert len((src / manifest.JOURNAL_NAME).read_text().splitlines()) == before


def test_download_many_external_cancel_raises_instead_of_silent_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(net, "_open", lambda url, **kw: pytest.fail("叫停之后不应联网"))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(net.Cancelled):
        net.download_many([net.DownloadJob("http://fake/0", tmp_path / "0.bin")], workers=1, cancel=cancel)


# ---------------------------------------------------------------------------
# Manifest 的 script 参数（D49-5 遗留 2）
# ---------------------------------------------------------------------------


def test_manifest_script_param_keeps_format(tmp_path):
    b200 = json.loads(B200_MANIFEST.read_text())
    Manifest(tmp_path / "a", "a", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None).write()
    Manifest(tmp_path / "b", "b", root=tmp_path, tmpdir=tmp_path, min_free_gb=0, log=lambda s: None, script="scripts/data/land_train.py").write()
    a, b = read_manifest(tmp_path / "a"), read_manifest(tmp_path / "b")
    assert DEFAULT_SCRIPT == "scripts/data/land_e1.py" == a["script"]["path"] == b200["script"]["path"]
    assert b["script"]["path"] == "scripts/data/land_train.py"
    for man in (a, b):  # 已有清单的格式不变：顶层键与 script 的键、顺序都与 B200 落地的那份相同
        assert list(man) == list(b200) and list(man["script"]) == list(b200["script"]) and list(man["script"]["git"]) == list(b200["script"]["git"])
    ctx = make_ctx(tmp_path)
    assert ctx.manifest("z").data["script"]["path"] == DEFAULT_SCRIPT
    assert ctx.manifest("z", script="scripts/data/land_train.py").data["script"]["path"] == "scripts/data/land_train.py"


@pytest.mark.parametrize("mod", [ade20k, nyu_depth_train, nyu_normals], ids=lambda m: m.SOURCE)
def test_train_sources_pass_script_param(mod, tmp_path, monkeypatch):
    """D57-4（DEV-0029）：ade20k、nyu_depth_train、nyu_normals 把落地入口作为 ctx.manifest 的 script 参数传入，不再事后改写
    m.data["script"]["path"]；清单内容与原来的做法（默认 script 建清单、再改写 path）逐项相同。在预算检查处停下，不联网。"""
    ctx = make_ctx(tmp_path)
    real = ctx.manifest
    calls = []

    def spy(name, **kw):
        calls.append((name, kw))
        return real(name, **kw)

    class _Stop(Exception):
        pass

    got = []

    def gate(c, m, est, detail=None):
        got.append(m)
        raise _Stop

    ctx.manifest = spy
    monkeypatch.setattr(mod, "budget_gate", gate)
    with pytest.raises(_Stop):
        mod.land(ctx)
    assert mod.SCRIPT == "scripts/data/land_train.py" and calls == [(mod.SOURCE, {"script": mod.SCRIPT})]
    m = got[0]
    old = real(mod.SOURCE)  # 原来的做法
    old.data["script"]["path"] = mod.SCRIPT
    old.data["license"] = m.data["license"]
    assert m.data == old.data and list(m.data["script"]) == list(old.data["script"])
    assert m.data["script"]["path"] == "scripts/data/land_train.py"
