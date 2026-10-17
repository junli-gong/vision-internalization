"""DEV-0047：训练加载器的"文件存在"检查改成按父目录列一次（src/data/train_loaders._DirListing）。

- 与旧实现（下面 _old_have 是 DEV-0047 之前的 _have 原样）在 strict 与非 strict 下逐条相同：refcoco（三个数据集）、NYU 深度、
  nyu_normals、ADE20K 的加载器与 E3 的 load_pool；落地树上造缺文件、断链、好链接、清单里有但磁盘上没有、磁盘上有但清单里没有、
  多余文件、整个目录不存在、目录位置是普通文件、目录链接、链接环路、目录不可读 / 不可进；strict 时异常类型与信息相同；
- _DirListing.exists 与 Path.exists 逐个相同（含异常）；名字按字节比较（大小写、Unicode 规范化不折叠）；
- stat / exists 次数随目录数增长、不随文件数增长；
- 续跑：运行身份（config.yaml、resume_state 里的 sha256）不含代码 commit，只含训练池的统计与 id；旧实现写下的运行身份，新实现续跑照收。
"""

from __future__ import annotations

import json
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from src.data import train_loaders as tl
from src.data.land.refcoco import SPEC as REFCOCO_SPEC
from src.e3 import data as D
from src.e3.config import load_config

REPO = Path(__file__).resolve().parents[1]
FIX_E2 = REPO / "tests" / "fixtures" / "e2" / "root"
FIX_E3 = REPO / "tests" / "fixtures" / "e3" / "root"
CFG = load_config()
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def _old_have(src, rels, files, strict, what, *_):
    """DEV-0047 之前的 _have（逐个 Path.exists()），原样保留作对照；多出来的参数（_DirListing）不用。"""
    if all(r in files and (src / r).exists() for r in rels):
        return True
    if strict:
        raise FileNotFoundError(f"{what}：{[r for r in rels if not (src / r).exists()]} 不存在；落地不全（strict=False 可跳过并计数）")
    return False


def _case_sensitive(d: Path) -> bool:
    p = d / "CaseProbe"
    p.write_bytes(b"")
    try:
        return not (d / "caseprobe").exists()
    finally:
        p.unlink()


def _norm_sensitive(d: Path) -> bool:
    nfc = unicodedata.normalize("NFC", "café")
    p = d / nfc
    p.write_bytes(b"")
    try:
        return not (d / unicodedata.normalize("NFD", nfc)).exists()
    finally:
        p.unlink()


# ---------------------------------------------------------------------------
# 落地树与缺陷
# ---------------------------------------------------------------------------

# 每个加载器：(名字, 调用 (树, strict) → LoadResult)。e3/ 是 RefCOCO 家族三个数据集，e2/ 是四个源（含一个 refcoco）
LOADERS: dict[str, Callable[[Path, bool], Any]] = {
    "refcoco3/refcoco": lambda t, s: tl.load_refcoco_train(t / "e3", "refcoco", strict=s),
    "refcoco3/refcoco+": lambda t, s: tl.load_refcoco_train(t / "e3", "refcoco+", strict=s),
    "refcoco3/refcocog": lambda t, s: tl.load_refcoco_train(t / "e3", "refcocog", strict=s),
    "refcoco_e2": lambda t, s: tl.load_refcoco_train(t / "e2", "refcoco", strict=s),
    "nyu_depth": lambda t, s: tl.load_nyu_depth_train(t / "e2", strict=s),
    "nyu_normals/train": lambda t, s: tl.load_nyu_normals(t / "e2", "train", strict=s),
    "nyu_normals/test": lambda t, s: tl.load_nyu_normals(t / "e2", "test", strict=s),
    "ade20k/train": lambda t, s: tl.load_ade20k(t / "e2", "train", strict=s),
    "ade20k/val": lambda t, s: tl.load_ade20k(t / "e2", "val", strict=s),
    # E3 的训练池（真实训练走这条路；剔除清单 "auto" = 各自根下的 dedupe/）
    "pool/refseg": lambda t, s: D.load_pool("refseg", D.E3Paths(refseg_root=t / "e3", depth_root=t / "e2", strict=s), CFG.task("refseg")),
    "pool/depth": lambda t, s: D.load_pool("depth", D.E3Paths(refseg_root=t / "e3", depth_root=t / "e2", strict=s), CFG.task("depth")),
}
REFCOCO3 = {"refcoco3/refcoco", "refcoco3/refcoco+", "refcoco3/refcocog", "pool/refseg"}
NYU = {"nyu_depth", "pool/depth"}

# 文件级缺陷落在每个加载器都用到的一个文件上（refcoco 的这张图三个数据集都有指代；NYU 是一对里的第二个文件）
FILE_TARGETS = (
    "e3/refcoco_train/images/COCO_train2014_000000002002.jpg",
    "e2/refcoco_train/images/COCO_train2014_000000001002.jpg",
    "e2/nyu_depth_train/sync/bedroom_0001b/sync_depth_00001.png",
    "e2/nyu_normals/train/img/0004.png",
    "e2/nyu_normals/test/norm/0002.png",
    "e2/ade20k/annotations/training/ADE_train_00000003.png",
    "e2/ade20k/images/validation/ADE_val_00000002.jpg",
)
# 目录级缺陷（refcoco_e2 与 nyu_normals/test 不碰）
DIR_TARGETS = (
    "e3/refcoco_train/images",
    "e2/nyu_depth_train/sync/office_0003",
    "e2/nyu_normals/train/norm",
    "e2/ade20k/images/training",
    "e2/ade20k/annotations/validation",
)
ALL = set(LOADERS)
DIR_AFFECTED = REFCOCO3 | NYU | {"nyu_normals/train", "ade20k/train", "ade20k/val"}


def _tree(tmp: Path) -> Path:
    t = tmp / "tree"
    shutil.copytree(FIX_E2, t / "e2")
    shutil.copytree(FIX_E3, t / "e3")
    (tmp / "elsewhere").mkdir()
    return t


def _drop_from_manifest(t: Path, rel: str) -> None:
    root, source, *rest = rel.split("/")
    mf = t / root / source / "MANIFEST.json"
    doc = json.loads(mf.read_text())
    del doc["files"]["/".join(rest)]
    mf.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


def _apply(defect: str, t: Path, chmods: list[Path]) -> set[str]:
    """在树 t 上造缺陷，返回受影响（非 strict 时 n_missing > 0、strict 时报错）的加载器。chmods：要在结束时恢复权限的目录。"""
    elsewhere = t.parent / "elsewhere"
    if defect == "clean":
        return set()
    if defect == "deleted":  # 清单里有、磁盘上没有
        for r in FILE_TARGETS:
            (t / r).unlink()
        return ALL
    if defect == "broken_symlink":  # 断链（绝对、相对各一半）
        for i, r in enumerate(FILE_TARGETS):
            (t / r).unlink()
            os.symlink(str(elsewhere / "nowhere.bin") if i % 2 else "../nowhere/x.bin", t / r)
        return ALL
    if defect == "good_symlink":  # 好链接（绝对、相对各一半）：存在
        for i, r in enumerate(FILE_TARGETS):
            dst = elsewhere / f"{i}_{Path(r).name}"
            shutil.move(t / r, dst)
            os.symlink(str(dst) if i % 2 else os.path.relpath(dst, (t / r).parent), t / r)
        return set()
    if defect == "not_in_manifest":  # 磁盘上有、清单里没有
        for r in FILE_TARGETS:
            _drop_from_manifest(t, r)
        return ALL
    if defect == "stray_files":  # 磁盘上多出来、谁都不引用的文件与子目录：不影响
        for r in FILE_TARGETS + DIR_TARGETS:
            d = (t / r).parent if (t / r).is_file() else t / r
            (d / "zz_stray.jpg").write_bytes(b"x")
            (d / "zz_stray_dir").mkdir(exist_ok=True)
        return set()
    if defect == "missing_dir":  # 整个目录不存在
        for r in DIR_TARGETS:
            shutil.rmtree(t / r)
        return DIR_AFFECTED
    if defect == "parent_is_file":  # 目录的位置是普通文件（ENOTDIR）
        for r in DIR_TARGETS:
            shutil.rmtree(t / r)
            (t / r).write_bytes(b"not a dir")
        return DIR_AFFECTED
    if defect == "dir_symlink":  # 目录是链接：好链接照常；断链 = 其下全不存在
        for r in ("e2/nyu_depth_train/sync/kitchen_0003a", "e2/ade20k/images/training"):
            dst = elsewhere / Path(r).name
            shutil.move(t / r, dst)
            os.symlink(str(dst), t / r)
        for r in ("e2/nyu_depth_train/sync/office_0003", "e2/nyu_normals/train/norm"):
            shutil.rmtree(t / r)
            os.symlink(str(elsewhere / "no_such_dir"), t / r)
        return NYU | {"nyu_normals/train"}
    if defect == "symlink_loop":  # 目录是自环链接（ELOOP：列不出来，退回 exists）；文件是自环链接
        d = t / "e2/nyu_depth_train/sync/living_room_0004"
        shutil.rmtree(d)
        os.symlink(d.name, d)
        f = t / "e2/ade20k/images/validation/ADE_val_00000001.jpg"
        f.unlink()
        os.symlink(f.name, f)
        return NYU | {"ade20k/val"}
    if defect == "unreadable_dir":  # 可进不可读（0o311）：列不出来，退回逐个 exists，文件照样在
        for r in ("e3/refcoco_train/images", "e2/nyu_depth_train/sync/bedroom_0001a", "e2/ade20k/annotations/training"):
            os.chmod(t / r, 0o311)
            chmods.append(t / r)
        return set()
    if defect == "no_access_dir":  # 不可进（0o000）：exists() 抛 PermissionError（3.12），新旧都抛同一个
        os.chmod(t / "e2/nyu_normals/train/norm", 0o000)
        chmods.append(t / "e2/nyu_normals/train/norm")
        return set() if IS_ROOT else {"nyu_normals/train"}
    if defect == "case":  # 磁盘上的名字大小写不同（只在大小写敏感的盘上比；Linux 上 exists() 判不存在）
        for r, new in (("e2/nyu_depth_train/sync/bedroom_0001a/rgb_00001.jpg", "RGB_00001.jpg"),
                       ("e2/ade20k/images/training/ADE_train_00000001.jpg", "ade_train_00000001.jpg")):
            (t / r).rename((t / r).with_name(new))
        return NYU | {"ade20k/train"}
    raise ValueError(defect)


DEFECTS = ("clean", "deleted", "broken_symlink", "good_symlink", "not_in_manifest", "stray_files", "missing_dir", "parent_is_file",
           "dir_symlink", "symlink_loop", "unreadable_dir", "no_access_dir", "case")


def _outcome(fn: Callable[[], Any]) -> tuple:
    try:
        res = fn()
    except Exception as e:  # noqa: BLE001 — 异常类型与信息也要逐字相同
        return ("err", type(e), str(e))
    return ("ok", res.records, list(res.stats.items()))


def _n_missing(stats: dict[str, Any]) -> int:
    if "datasets" in stats:
        return sum(v["n_missing"] for v in stats["datasets"].values())
    return stats["n_missing"]


@pytest.mark.parametrize("defect", DEFECTS)
def test_loaders_identical_to_old_have(defect, tmp_path, monkeypatch):
    t = _tree(tmp_path)
    if defect == "case" and not _case_sensitive(t):
        pytest.skip("这块盘大小写不敏感（macOS）：exists() 会认大小写不同的名字，新实现按字节比较（同 Linux）")
    chmods: list[Path] = []
    affected = _apply(defect, t, chmods)
    scans = [0]
    real_scandir = os.scandir

    def counting_scandir(*a, **k):
        scans[0] += 1
        return real_scandir(*a, **k)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    try:
        new = {(n, s): _outcome(lambda: f(t, s)) for n, f in LOADERS.items() for s in (True, False)}
        n_scans_new = scans[0]
        with monkeypatch.context() as m:
            m.setattr(tl, "_have", _old_have)
            scans[0] = 0
            old = {(n, s): _outcome(lambda: f(t, s)) for n, f in LOADERS.items() for s in (True, False)}
            assert scans[0] == 0  # 对照确实走的是逐个 exists()
    finally:
        for d in chmods:
            os.chmod(d, 0o755)
    assert n_scans_new > 0
    for key in LOADERS:
        for s in (True, False):
            assert new[(key, s)] == old[(key, s)], (defect, key, s)
    # 缺陷确实起了作用（不是空比较）
    for key in LOADERS:
        strict, loose = new[(key, True)], new[(key, False)]
        if key in affected:
            assert strict[0] == "err", (defect, key)
            assert loose[0] == "err" or _n_missing(dict(loose[2])) > 0, (defect, key)
        else:
            assert strict[0] == loose[0] == "ok" and _n_missing(dict(loose[2])) == 0, (defect, key)
    if defect == "not_in_manifest":  # 文件都在、只是清单里没有：报错信息列出的"不存在"是空表（旧行为）
        assert new[("ade20k/val", True)][1:] == (FileNotFoundError, "ade20k ADE_val_00000002：[] 不存在；落地不全（strict=False 可跳过并计数）")
    if defect == "deleted":
        assert new[("nyu_depth", True)][1:] == (
            FileNotFoundError, "nyu pair bedroom_0001b/00001：['sync/bedroom_0001b/sync_depth_00001.png'] 不存在；落地不全（strict=False 可跳过并计数）")


# ---------------------------------------------------------------------------
# _DirListing.exists 与 Path.exists 逐个相同
# ---------------------------------------------------------------------------


def _probe(fn: Callable[[], bool]) -> tuple:
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("err", type(e), str(e))


def test_dir_listing_matches_path_exists(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "f.jpg").write_bytes(b"x")
    (d / "sub" / "g.png").write_bytes(b"x")
    os.symlink("f.jpg", d / "link_ok")
    os.symlink("nope", d / "link_broken")
    os.symlink("sub", d / "link_dir")
    os.symlink("link_dir_broken_target", d / "link_dir_broken")
    os.symlink("loop", d / "loop")
    (d / "afile").write_bytes(b"x")
    (d / "x_only").mkdir()
    (d / "x_only" / "h.png").write_bytes(b"x")
    (d / "none").mkdir()
    (d / "none" / "i.png").write_bytes(b"x")
    rels = ["f.jpg", "nope.jpg", "sub", "sub/g.png", "sub/zz.png", "link_ok", "link_broken", "link_dir/g.png", "link_dir/zz.png",
            "link_dir_broken/g.png", "loop", "loop/g.png", "afile/x.png", "missing_dir/x.png", "sub/../f.jpg", "sub/..", "sub/.",
            "x_only/h.png", "x_only/zz.png", "none/i.png", "f.jpg/", "nul\0.png", "sub/nul\0/g.png"]
    os.chmod(d / "x_only", 0o311)
    os.chmod(d / "none", 0o000)
    try:
        disk = tl._DirListing()
        got = {r: _probe(lambda: disk.exists(d / r)) for r in rels}
        want = {r: _probe(lambda: (d / r).exists()) for r in rels}
        if not IS_ROOT:
            assert disk._dirs[d / "x_only"] is None and disk._dirs[d / "none"] is None  # 列不出来：退回逐个 exists()
    finally:
        os.chmod(d / "x_only", 0o755)
        os.chmod(d / "none", 0o755)
    assert got == want
    assert disk._dirs[d / "missing_dir"] == {} and disk._dirs[d / "afile"] == {}  # 目录不存在 / 不是目录：其下全不存在
    assert want["link_broken"] == ("ok", False) and want["link_ok"] == ("ok", True)


def test_dir_listing_compares_names_by_bytes(tmp_path):
    """大小写与 Unicode 规范化不折叠（同 Linux 上的 exists()）；在大小写 / 规范化敏感的盘上与 exists() 相同。"""
    nfc = unicodedata.normalize("NFC", "café.jpg")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    (tmp_path / "Img.JPG").write_bytes(b"x")
    (tmp_path / nfc).write_bytes(b"x")
    disk = tl._DirListing()
    assert disk.exists(tmp_path / "Img.JPG") and disk.exists(tmp_path / nfc)
    assert not disk.exists(tmp_path / "img.jpg") and not disk.exists(tmp_path / nfd)
    assert disk.exists(tmp_path / "IMG.JPG") is False
    if _case_sensitive(tmp_path):
        assert (tmp_path / "img.jpg").exists() is False
    if _norm_sensitive(tmp_path):
        assert (tmp_path / nfd).exists() is False


# ---------------------------------------------------------------------------
# 元数据查询次数：随目录数增长、不随文件数增长
# ---------------------------------------------------------------------------


def _synth_nyu(root: Path, n_dirs: int, per_dir: int) -> Path:
    src = root / "nyu_depth_train"
    files, lines = {}, []
    for i in range(n_dirs):
        scene = f"zz_synth_{i:04d}"
        (src / "sync" / scene).mkdir(parents=True)
        for j in range(per_dir):
            fr = f"{j:05d}"
            rgb, dep = f"sync/{scene}/rgb_{fr}.jpg", f"sync/{scene}/sync_depth_{fr}.png"
            for rel in (rgb, dep):
                (src / rel).write_bytes(b"")
                files[rel] = {"path": rel, "sha256": "0" * 64, "bytes": 0}
            lines.append(json.dumps({"id": f"{scene}/{fr}", "scene": scene, "base_scene": scene, "frame": fr, "rgb": rgb,
                                     "depth": dep, "focal": 518.8579}))
    (src / "pairs_train.jsonl").write_text("\n".join(lines) + "\n")
    (src / "MANIFEST.json").write_text(json.dumps({"schema": 1, "source": "nyu_depth_train", "files": files}))
    return root


def _synth_refcoco(root: Path, n_refs: int, n_images: int) -> Path:
    src = root / "refcoco_train"
    (src / "images").mkdir(parents=True)
    files, images, anns, refs = {}, {}, {}, []
    for i in range(n_images):
        name = f"COCO_train2014_{i:012d}.jpg"
        (src / "images" / name).write_bytes(b"")
        files[f"images/{name}"] = {"path": f"images/{name}", "sha256": "0" * 64, "bytes": 0}
        images[str(i)] = {"id": i, "file_name": name, "height": 10, "width": 10}
    for k in range(n_refs):
        anns[str(k)] = {"segmentation": [[0, 0, 5, 0, 5, 5]], "bbox": [0, 0, 5, 5], "iscrowd": 0}
        refs.append({"ref_id": k, "ann_id": k, "image_id": k % n_images, "category_id": 1, "split": "train",
                     "sentences": [{"sent_id": k, "sent": f"thing {k}", "raw": f"thing {k}"}]})
    split_by = REFCOCO_SPEC["datasets"]["refcoco"]["split_by"]
    (src / f"refcoco_{split_by}_train.json").write_text(json.dumps({"refs": refs, "anns": anns, "images": images,
                                                                   "categories": {"1": "thing"}}))
    (src / "MANIFEST.json").write_text(json.dumps({"schema": 1, "source": "refcoco_train", "files": files}))
    return root


def _count_calls(monkeypatch, fn: Callable[[], Any], *, old: bool) -> dict[str, int]:
    n = {"stat": 0, "lstat": 0, "scandir": 0, "listdir": 0}
    with monkeypatch.context() as m:
        for name in n:
            real = getattr(os, name)

            def wrap(*a, _real=real, _name=name, **k):
                n[_name] += 1
                return _real(*a, **k)

            m.setattr(os, name, wrap)
        if old:
            m.setattr(tl, "_have", _old_have)
        fn()
    return n


def test_metadata_calls_scale_with_dirs_not_files(tmp_path, monkeypatch):
    def nyu(n_dirs, per_dir, old=False):
        root = _synth_nyu(tmp_path / f"nyu_{n_dirs}_{per_dir}_{old}", n_dirs, per_dir)
        return _count_calls(monkeypatch, lambda: tl.load_nyu_depth_train(root, exclusions=None, require_dedupe=False), old=old)

    small, many_files, many_dirs = nyu(3, 4), nyu(3, 40), nyu(6, 4)
    assert many_files == small  # 文件多 10 倍：一次都不多
    assert small["scandir"] == 3 and many_dirs["scandir"] == 6  # 每个场景目录列一次
    assert many_dirs["stat"] == small["stat"] and many_dirs["lstat"] == small["lstat"] == 0
    o_small, o_many = nyu(3, 4, old=True), nyu(3, 40, old=True)
    assert o_small["scandir"] == o_many["scandir"] == 0
    assert o_many["stat"] - o_small["stat"] == 2 * 3 * 36  # 旧实现：每对两次 stat
    assert o_small["stat"] - small["stat"] == 2 * 3 * 4

    def ref(n_refs, old=False):
        root = _synth_refcoco(tmp_path / f"ref_{n_refs}_{old}", n_refs, 7)
        return _count_calls(monkeypatch, lambda: tl.load_refcoco_train(root, "refcoco", exclusions=None, require_dedupe=False), old=old)

    r10, r200 = ref(10), ref(200)
    assert r10 == r200 and r10["scandir"] == 1  # 同一张图被重复查：一个目录只列一次
    assert ref(200, old=True)["stat"] - ref(10, old=True)["stat"] == 190  # 旧实现：每条指代一次 stat


# ---------------------------------------------------------------------------
# 续跑：运行身份不含代码 commit；旧实现写下的运行身份，新实现续跑照收
# ---------------------------------------------------------------------------


def test_e3paths_as_dict_unchanged():
    assert D.E3Paths.single_root("/data/root").as_dict() == {
        "refseg_root": "/data/root", "depth_root": "/data/root", "exclusions": "auto", "require_dedupe": True, "strict": True}


def test_resume_across_have_implementations(tmp_path, monkeypatch):
    """旧实现跑 2 步停下（写 config.yaml 与 resume_state），新实现 --resume 接着跑完：运行身份相同、续跑被接受、config.yaml 不变。
    对照：训练池少一条（同一棵树、_have 丢掉一条）时，续跑被拒（运行身份覆盖训练池，任何记录的变化都会被拦下）。"""
    import torch  # noqa: F401 — 训练依赖

    from src.e3.config import resolve_run
    from src.e3.train import RESUME_STATE, leaf_dir, run_training
    from tests.test_e3_train import _spec, _tiny

    key, sup = "qwen3.5-0.8b", "grid"
    rc = resolve_run(CFG, supervision=sup, tiny=True)
    paths = D.E3Paths.fixtures()
    out = tmp_path / "run"
    leaf = leaf_dir(out, rc["variant"], key, 0)
    with monkeypatch.context() as m:
        m.setattr(tl, "_have", _old_have)
        ad = _tiny(key)
        assert run_training(ad, CFG, _spec(key, sup, ad), rc, paths, out, stop_after_steps=2, log=lambda s: None) is None
    cfg_old = (leaf / "config.yaml").read_bytes()
    meta_old = (leaf / "meta.json").read_bytes()
    assert (leaf / "ckpt" / "latest" / RESUME_STATE).exists()
    # 运行身份（config.yaml）里没有代码版本：git commit 只记在 meta.json（新开时写一次），续跑不核对它
    assert sorted(yaml.safe_load(cfg_old)) == ["c3", "config", "data", "experiment", "framework", "grid_size", "image_policy",
                                               "instructions", "model", "noise", "seed", "stop_id", "supervision", "tokens",
                                               "training", "variant"]
    assert "git" in json.loads(meta_old)

    # 对照：训练池变了 → 拒绝
    real_have = tl._have
    dropped = []

    def drop_one(src, rels, files, strict, what, disk):
        if not dropped and "nyu pair" in what:
            dropped.append(what)
            return False
        return real_have(src, rels, files, strict, what, disk)

    with monkeypatch.context() as m:
        m.setattr(tl, "_have", drop_one)
        ad = _tiny(key)
        with pytest.raises(ValueError, match="续跑的运行身份与已有的不同"):
            run_training(ad, CFG, _spec(key, sup, ad), rc, paths, out, resume=True, log=lambda s: None)
    assert dropped

    # 新实现续跑：接受，跑完
    ad = _tiny(key)
    s = run_training(ad, CFG, _spec(key, sup, ad), rc, paths, out, resume=True, log=lambda s: None)
    assert s["complete"] and s["steps"] == rc["steps"]
    assert (leaf / "config.yaml").read_bytes() == cfg_old and (leaf / "meta.json").read_bytes() == meta_old
