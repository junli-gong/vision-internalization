"""NYUv2 深度训练集（BTS 清单）的落地（DEV-0022 任务书 §1；D11-2；data_sources B §1.4）。

D11-2：用 BTS 清单（cleinc/bts 的 train_test_inputs/nyudepthv2_train_files_with_gt.txt，24,231 对），按场景过滤：
基础场景名（去掉末尾的字母后缀，如 bookstore_0001d → bookstore_0001）属于官方 test 场景集合的帧全部去掉。
开发机 2026-10-02 用 P0 存档算出：去掉 1,383 帧（只有 bookstore_0001d–j 这 7 个目录），剩 22,848 对。

来源：
- BTS 清单：cleinc/bts 固定 commit（SPEC 钉住 sha256）。该文件自 2019 年首个 commit 起没改过；P0 存档的副本
  （experiments/P0/…/gh/）与固定 commit 的文件 sha256 相同，所以先用仓库里的副本，没有才从 GitHub 下载。
- 官方 test 场景：215 个，由官方 labeled .mat 的 scenes 与 splits.mat 的 testNdxs 得出（P0，data_sources B §1.3；
  存档 nyu_labeled_scenes_names.json），本模块 TEST_SCENES 钉住，tests 核对它与存档相同。
- RGB 与深度：HF aradhye/nyu_depth_v2 固定 revision 的 nyu_depth_v2.zip（6.86 GB，目录布局与 BTS 的 sync 相同，
  BTS 的 24,231 对全在里面）。用 Range 只取清单里的成员；中央目录（尾部 9.66 MB）的 sha256 钉在 SPEC 里，逐个成员核对
  CRC32；用本地 zip（--local-zip）时另核对整个文件的 sha256 = HF 的 LFS sha256。
- 逐对核对：文件名与 BTS 清单一一对应；深度 PNG 是 16 位灰度、640×480；RGB 是 640×480。
- 深度单位：PNG 的整数值 / 1000 = 米，0 表示无效（BTS pytorch/bts_dataloader.py@5e3406b：nyu 分支
  `depth_gt = depth_gt / 1000.0`）。focal 518.8579（清单第 3 列，全部相同）。
- 国内网络连不上 huggingface.co 时：--hf-base https://hf-mirror.com（同一套 resolve 接口，核对方式不变），或先把
  nyu_depth_v2.zip 整个下到本地再 --local-zip。

落地后的布局（<root>/nyu_depth_train/）：sync/<scene>/rgb_<帧>.jpg、sync/<scene>/sync_depth_<帧>.png（与 BTS 相同）、
pairs_train.jsonl（保留的每一对）、scene_filter.json（D11-2 的过滤前后计数与被去掉的目录）、MANIFEST.json。
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.constants import NYU_EVAL_HW
from src.data.land import net
from src.data.land.common import (
    LandContext,
    MemberJob,
    budget_gate,
    check_zip_pins,
    fetch_zip_members,
    hf_resolve_url,
    hf_tree,
    image_meta,
    read_zip_index,
    write_json,
    write_jsonl,
    zip_source,
)
from src.data.land.manifest import REPO_ROOT

SOURCE = "nyu_depth_train"
SCRIPT = "scripts/data/land_train.py"
SPLIT = "train"

# 官方 test 场景（215 个，按名字排序）。出处见模块说明；sha256("\n".join(TEST_SCENES)) = SPEC["test_scenes_sha256"]。
TEST_SCENES: tuple[str, ...] = (
    "bathroom_0003", "bathroom_0004", "bathroom_0008", "bathroom_0009", "bathroom_0012", "bathroom_0015", "bathroom_0017",
    "bathroom_0018", "bathroom_0020", "bathroom_0021", "bathroom_0022", "bathroom_0025", "bathroom_0026", "bathroom_0029",
    "bathroom_0031", "bathroom_0036", "bathroom_0037", "bathroom_0038", "bathroom_0040", "bathroom_0043", "bathroom_0044",
    "bathroom_0046", "bathroom_0047", "bathroom_0052", "bathroom_0058", "bathroom_0060", "bedroom_0003", "bedroom_0005",
    "bedroom_0006", "bedroom_0007", "bedroom_0011", "bedroom_0013", "bedroom_0018", "bedroom_0022", "bedroom_0024",
    "bedroom_0027", "bedroom_0030", "bedroom_0032", "bedroom_0037", "bedroom_0043", "bedroom_0044", "bedroom_0046",
    "bedroom_0048", "bedroom_0049", "bedroom_0054", "bedroom_0055", "bedroom_0058", "bedroom_0061", "bedroom_0064",
    "bedroom_0068", "bedroom_0070", "bedroom_0073", "bedroom_0075", "bedroom_0077", "bedroom_0083", "bedroom_0084",
    "bedroom_0085", "bedroom_0087", "bedroom_0088", "bedroom_0089", "bedroom_0091", "bedroom_0092", "bedroom_0093",
    "bedroom_0095", "bedroom_0099", "bedroom_0101", "bedroom_0102", "bedroom_0103", "bedroom_0105", "bedroom_0108",
    "bedroom_0109", "bedroom_0110", "bedroom_0111", "bedroom_0112", "bedroom_0114", "bedroom_0115", "bedroom_0117",
    "bedroom_0119", "bedroom_0121", "bedroom_0122", "bedroom_0123", "bedroom_0127", "bedroom_0128", "bedroom_0131",
    "bedroom_0133", "bedroom_0134", "bedroom_0135", "bedroom_0137", "bedroom_0139", "bookstore_0001", "classroom_0001",
    "classroom_0007", "classroom_0008", "classroom_0009", "classroom_0014", "classroom_0015", "classroom_0017",
    "classroom_0023", "classroom_0024", "classroom_0026", "computer_lab_0001", "dining_room_0003", "dining_room_0005",
    "dining_room_0006", "dining_room_0009", "dining_room_0011", "dining_room_0017", "dining_room_0018", "dining_room_0020",
    "dining_room_0021", "dining_room_0022", "dining_room_0025", "dining_room_0026", "dining_room_0030", "dining_room_0032",
    "dining_room_0035", "dining_room_0036", "dining_room_0038", "foyer_0001", "home_office_0001", "home_office_0002",
    "home_office_0003", "home_office_0009", "home_office_0010", "home_office_0012", "home_office_0014", "kitchen_0002",
    "kitchen_0004", "kitchen_0005", "kitchen_0007", "kitchen_0009", "kitchen_0015", "kitchen_0021", "kitchen_0022",
    "kitchen_0023", "kitchen_0024", "kitchen_0025", "kitchen_0026", "kitchen_0027", "kitchen_0030", "kitchen_0032",
    "kitchen_0034", "kitchen_0038", "kitchen_0039", "kitchen_0040", "kitchen_0042", "kitchen_0044", "kitchen_0046",
    "kitchen_0054", "kitchen_0055", "kitchen_0056", "kitchen_0057", "kitchen_0058", "living_room_0002", "living_room_0003",
    "living_room_0008", "living_room_0009", "living_room_0013", "living_room_0014", "living_room_0021", "living_room_0025",
    "living_room_0026", "living_room_0027", "living_room_0028", "living_room_0030", "living_room_0031", "living_room_0034",
    "living_room_0036", "living_room_0041", "living_room_0043", "living_room_0044", "living_room_0045", "living_room_0048",
    "living_room_0049", "living_room_0051", "living_room_0052", "living_room_0053", "living_room_0054", "living_room_0056",
    "living_room_0057", "living_room_0059", "living_room_0060", "living_room_0061", "living_room_0064", "living_room_0066",
    "living_room_0072", "living_room_0075", "living_room_0076", "living_room_0079", "living_room_0080", "living_room_0081",
    "living_room_0084", "living_room_0087", "office_0005", "office_0007", "office_0008", "office_0010", "office_0013",
    "office_0014", "office_0015", "office_0016", "office_0017", "office_0020", "office_0022", "office_0027",
    "office_kitchen_0002", "playroom_0005", "playroom_0007", "playroom_0008", "reception_room_0003", "study_0001",
    "study_0002", "study_0007", "study_room_0001", "study_room_0006",
)  # fmt: skip

_P0_B = "experiments/P0/0001_env_probe/raw/p6/B"

SPEC: dict[str, Any] = {
    "bts": {
        "repo": "cleinc/bts",
        "commit": "5e3406b35d1497b2e55d2dd600524d1f4efacaed",  # master @ 2026-02-24；该文件最后一次改动是首个 commit d1ed7caf（2019-08-21）
        "path": "train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        "url": "https://raw.githubusercontent.com/cleinc/bts/5e3406b35d1497b2e55d2dd600524d1f4efacaed/train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        "sha256": "66e2beb718a58c5c946c169f6e70cc6ae166ddd19f5618266fff52ff7443e0e9",
        "git_blob_sha1": "278b2dd04e145ed2b4063d31bc4f5346164bd09d",
        "bytes": 1_855_580,
        "n_pairs": 24_231,
        "repo_copy": f"{_P0_B}/gh/cleinc_bts_master_train_test_inputs_nyudepthv2_train_files_with_gt.txt",
        "license": "仓库代码 GPL-3.0（GitHub API）；数据本身没有单独的许可说明（data_sources B §1.4）",
    },
    "test_scenes_sha256": "5334a7857d9bb5470bb83a2f3658af8280da3c8aaa18ce142ddac9c93fe8a85c",
    "test_scenes_source": f"{_P0_B}/nyu_labeled_scenes_names.json 的 test_scenes_sorted（官方 labeled .mat 的 scenes × splits.mat 的 testNdxs）",
    "hf": {
        "repo": "aradhye/nyu_depth_v2",
        "rev": "4d7f422d47e78ce7b59d4c635294564cca7fadba",
        "path": "nyu_depth_v2.zip",
        "lfs_sha256": "0eb48e96220217a3a49c30a4dedd925beb3ecac039d5f9ece9f86ca65548f37d",
        "card_license": "mit（卡片只有 license 字段，没有正文；镜像非官方）",
    },
    "hf_base": "https://huggingface.co",
    # 开发机 2026-10-02 用 Range 读中央目录（尾部 9,659,696 B）；成员表与 P0 的 listing 逐条相同
    "zip_pins": {"size": 6_857_931_462, "n_entries": 77_347, "start_dir": 6_848_271_766, "tail_bytes": 9_659_696, "tail_sha256": "2e8b36d1a4f3546fd2786d2b29b6e5798c553698e38d58e80e0087cfae2a836b"},
    "member_prefix": "nyu_depth_v2/sync",
    # D11-2 的结果（开发机用 P0 存档算出；落地时重算比对）
    "expect": {
        "n_pairs": 24_231,
        "n_removed": 1_383,
        "n_kept": 22_848,
        "removed_dirs": {"bookstore_0001d": 242, "bookstore_0001e": 158, "bookstore_0001f": 348, "bookstore_0001g": 189, "bookstore_0001h": 119, "bookstore_0001i": 114, "bookstore_0001j": 213},
    },
    "kept_bytes": 3_961_165_376,  # 22,848 对解压后的字节；压缩后 3,943,015,485 B（Range 实际取的量 ≈ 后者 + 5% 间隙）
    "mean_pair_bytes": 173_370,
    "zip_gap": 64 << 10,  # 合并间隔 ≤ 64 KB 的成员：约 1 万个 Range 请求，多取约 5%
    "depth_scale": 1000.0,
    "focal": "518.8579",
    "hw": NYU_EVAL_HW,  # (H, W)，src/constants.py
    "license": {
        "text": "NYU 官方主页没有任何 license、terms 或 copyright 文字，只要求引用 Silberman et al., ECCV 2012（data_sources B §1.1）。HF 镜像 aradhye/nyu_depth_v2 的卡片写 license: mit（非官方）；BTS 清单所在仓库为 GPL-3.0（代码）。",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}

_LINE_RE = re.compile(r"^/(?P<scene>[^/\s]+)/rgb_(?P<frame>\d+)\.jpg /(?P=scene)/sync_depth_(?P=frame)\.png (?P<focal>[0-9.]+)$")
_SUFFIX_RE = re.compile(r"(?<=\d)[a-z]+$")


def base_scene(scene: str) -> str:
    """原始目录名去掉数字之后的字母后缀：bookstore_0001d → bookstore_0001；nyu_office_0、bathroom_0001 不变。"""
    return _SUFFIX_RE.sub("", scene)


def parse_bts_list(text: str) -> list[dict[str, str]]:
    """BTS 清单的每一行 → {scene, frame, rgb, depth, focal}。格式不对就报错（不猜）。"""
    out = []
    for k, line in enumerate(text.splitlines(), 1):
        mm = _LINE_RE.match(line)
        if not mm:
            raise ValueError(f"BTS 清单第 {k} 行格式不对：{line!r}")
        sc, fr = mm["scene"], mm["frame"]
        out.append({"scene": sc, "frame": fr, "rgb": f"sync/{sc}/rgb_{fr}.jpg", "depth": f"sync/{sc}/sync_depth_{fr}.png", "focal": mm["focal"]})
    return out


def scene_filter(pairs: Sequence[dict[str, str]], test_scenes: Iterable[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """D11-2：去掉基础场景名属于官方 test 场景的帧。返回 (保留的对, 统计)。"""
    test = set(test_scenes)
    kept, removed_dirs = [], {}
    for p in pairs:
        if base_scene(p["scene"]) in test:
            removed_dirs[p["scene"]] = removed_dirs.get(p["scene"], 0) + 1
        else:
            kept.append(p)
    return kept, {
        "n_pairs": len(pairs),
        "n_removed": len(pairs) - len(kept),
        "n_kept": len(kept),
        "removed_dirs": dict(sorted(removed_dirs.items())),
        "removed_base_scenes": sorted({base_scene(s) for s in removed_dirs}),
        "n_dirs_in": len({p["scene"] for p in pairs}),
        "n_dirs_kept": len({p["scene"] for p in kept}),
    }


def png_header(data: bytes) -> dict[str, int]:
    """PNG 的 IHDR：宽、高、位深、颜色类型（0 灰度、2 RGB、3 调色板、4 灰度+α、6 RGBA）。不是 PNG 就报 ValueError。"""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("不是 PNG")
    w, h, bit, ctype = struct.unpack(">IIBB", data[16:26])
    return {"width": w, "height": h, "bit_depth": bit, "color_type": ctype}


def read_png_header(path: str | Path) -> dict[str, int]:
    with open(path, "rb") as f:
        return png_header(f.read(32))


def load_bts_list(ctx: LandContext, m, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """BTS 清单的文本：--bts-list 指定的文件 → 仓库里的 P0 副本 → 固定 commit 的 GitHub 原始文件；sha256 必须 = 钉住的值。"""
    bts = spec["bts"]
    cands: list[tuple[str, Path]] = []
    if ctx.options.get("bts_list"):
        cands.append(("--bts-list", Path(ctx.options["bts_list"])))
    cands.append(("仓库里的 P0 副本", REPO_ROOT / bts["repo_copy"]))
    for what, p in cands:
        if p.is_file() and net.sha256_file(p) == bts["sha256"]:
            rec = {"role": "BTS 训练清单", "url": bts["url"], "repo": bts["repo"], "commit": bts["commit"], "sha256": bts["sha256"], "bytes": p.stat().st_size, "read_from": f"{what}：{p}", "official_check": f"sha256 = 固定 commit {bts['commit'][:10]} 上该文件的 sha256（开发机 2026-10-02 核对）"}
            return p.read_text(), rec
        if p.is_file():
            ctx.log(f"[BTS 清单] {what} {p} 的 sha256 不符，不用")
    dest = ctx.scratch(SOURCE) / "nyudepthv2_train_files_with_gt.txt"
    res = net.download(bts["url"], dest, expected_sha256=bts["sha256"], expected_size=bts["bytes"], retries=ctx.retries, backoff=ctx.backoff, log=ctx.log)
    text = dest.read_text()
    dest.unlink()
    return text, {"role": "BTS 训练清单", "url": bts["url"], "repo": bts["repo"], "commit": bts["commit"], "sha256": res.sha256, "bytes": res.bytes, "read_from": "GitHub 固定 commit", "official_check": "下载后 sha256 = 钉住的值"}


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    import hashlib

    spec = {**SPEC, **(spec or {})}
    test_scenes = tuple(spec.get("test_scenes", TEST_SCENES))
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["script"]["path"] = SCRIPT
    m.data["license"] = {**spec["license"], "bts_list": spec["bts"]["license"], "hf_card": spec["hf"]["card_license"]}

    est = int((spec["kept_bytes"] if ctx.limit is None else spec["mean_pair_bytes"] * ctx.limit) * 1.05) + 10_000_000
    budget_gate(ctx, m, est, {"n_pairs": spec["expect"]["n_kept"] if ctx.limit is None else ctx.limit, "margin": "5%", "tmpdir_peak_bytes": 0})

    ts_sha = hashlib.sha256("\n".join(test_scenes).encode()).hexdigest()
    m.verify("官方 test 场景表的 sha256 = 钉住的值", ts_sha == spec["test_scenes_sha256"], got=ts_sha, expected=spec["test_scenes_sha256"], n=len(test_scenes), source=spec["test_scenes_source"])
    text, bts_rec = load_bts_list(ctx, m, spec)
    m.data["upstream"].append(bts_rec)
    pairs = parse_bts_list(text)
    kept, fstats = scene_filter(pairs, test_scenes)
    exp = spec["expect"]
    got = {k: fstats[k] for k in ("n_pairs", "n_removed", "n_kept", "removed_dirs")}
    ok = got == exp and fstats["n_pairs"] == spec["bts"]["n_pairs"]
    m.verify("D11-2：过滤前后的对数与被去掉的目录 = 开发机用 P0 存档算出的值", ok, got=got, expected=exp)
    if not ok:
        raise ValueError(f"D11-2 过滤结果与钉住的不符：{got}")
    focals = sorted({p["focal"] for p in pairs})
    m.verify("BTS 清单的 focal 全部相同", focals == [spec["focal"]], got=focals)
    write_json(m, src_dir, src_dir / "scene_filter.json", {"rule": "D11-2：基础场景名（去掉末尾字母后缀）属于官方 test 场景集合的帧全部去掉", "test_scenes_sha256": ts_sha, **fstats})

    # 远端（HF Range）或本地 zip
    hf = spec["hf"]
    local = ctx.options.get("local_zip")
    url = hf_resolve_url(hf["repo"], hf["rev"], hf["path"], base=ctx.options.get("hf_base") or spec["hf_base"])
    if not local:  # HF tree API 是补充核对（镜像站可能不提供），拿不到只记 note；主核对是下面的中央目录 sha256
        try:
            tree = hf_tree(hf["repo"], hf["rev"], "", base=ctx.options.get("hf_base") or spec["hf_base"])
            meta = tree.get(hf["path"]) or {}
            m.verify("HF tree：zip 的 LFS sha256 与大小 = 钉住的值", meta.get("sha256") == hf["lfs_sha256"] and meta.get("size") == spec["zip_pins"]["size"], got=meta)
        except (net.FetchError, ValueError) as e:
            m.note(f"HF tree API 取不到（{e}）；只按中央目录 sha256 与成员 CRC32 核对")
    src = zip_source(ctx, url, local)
    index = read_zip_index(src)
    check_zip_pins(m, "aradhye/nyu_depth_v2 nyu_depth_v2.zip", index, spec["zip_pins"])
    up: dict[str, Any] = {"role": "RGB 与深度", "repo": hf["repo"], "revision": hf["rev"], "path": hf["path"], "lfs_sha256": hf["lfs_sha256"], "zip": index.pins()}
    if local:
        full = net.sha256_file(local)
        m.verify("本地 zip 的整文件 sha256 = HF 的 LFS sha256", full == hf["lfs_sha256"], got=full, expected=hf["lfs_sha256"])
        up["local_full_sha256"] = full

    pre = spec["member_prefix"]
    missing = [p for p in kept for rel in (p["rgb"], p["depth"]) if f"{pre}/{rel[len('sync/'):]}" not in index.members]
    m.verify("BTS 清单的每一对（RGB 与深度）都在 zip 里", not missing, n_missing=len(missing), head=missing[:10])
    want = kept if ctx.limit is None else kept[: ctx.limit]
    jobs = []
    for p in want:
        for kind in ("rgb", "depth"):
            rel = p[kind]
            jobs.append(MemberJob(f"{pre}/{rel[len('sync/'):]}", src_dir / rel, {"kind": kind, "pair": f"{p['scene']}/{p['frame']}"}))
    H, W = spec["hw"]
    done = [0]

    def process(job: MemberJob, path: Path, rec: dict[str, Any]) -> dict[str, Any]:
        rel = path.relative_to(src_dir).as_posix()
        if job.meta["kind"] == "depth":
            hdr = read_png_header(path)
            if (hdr["width"], hdr["height"], hdr["bit_depth"], hdr["color_type"]) != (W, H, 16, 0):
                raise ValueError(f"{rel}：深度 PNG 应为 16 位灰度 {W}×{H}，实际 {hdr}")
            return {"kind": "depth", "width": W, "height": H, "png_bit_depth": 16, "depth_scale": spec["depth_scale"]}
        prev = m.known(rel)
        if rec["skipped"] and prev and prev.get("sha256") == rec["sha256"] and "phash" in prev:
            meta = {k: prev[k] for k in ("kind", "width", "height", "format", "phash")}
        else:
            meta = image_meta(path)
        if (meta["width"], meta["height"]) != (W, H):
            raise ValueError(f"{rel}：RGB 应为 {W}×{H}，实际 {meta['width']}×{meta['height']}")
        return meta

    def on_done(job: MemberJob, rec: dict[str, Any]) -> None:
        rel = job.dest.relative_to(src_dir).as_posix()
        extra = {k: v for k, v in rec.items() if k not in ("sha256", "bytes", "skipped")}
        if job.meta["kind"] == "rgb":
            extra["image_key"] = f"nyuraw:{job.meta['pair']}"
        m.add_file(rel, sha256=rec["sha256"], nbytes=rec["bytes"], split=SPLIT, pair=job.meta["pair"], **extra)
        done[0] += 1
        if done[0] % 4000 == 0:
            m.df(f"BTS 训练对（已完成 {done[0]} 个文件）")

    ok, failures = fetch_zip_members(src, index, jobs, workers=ctx.workers, gap=spec["zip_gap"], process=process, on_done=on_done, log=ctx.log)
    for f in failures:
        m.fail(step="BTS 训练对", **f)
        ctx.log(f"[失败] {f['member']}: {f['error']}")
    m.df("BTS 训练对")
    m.data["upstream"].append({**up, **src.record(), "official_check": "HF 固定 revision（或 --local-zip 的本地副本，另核对整文件 = LFS sha256）；中央目录 sha256 = SPEC；逐个成员核对 CRC32 与大小；深度 PNG 16 位灰度 640×480，RGB 640×480"})

    files = m.data["files"]
    rows, incomplete = [], []
    for p in want:
        r, d = files.get(p["rgb"]), files.get(p["depth"])
        if r is None or d is None:
            incomplete.append(f"{p['scene']}/{p['frame']}")
            continue
        rows.append({"id": f"{p['scene']}/{p['frame']}", "scene": p["scene"], "base_scene": base_scene(p["scene"]), "frame": p["frame"], "rgb": p["rgb"], "depth": p["depth"], "focal": float(p["focal"]), "rgb_sha256": r["sha256"], "depth_sha256": d["sha256"]})
    m.verify("落地的每一对 RGB 与深度都齐全（文件名与 BTS 清单一一对应）", not incomplete, n_incomplete=len(incomplete), head=incomplete[:10])
    write_jsonl(m, src_dir, src_dir / "pairs_train.jsonl", rows)
    m.data["selection"] = {
        "rule": "BTS 训练清单经 D11-2 场景过滤（scene_filter.json）",
        "pairs": "全部保留的对" if ctx.limit is None else f"--limit {ctx.limit}：保留的对按清单顺序的前 {ctx.limit} 对",
        "n_pairs_requested": len(want),
    }
    m.data["stats"] = {"d11_2": {k: fstats[k] for k in ("n_pairs", "n_removed", "n_kept", "removed_dirs")}, "n_pairs_landed": len(rows), "depth_units": f"PNG 整数值 / {spec['depth_scale']:g} = 米；0 = 无效（BTS bts_dataloader.py）"}
    m.data["dedupe_facts"] = {
        "image_key_scheme": "nyuraw:<原始目录>/<帧号>（BTS 的 raw 帧，没有 labeled 集的 id，与 NYU labeled/test 只能靠感知哈希比）",
        "split_field": "清单里每个文件记 split=train（去重报告把本源归为训练集 nyu_depth_train/train）",
        "known": "D11-2 已按场景去掉官方 test 场景的帧；官方 train 场景的 raw 帧可能与 labeled train 图（含 E1 的 NYU 开发子集 50 张）是同一帧或近邻帧，去重报告按感知哈希核对",
    }
    m.df("完成")
    m.write()

