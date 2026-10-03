"""ADE20K（SceneParse150，ADEChallengeData2016）的落地（DEV-0022 任务书 §1；data_sources C §1）。

来源：MIT 官方 https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip（967,382,037 B）。要的是
train 20,210 + val 2,000 对（图像 + 标注）与 objectInfo150.txt、sceneCategories.txt，也就是 zip 里的全部文件，所以
相邻成员合并成约 30 个 32 MB 的 Range 请求（不单独存整包）。官方没有发布校验和：核对中央目录（尾部 5,842,204 B）的
sha256 = SPEC 钉住的值（开发机 2026-10-02），逐个成员核对 CRC32；每张标注是单通道 8 位 PNG、取值不超过类别数、与图像同尺寸。

标签：0 = "other objects"，不参与评测；1 到类别数（src/constants.ADE_NUM_CLASSES）对应 objectInfo150.txt 的 Idx
（data_sources C §1.4）。
许可：图像仅限非商业研究与教育用途（ADE20K Terms of Use）；标注 BSD-3（D28-9 限非商业研究）。
国内网络连不上 data.csail.mit.edu 时：把 ADEChallengeData2016.zip 放到本地，--local-zip 指过去（同样核对）。

落地后的布局（<root>/ade20k/）：images/{training,validation}/ADE_*.jpg、annotations/{training,validation}/ADE_*.png、
objectInfo150.txt、sceneCategories.txt、train.jsonl、val.jsonl、MANIFEST.json。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.constants import ADE_NUM_CLASSES
from src.data.land.common import (
    LandContext,
    MemberJob,
    budget_gate,
    check_zip_pins,
    fetch_zip_members,
    image_meta,
    read_zip_index,
    write_jsonl,
    zip_source,
)

SOURCE = "ade20k"
SCRIPT = "scripts/data/land_train.py"
PREFIX = "ADEChallengeData2016/"
# 我们的划分名 → zip 里的目录名；train 是训练集，val 是评测集（E3a 的 mIoU；CV-Bench 的 ADE20K 633 张也来自 val）
SPLIT_DIRS = {"train": "training", "val": "validation"}
ROLE = {"train": "train", "val": "eval"}
N_CLASSES = ADE_NUM_CLASSES  # 单一来源：src/constants.py

SPEC: dict[str, Any] = {
    "zip_url": "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
    "last_modified": "Fri, 22 Jul 2022 22:32:22 GMT",
    "etag": '"39a91415-5e46c686a7a37"',
    "zip_pins": {"size": 967_382_037, "n_entries": 44_429, "start_dir": 961_539_833, "tail_bytes": 5_842_204, "tail_sha256": "d43af205ed4bc6deb52c32d83a2110735937688ef27f1e551f5442484d21ac04"},
    "expect": {"train": 20_210, "val": 2_000},
    "extra_members": ("objectInfo150.txt", "sceneCategories.txt"),
    "members_bytes": 967_861_125,
    "mean_pair_bytes": 43_600,
    "zip_gap": 1 << 20,
    "license": {
        "images": "ADE20K Terms of Use 原文：\"MIT, CSAIL does not own the copyright of the images. … Researcher shall use the Database only for non-commercial research and educational purposes. … If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions\"（https://ade20k.csail.mit.edu/terms/）",
        "annotations": "\"Software and Annotations — This website, image annotations and the software provided belongs to MIT CSAIL and is licensed under a Creative Commons BSD-3 License Agreement\"；SceneParse150 页面页脚 \"License BSD 3-Clause License\"（https://github.com/CSAILVision/sceneparsing/blob/master/LICENSE）",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}


def split_stems(index, split: str) -> list[str]:
    """zip 里某个划分图像与标注都有的样本名（ADE_train_00000001 …），升序。两边不一致就报错。"""
    d = SPLIT_DIRS[split]
    img = sorted(Path(k).stem for k in index.members if k.startswith(f"{PREFIX}images/{d}/") and k.endswith(".jpg"))
    ann = sorted(Path(k).stem for k in index.members if k.startswith(f"{PREFIX}annotations/{d}/") and k.endswith(".png"))
    if img != ann:
        raise ValueError(f"{split}：图像与标注的样本名不一致")
    return img


def parse_scene_categories(text: str) -> dict[str, str]:
    """sceneCategories.txt：每行 "ADE_train_00000001 airport_terminal"。"""
    out = {}
    for line in text.splitlines():
        if line.strip():
            stem, cat = line.split()
            out[stem] = cat
    return out


def label_meta(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        im.load()
        if im.mode != "L":
            raise ValueError(f"{path.name}：标注应为单通道 8 位（mode L），实际 {im.mode}")
        a = np.asarray(im)
    mx = int(a.max()) if a.size else 0
    if mx > N_CLASSES:
        raise ValueError(f"{path.name}：标签最大值 {mx} > {N_CLASSES}")
    return {"kind": "label", "width": int(a.shape[1]), "height": int(a.shape[0]), "max_label": mx, "frac_ignore0": round(float((a == 0).mean()), 6)}


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE, script=SCRIPT)
    m.data["license"] = spec["license"]
    est = int((spec["members_bytes"] if ctx.limit is None else spec["mean_pair_bytes"] * ctx.limit) * 1.05) + 1_000_000
    budget_gate(ctx, m, est, {"n_pairs": sum(spec["expect"].values()) if ctx.limit is None else ctx.limit, "margin": "5%", "tmpdir_peak_bytes": 0})

    src = zip_source(ctx, spec["zip_url"], ctx.options.get("local_zip"))
    index = read_zip_index(src)
    check_zip_pins(m, "ADEChallengeData2016.zip", index, spec["zip_pins"])
    stems = {sp: split_stems(index, sp) for sp in SPLIT_DIRS}
    got = {sp: len(v) for sp, v in stems.items()}
    m.verify("train / val 的图像与标注一一对应，对数 = 官方（20,210 / 2,000）", got == spec["expect"], got=got, expected=spec["expect"])
    if ctx.limit is None:
        want = [(s, sp) for sp in SPLIT_DIRS for s in stems[sp]]
    else:  # 小样本：前 limit−1 对 train 加 1 对 val
        want = [(s, "train") for s in stems["train"][: max(ctx.limit - 1, 1)]] + ([(stems["val"][0], "val")] if ctx.limit >= 2 else [])

    jobs = [MemberJob(PREFIX + name, src_dir / name, {"part": "meta"}) for name in spec["extra_members"]]
    for stem, sp in want:
        d = SPLIT_DIRS[sp]
        jobs.append(MemberJob(f"{PREFIX}images/{d}/{stem}.jpg", src_dir / f"images/{d}/{stem}.jpg", {"part": "image", "split": sp, "stem": stem}))
        jobs.append(MemberJob(f"{PREFIX}annotations/{d}/{stem}.png", src_dir / f"annotations/{d}/{stem}.png", {"part": "label", "split": sp, "stem": stem}))
    done = [0]

    def process(job: MemberJob, path: Path, rec: dict[str, Any]) -> dict[str, Any]:
        part = job.meta["part"]
        if part == "meta":
            return {"kind": "metadata"}
        if part == "label":
            return label_meta(path)
        rel = path.relative_to(src_dir).as_posix()
        prev = m.known(rel)
        if rec["skipped"] and prev and prev.get("sha256") == rec["sha256"] and "phash" in prev:
            return {k: prev[k] for k in ("kind", "width", "height", "format", "phash")}
        return image_meta(path)

    def on_done(job: MemberJob, rec: dict[str, Any]) -> None:
        rel = job.dest.relative_to(src_dir).as_posix()
        extra = {k: v for k, v in rec.items() if k not in ("sha256", "bytes", "skipped")}
        if job.meta["part"] != "meta":
            extra["split"] = job.meta["split"]
        if job.meta["part"] == "image":
            extra["image_key"] = f"ade20k:{job.meta['stem']}"
        m.add_file(rel, sha256=rec["sha256"], nbytes=rec["bytes"], **extra)
        done[0] += 1
        if done[0] % 5000 == 0:
            m.df(f"ADE20K（已完成 {done[0]} 个文件）")

    ok, failures = fetch_zip_members(src, index, jobs, workers=ctx.workers, gap=spec["zip_gap"], process=process, on_done=on_done, log=ctx.log)
    for f in failures:
        m.fail(step="ADE20K", **f)
        ctx.log(f"[失败] {f['member']}: {f['error']}")
    m.df("ADE20K")
    m.data["upstream"].append({"role": "图像与标注", "last_modified": spec["last_modified"], "etag": spec["etag"], **src.record(), "zip": index.pins(), "official_check": "MIT 官方直链（或 --local-zip 的本地副本）；官方没有校验和：中央目录 sha256 = SPEC，逐个成员 CRC32，标注 mode L 且取值 ≤ 类别数、与图像同尺寸"})

    files = m.data["files"]
    scenes = parse_scene_categories((src_dir / "sceneCategories.txt").read_text()) if (src_dir / "sceneCategories.txt").exists() else {}
    bad_size, rows = [], {sp: [] for sp in SPLIT_DIRS}
    for stem, sp in want:
        d = SPLIT_DIRS[sp]
        im, lb = files.get(f"images/{d}/{stem}.jpg"), files.get(f"annotations/{d}/{stem}.png")
        if im is None or lb is None:
            continue
        if (im["width"], im["height"]) != (lb["width"], lb["height"]):
            bad_size.append(stem)
        rows[sp].append({"stem": stem, "split": sp, "image": f"images/{d}/{stem}.jpg", "label": f"annotations/{d}/{stem}.png", "width": im["width"], "height": im["height"], "scene": scenes.get(stem), "image_sha256": im["sha256"], "label_sha256": lb["sha256"]})
    m.verify("每对图像与标注同尺寸", not bad_size, n_bad=len(bad_size), head=bad_size[:10])
    for sp in SPLIT_DIRS:
        write_jsonl(m, src_dir, src_dir / f"{sp}.jsonl", rows[sp])
    m.data["selection"] = {"rule": "ADEChallengeData2016 的 train 与 val 全部", "landed": "全部" if ctx.limit is None else [f"{sp}:{s}" for s, sp in want]}
    m.data["stats"] = {"n_train": len(rows["train"]), "n_val": len(rows["val"]), "label_encoding": f"0 = other objects（不参与评测），1–{N_CLASSES} = objectInfo150.txt 的 Idx"}
    m.data["dedupe_facts"] = {
        "image_key_scheme": "ade20k:<样本名>（与 E1 cvbench 的 ade20k:ADE_val_… 同一套键）",
        "split_field": "清单里每张图记 split=train|val；train 是训练集，val 是评测集",
        "known": "CV-Bench 含 ADE20K val 的 633 张（data_sources C §2.3），与本源的 val 同键；train 与 CV-Bench 无同键",
    }
    m.df("完成")
    m.write()
