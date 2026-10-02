"""检测开发图的完整 COCO 标注（DEV-0017 任务书 §6；D36-1）：RefCOCO 三个开发子集用到的 COCO train2014 图，在官方
instances_train2014.json 里的全部标注。

为什么要它：E1 的检测没有开发子集，开发核对借 RefCOCO 开发子集的图配 COCO 80 类（src/e1/tasks）。落地的 refcoco 文件只含
被指代的那个标注，不是这些图的全部标注，所以算不出每张图的 TP/FP（D32-2、D33）。补上这份 GT 后，验收里检测也能按判定
结果比（每张图在 IoU 0.5 下的 TP 与 FP）。

- 图：<root>/refcoco/ 的三个文件（refcoco_unc、refcoco+_unc、refcocog_umd）里，开发子集（configs/e1_dev_subsets.json 的
  refcoco_unc_val、refcoco+_unc_val、refcocog_umd_val）的句子所属的图，去重后按 image id 升序。图像文件已经在
  <root>/refcoco/images/ 里，这里不下图。先落地 refcoco 才能跑。
- 标注：官方 annotations_trainval2014.zip 用 Range 只取成员 annotations/instances_train2014.json 到 TMPDIR（压缩后约 107 MB，
  不下 253 MB 整包），只把这些图的 images、annotations（含 iscrowd）与 categories 写进 --root，然后删掉临时文件。

与官方的核对：
1. 远端 zip 的中央目录里该成员的字节数、压缩后字节数与 CRC32 = SPEC 钉住的值（开发机 2026-10-01 用 Range 读中央目录核对，
   只取了约 35 KB）；zipfile 读完成员时再核对一次 CRC32。成员的 sha256 记进清单。
2. 每张开发图都在 instances_train2014 里，宽高与 refcoco 文件里的相同。
3. 每个被指代的标注（开发子集句子所属 ref 的 ann_id）都在 instances_train2014 里，image_id、category_id、bbox 逐位相同。
4. 类别表恰好 80 类；coco_val2017 已落地时，与它的类别表（id 与名字）逐个相同（E1 检测的提示用这张表）。

落地后的布局（<root>/coco_train2014_e1dev/）：
- instances_train2014_e1dev.json：标准 COCO 格式（info、licenses、categories 原样；images、annotations 只含开发图），
  pycocotools 可直接读；
- MANIFEST.json。
--limit N：只取前 N 张开发图（开发机小样本）。
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

from src.data.land import net
from src.data.land.common import LandContext, budget_gate, extract_member, open_remote_zip, write_json
from src.data.loaders import REFCOCO_FILES, refcoco_dev_name
from src.data.subsets import DEV_SUBSETS_PATH, dev_ids, population_digest

SOURCE = "coco_train2014_e1dev"
SUBSET_FILE = "instances_train2014_e1dev.json"
REFCOCO_DATASETS = ("refcoco", "refcoco+", "refcocog")  # 开发子集都在 val 上（src/data/loaders.refcoco_dev_name）
N_COCO_CATEGORIES = 80

SPEC: dict[str, Any] = {
    "ann_zip_url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "zip_bytes": 252_872_794,
    "ann_member": "annotations/instances_train2014.json",
    # 远端 zip 中央目录（开发机 2026-10-01，Range 读取）：
    "member_bytes": 332_556_225,
    "member_compress_bytes": 107_297_879,
    "member_crc32": "3cd50ca9",
    "estimate_bytes": 30_000_000,  # 只写开发图的子集标注（几百张图，远小于此）；成员本身只进 TMPDIR
    "license": {
        "annotations": "CC BY 4.0。COCO Terms of Use 原文：\"The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License.\"",
        "images": "图像不在本源里（用 <root>/refcoco/images/ 的 COCO train2014 图，许可见 refcoco 清单）。https://cocodataset.org/#termsofuse",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}


def dev_images(root: str | Path, dev_config: str | Path = DEV_SUBSETS_PATH) -> dict[str, Any]:
    """RefCOCO 三个开发子集用到的图：{image_ids（去重升序）, sizes: id → (W, H), referred: ann_id → 标注, per_dataset}。

    referred 是开发子集句子所属 ref 的标注（取自落地的 refcoco 文件，即 refer zip 自带的 instances.json），用来与官方
    instances_train2014 逐条核对。"""
    src = Path(root) / "refcoco"
    ids: set[int] = set()
    sizes: dict[int, tuple[int, int]] = {}
    referred: dict[int, dict[str, Any]] = {}
    per: dict[str, Any] = {}
    for ds in REFCOCO_DATASETS:
        path, dev_name = src / REFCOCO_FILES[ds], refcoco_dev_name(ds)
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在：先落地 refcoco（scripts/data/land_e1.py refcoco）")
        doc = json.loads(path.read_text())
        want = dev_ids(dev_name, dev_config)
        found: set[str] = set()
        ds_ids: set[int] = set()
        for r in doc["refs"]:
            if r["split"] != "val":
                continue
            hit = [s for s in r["sentences"] if str(s["sent_id"]) in want]
            if not hit:
                continue
            found.update(str(s["sent_id"]) for s in hit)
            iid = int(r["image_id"])
            ds_ids.add(iid)
            im = doc["images"][str(iid)]
            sizes[iid] = (int(im["width"]), int(im["height"]))
            a = doc["anns"][str(r["ann_id"])]
            referred[int(a["id"])] = {"image_id": int(a["image_id"]), "category_id": int(a["category_id"]),
                                      "bbox": [float(v) for v in a["bbox"]], "dataset": ds}
        missing = sorted(want - found)
        if missing:
            raise KeyError(f"{dev_name}：开发子集里的 {len(missing)} 句在 {path} 里找不到，例如 {missing[:5]}")
        ids |= ds_ids
        per[ds] = {"dev_subset": dev_name, "n_sentences": len(want), "n_images": len(ds_ids)}
    return {"image_ids": sorted(ids), "sizes": sizes, "referred": referred, "per_dataset": per}


def subset_instances(inst: dict[str, Any], image_ids: list[int]) -> dict[str, Any]:
    """instances_train2014 → 只含这些图的 COCO 格式（images 按 id、annotations 按 id 排序；categories 原样）。"""
    keep = set(image_ids)
    return {
        "info": inst.get("info", {}),
        "licenses": inst.get("licenses", []),
        "images": sorted((im for im in inst["images"] if im["id"] in keep), key=lambda im: im["id"]),
        "annotations": sorted((a for a in inst["annotations"] if a["image_id"] in keep), key=lambda a: a["id"]),
        "categories": inst["categories"],
    }


def crosscheck(inst: dict[str, Any], sub: dict[str, Any], dev: dict[str, Any], image_ids: list[int]) -> dict[str, list]:
    """核对 2、3（见模块说明）：缺的图、宽高不符的图、缺的或不符的被指代标注。返回各类问题的列表（都空 = 通过）。"""
    have = {im["id"]: im for im in sub["images"]}
    missing_images = [i for i in image_ids if i not in have]
    size_mismatch = [{"image_id": i, "refcoco": list(dev["sizes"][i]), "coco": [have[i]["width"], have[i]["height"]]}
                     for i in image_ids if i in have and (have[i]["width"], have[i]["height"]) != dev["sizes"][i]]
    anns = {a["id"]: a for a in inst["annotations"]}
    keep = set(image_ids)
    ann_problems = []
    for aid, ref in sorted(dev["referred"].items()):
        if ref["image_id"] not in keep:
            continue
        a = anns.get(aid)
        if a is None:
            ann_problems.append({"ann_id": aid, "problem": "instances_train2014 里没有这个标注"})
            continue
        got = {"image_id": int(a["image_id"]), "category_id": int(a["category_id"]), "bbox": [float(v) for v in a["bbox"]]}
        want = {k: ref[k] for k in ("image_id", "category_id", "bbox")}
        if got != want:
            ann_problems.append({"ann_id": aid, "refcoco": want, "coco": got})
    return {"missing_images": missing_images, "size_mismatch": size_mismatch, "referred_ann_problems": ann_problems}


def _crc32_file(path: Path) -> str:
    """TMPDIR 里上次留下的成员是否完好（续跑时跳过再取一遍）：流式算 CRC32，与中央目录里的值比。"""
    crc = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def land(ctx: LandContext, spec: dict[str, Any] | None = None, *, dev_config: str | Path = DEV_SUBSETS_PATH) -> None:
    spec = {**SPEC, **(spec or {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["license"] = spec["license"]
    dev = dev_images(ctx.root, dev_config)
    image_ids = ctx.take(dev["image_ids"])
    budget_gate(ctx, m, spec["estimate_bytes"], {"annotations_subset": spec["estimate_bytes"],
                                                 "tmpdir_member_bytes": spec["member_bytes"]})

    tmp = ctx.scratch(SOURCE) / "instances_train2014.json"
    zf, rf = open_remote_zip(spec["ann_zip_url"], ctx)
    info = zf.getinfo(spec["ann_member"])
    cd = {"bytes": info.file_size, "compress_bytes": info.compress_size, "crc32": f"{info.CRC:08x}"}
    want_cd = {"bytes": spec["member_bytes"], "compress_bytes": spec["member_compress_bytes"], "crc32": spec["member_crc32"]}
    m.verify("远端 zip 中央目录里成员的字节数、压缩后字节数、CRC32 = SPEC 钉住的值", cd == want_cd, got=cd, expected=want_cd,
             zip_bytes=rf.size, expected_zip_bytes=spec["zip_bytes"])
    if cd != want_cd:
        raise ValueError(f"{spec['ann_member']} 与 SPEC 不符：{cd} ≠ {want_cd}")
    if not (tmp.exists() and tmp.stat().st_size == spec["member_bytes"] and _crc32_file(tmp) == spec["member_crc32"]):
        extract_member(zf, spec["ann_member"], tmp)  # zipfile 读完时核对 CRC32，不符抛 BadZipFile
    sha = net.sha256_file(tmp)
    m.data["upstream"].append({
        "url": spec["ann_zip_url"], "zip_bytes": rf.size, "member": spec["ann_member"], "member_sha256": sha,
        "member_bytes": info.file_size, "member_crc32": cd["crc32"], "range_bytes_fetched": rf.bytes_fetched,
        "official_check": "官方 cocodataset.org 直链；中央目录的字节数与 CRC32 = SPEC，zipfile 读完时再核 CRC32；sha256 记录在此",
    })
    m.df("标注已取到 TMPDIR")
    inst = json.loads(tmp.read_text())
    sub = subset_instances(inst, image_ids)
    problems = crosscheck(inst, sub, dev, image_ids)
    del inst
    m.verify("每张开发图都在 instances_train2014 里", not problems["missing_images"], missing=problems["missing_images"][:20])
    m.verify("开发图的宽高 = refcoco 文件里的宽高", not problems["size_mismatch"], mismatch=problems["size_mismatch"][:20])
    keep = set(image_ids)
    n_ref = sum(1 for r in dev["referred"].values() if r["image_id"] in keep)
    m.verify("被指代的标注（refcoco 文件）在 instances_train2014 里，image_id、category_id、bbox 逐位相同",
             not problems["referred_ann_problems"], n_checked=n_ref, problems_head=problems["referred_ann_problems"][:20])
    cats = sorted((int(c["id"]), c["name"]) for c in sub["categories"])
    m.verify(f"类别表恰好 {N_COCO_CATEGORIES} 类", len(cats) == N_COCO_CATEGORIES, n=len(cats))
    val = ctx.source_dir("coco_val2017") / "instances_val2017_e1sub500.json"
    if val.exists():
        vcats = sorted((int(c["id"]), c["name"]) for c in json.loads(val.read_text())["categories"])
        m.verify("类别表（id 与名字）= coco_val2017 的类别表（E1 检测提示用的那张）", cats == vcats,
                 only_train2014=[c for c in cats if c not in vcats][:10], only_val2017=[c for c in vcats if c not in cats][:10])
    else:
        m.note("coco_val2017 还没有落地，没有核对两份类别表相同")
    write_json(m, src_dir, src_dir / SUBSET_FILE, sub)
    tmp.unlink()

    n_ann: dict[int, int] = {}
    for a in sub["annotations"]:
        n_ann[a["image_id"]] = n_ann.get(a["image_id"], 0) + 1
    m.data["selection"] = {
        "rule": "RefCOCO、RefCOCO+、RefCOCOg 开发子集（configs/e1_dev_subsets.json）的句子所属的 COCO train2014 图，去重后按 "
                "image id 升序；标注取这些图在 instances_train2014 里的全部（含 iscrowd）",
        "per_dataset": dev["per_dataset"],
        "n_union": len(dev["image_ids"]),
        "image_ids": image_ids,
        "limit": ctx.limit,
    }
    m.data["stats"] = {
        "n_images": len(sub["images"]),
        "n_images_without_annotations": sum(1 for i in image_ids if i not in n_ann),
        "n_annotations": len(sub["annotations"]),
        "n_crowd": sum(int(a.get("iscrowd", 0)) for a in sub["annotations"]),
        "n_categories_present": len({a["category_id"] for a in sub["annotations"]}),
        "n_referred_checked": n_ref,
    }
    m.data["dedupe_facts"] = {
        "image_key_scheme": "coco:<image_id>（与 refcoco 源的图同一身份键）",
        "known": "这些图就是 RefCOCO 开发子集的图（只用于开发核对与批处理验收，从报告的数字里剔除）；与 COCO val2017 的交集为 0",
        "subset_ids_sha256": population_digest(str(i) for i in image_ids),
    }
    m.df("完成")
    m.write()


def annotations_path(root: str | Path) -> Path:
    return Path(root) / SOURCE / SUBSET_FILE


__all__ = ["N_COCO_CATEGORIES", "REFCOCO_DATASETS", "SOURCE", "SPEC", "SUBSET_FILE", "annotations_path", "crosscheck",
           "dev_images", "land", "subset_instances"]
