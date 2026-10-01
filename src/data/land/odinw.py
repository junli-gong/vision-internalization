"""ODinW13 子集的落地（DEV-0012 任务书 §1.3；data_sources C §4）。

- 13 个数据集与各自的评测划分：GLIP `configs/odinw_13/*.yaml` 的 DATASETS.TEST（11 个用 val，EgoHands 与
  VehiclesOpenImages 用 minival），标注文件与图像目录取该划分 REGISTER 里的 ann_file / img_dir。下表从 P0 存档的
  13 个 yaml（experiments/P0/0001_env_probe/raw/p6/C/odinw_glip_configs13/）抄来，tests/test_data_land.py 逐项钉住。
- 数据在 HF `GLIPModel/GLIP`（model 仓库）`odinw_35/<zip>.zip`，revision 钉 P0 记录的 commit。用 Range 只读 zip 的
  中央目录、评测划分的标注 json、README 与抽中的图像成员，不下整包（13 个 zip 共 2.25 GB）。
- 抽样：每个数据集在评测划分里按哈希排序取 ≤50 张（不足 50 张全取）。
- 与官方的核对：zip 大小 = HF tree 记的 LFS 大小；每个成员读完核对 CRC32；HF 上的 LFS sha256 记进清单
  （没有下整包，所以不能对整包算 sha256——这一点写在清单的 official_check 里）。
- 许可：逐数据集记 README.dataset.txt 原文（pistols 的 zip 里没有 README，记标注 json 的 licenses 与 info）。

落地后的布局（<root>/odinw13/<config 名>/）：annotations_e1sub.json（COCO 格式子集，另加 glip 字段：评测划分、
OVERRIDE_CATEGORY）、images/<file_name>；<root>/odinw13/MANIFEST.json。
"""

from __future__ import annotations

import json
import zipfile
from typing import Any

from src.data.constants import ODINW_SUBSET_MAX
from src.data.land import net
from src.data.land.common import (
    LandContext,
    budget_gate,
    hf_tree,
    image_meta,
    open_remote_zip,
    write_json,
)
from src.data.subsets import seeded_sample

SOURCE = "odinw13"
GLIP_REPO = "GLIPModel/GLIP"
GLIP_REV = "66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1"

# config 名 → (zip 名, GLIP TEST 划分, 标注成员, 图像目录成员, OVERRIDE_CATEGORY 的名字按 id 顺序)
# 成员路径 = GLIP yaml 里的路径去掉开头的 "odinw/"。
CONFIGS: dict[str, dict[str, Any]] = {
    "AerialMaritimeDrone_large": {
        "zip": "AerialMaritimeDrone",
        "test": "val",
        "ann": "AerialMaritimeDrone/large/valid/annotations_without_background.json",
        "img_dir": "AerialMaritimeDrone/large/valid",
        "override": ["boat", "car", "dock", "jetski", "lift"],
    },
    "Aquarium_Aquarium_Combined.v2-raw-1024.coco": {
        "zip": "Aquarium",
        "test": "val",
        "ann": "Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json",
        "img_dir": "Aquarium/Aquarium Combined.v2-raw-1024.coco/valid",
        "override": ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"],
    },
    "CottontailRabbits": {
        "zip": "CottontailRabbits",
        "test": "val",
        "ann": "CottontailRabbits/valid/annotations_without_background.json",
        "img_dir": "CottontailRabbits/valid",
        "override": ["Cottontail-Rabbit"],
    },
    "EgoHands_generic": {
        "zip": "EgoHands",
        "test": "minival",
        "ann": "EgoHands/generic/mini_val/annotations_without_background.json",
        "img_dir": "EgoHands/generic/mini_val",
        "override": ["hand"],
    },
    "NorthAmericaMushrooms_North_American_Mushrooms.v1-416x416.coco": {
        "zip": "NorthAmericaMushrooms",
        "test": "val",
        "ann": "NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/annotations_without_background.json",
        "img_dir": "NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid",
        "override": ["CoW", "chanterelle"],
    },
    "Packages_Raw": {
        "zip": "Packages",
        "test": "val",
        "ann": "Packages/Raw/valid/annotations_without_background.json",
        "img_dir": "Packages/Raw/valid",
        "override": ["package"],
    },
    "PascalVOC": {
        "zip": "PascalVOC",
        "test": "val",
        "ann": "PascalVOC/valid/annotations_without_background.json",
        "img_dir": "PascalVOC/valid",
        "override": [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ],
    },
    "Raccoon_Raccoon.v2-raw.coco": {
        "zip": "Raccoon",
        "test": "val",
        "ann": "Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json",
        "img_dir": "Raccoon/Raccoon.v2-raw.coco/valid",
        "override": ["raccoon"],
    },
    "ShellfishOpenImages_raw": {
        "zip": "ShellfishOpenImages",
        "test": "val",
        "ann": "ShellfishOpenImages/raw/valid/annotations_without_background.json",
        "img_dir": "ShellfishOpenImages/raw/valid",
        "override": ["Crab", "Lobster", "Shrimp"],
    },
    "VehiclesOpenImages_416x416": {
        "zip": "VehiclesOpenImages",
        "test": "minival",
        "ann": "VehiclesOpenImages/416x416/mini_val/annotations_without_background.json",
        "img_dir": "VehiclesOpenImages/416x416/mini_val",
        "override": ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"],
    },
    "pistols_export": {
        "zip": "pistols",
        "test": "val",
        "ann": "pistols/export/val_annotations_without_background.json",
        "img_dir": "pistols/export",
        "override": ["pistol"],
    },
    "pothole": {
        "zip": "pothole",
        "test": "val",
        "ann": "pothole/valid/annotations_without_background.json",
        "img_dir": "pothole/valid",
        "override": ["pothole"],
    },
    "thermalDogsAndPeople": {
        "zip": "thermalDogsAndPeople",
        "test": "val",
        "ann": "thermalDogsAndPeople/valid/annotations_without_background.json",
        "img_dir": "thermalDogsAndPeople/valid",
        "override": ["dog", "person"],
    },
}

SPEC: dict[str, Any] = {
    "repo": GLIP_REPO,
    "rev": GLIP_REV,
    "kind": "model",
    "hf_base": "https://huggingface.co",
    "configs": CONFIGS,
    "license_note": "逐数据集不同（data_sources C §4.3：5 个 Public Domain、5 个 CC BY 4.0、1 个 MIT、1 个 ODbL、1 个 undefined）；原文见各数据集的 readme 字段。项目条款 D28-9：限非商业研究，不再分发。",
}


def _readme_members(zf: zipfile.ZipFile, cfg: dict[str, Any]) -> list[str]:
    """README.dataset.txt 在数据集子目录（评测划分目录的上一级，pistols 是图像目录本身）。"""
    names = set(zf.namelist())
    base = cfg["img_dir"].rsplit("/", 1)[0] if cfg["img_dir"].count("/") >= 1 else cfg["img_dir"]
    out = []
    for cand in (f"{base}/README.dataset.txt", f"{cfg['img_dir']}/README.dataset.txt"):
        if cand in names and cand not in out:
            out.append(cand)
    return out


def subset_annotations(ann: dict[str, Any], image_ids: list[int], cfg: dict[str, Any], name: str) -> dict[str, Any]:
    keep = set(image_ids)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    if [c["id"] for c in cats] != list(range(1, len(cfg["override"]) + 1)):
        raise ValueError(f"{name}：类别 id {[c['id'] for c in cats]} 与 OVERRIDE_CATEGORY 的个数 {len(cfg['override'])} 对不上")
    return {
        "info": ann.get("info"),
        "licenses": ann.get("licenses"),
        "categories": cats,
        "images": sorted((im for im in ann["images"] if im["id"] in keep), key=lambda im: im["id"]),
        "annotations": sorted((a for a in ann["annotations"] if a["image_id"] in keep), key=lambda a: a["id"]),
        "glip": {
            "config": name,
            "test_split": cfg["test"],
            "ann_member": cfg["ann"],
            "img_dir_member": cfg["img_dir"],
            "override_category_names": cfg["override"],
        },
    }


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    configs = spec["configs"]
    which = ctx.options.get("odinw_configs") or list(configs)
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    tree = hf_tree(spec["repo"], spec["rev"], "odinw_35", kind=spec["kind"], base=spec["hf_base"])
    from src.data.land.common import hf_resolve_url

    # 预算：ODinW 图像多为 416×416 或 1024 级 JPEG；P0 的 zip 目录显示每张 15–100 KB。估 13×50 张 × 200 KB，偏保守。
    n_per = min(ODINW_SUBSET_MAX, ctx.limit or ODINW_SUBSET_MAX)
    est = len(which) * n_per * 200_000 + 5_000_000
    budget_gate(ctx, m, est, {"assumed_bytes_per_image": 200_000, "images_per_dataset": n_per})

    licenses: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for name in which:
        cfg = configs[name]
        zpath = f"odinw_35/{cfg['zip']}.zip"
        meta = tree[zpath]
        url = hf_resolve_url(spec["repo"], spec["rev"], zpath, kind=spec["kind"], base=spec["hf_base"])
        zf, rf = open_remote_zip(url, ctx, block_size=256 << 10)
        if rf.size != meta["size"]:
            raise net.FetchError(url, f"zip 大小 {rf.size} 与 HF 记录 {meta['size']} 不符")
        ann = json.loads(zf.read(cfg["ann"]))
        all_ids = sorted(im["id"] for im in ann["images"])
        chosen = [int(s) for s in seeded_sample([str(i) for i in all_ids], ODINW_SUBSET_MAX, f"odinw13:{name}")]
        sub = subset_annotations(ann, chosen, cfg, name)
        ds_dir = src_dir / name
        write_json(m, src_dir, ds_dir / "annotations_e1sub.json", sub)
        readmes = {r: zf.read(r).decode("utf-8", "replace") for r in _readme_members(zf, cfg)}
        licenses[name] = {"readme": readmes, "annotation_licenses": ann.get("licenses"), "annotation_info": ann.get("info")}

        land_ids = chosen if ctx.limit is None else chosen[: ctx.limit]
        by_id = {im["id"]: im for im in sub["images"]}
        n_ok = 0
        for i in land_ids:
            fn = by_id[i]["file_name"]
            member = f"{cfg['img_dir']}/{fn}"
            rel = f"{name}/images/{fn}"
            dest = src_dir / rel
            info = zf.getinfo(member)
            prev = m.known(rel)
            if dest.exists() and prev and net.sha256_file(dest) == prev["sha256"]:
                m.add_file(rel, sha256=prev["sha256"], nbytes=prev["bytes"], **{k: prev[k] for k in ("kind", "width", "height", "format", "phash", "image_key", "zip_member", "crc32") if k in prev})
                n_ok += 1
                continue
            try:
                data = zf.read(member)  # 读完核对 CRC32
            except (zipfile.BadZipFile, net.FetchError) as e:
                m.fail(step=name, member=member, error=str(e))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(data)
            part.replace(dest)
            im_meta = image_meta(dest)
            if (im_meta["width"], im_meta["height"]) != (by_id[i]["width"], by_id[i]["height"]):
                m.verify(f"{name}/{fn} 宽高与标注一致", False, file=(im_meta["width"], im_meta["height"]), ann=(by_id[i]["width"], by_id[i]["height"]))
            m.add_file(rel, sha256=net.sha256_file(dest), nbytes=len(data), image_key=f"odinw:{name}:{fn}", zip_member=member, crc32=f"{info.CRC:08x}", **im_meta)
            n_ok += 1
        m.data["upstream"].append(
            {
                "config": name,
                "url": url,
                "repo": spec["repo"],
                "revision": spec["rev"],
                "zip_bytes": rf.size,
                "zip_lfs_sha256": meta["sha256"],
                "range_bytes_fetched": rf.bytes_fetched,
                "official_check": "GLIP 官方 download.py 指向的 HF 仓库（/resolve/ 直链，revision 钉死）；zip 大小 = LFS 记录；成员 CRC32；未下整包故未对整包算 sha256",
            }
        )
        selection[name] = {"test_split": cfg["test"], "n_population": len(all_ids), "image_ids": chosen, "landed": len(land_ids)}
        stats[name] = {
            "n_images": len(chosen),
            "n_annotations": len(sub["annotations"]),
            "n_categories": len(sub["categories"]),
            "n_images_landed": n_ok,
        }
        m.df(f"{name} 完成")
    m.data["license"] = {"note": spec["license_note"], "per_dataset": licenses}
    m.data["selection"] = {"rule": f"每个数据集在 GLIP TEST 划分里按哈希排序取 ≤{ODINW_SUBSET_MAX} 张（purpose=odinw13:<config>）", "per_dataset": selection}
    m.data["stats"] = stats
    m.data["dedupe_facts"] = {"image_key_scheme": "odinw:<config>:<file_name>", "known": "无 COCO/NYU/ADE 的 id 可对；跨集只能靠感知哈希（PascalVOC 与 COCO 都是 Flickr 图）"}
    m.write()
