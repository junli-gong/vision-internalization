"""COCO val2017 开放词汇检测子集的落地（DEV-0012 任务书 §1.2；D11-5、D20；data_sources A §2）。

- 标注：从官方 annotations_trainval2017.zip 用 Range 只取成员 annotations/instances_val2017.json 到 TMPDIR
  （不下 252 MB 整包），核对 sha256 与 P0 记录一致，只把子集标注写进 --root，然后删掉临时文件。
- 抽样：先剔除 D20 的两张探针用图，再在其余 4,998 张（含 48 张无标注图）里按哈希排序取 500 张。
- 图像：按 images[].coco_url 逐张下载（P0 实测 5,000 张全部 200）。

落地后的布局（<root>/coco_val2017/）：
- instances_val2017_e1sub500.json：标准 COCO 格式（info、licenses、categories 原样；images、annotations 只含子集），
  pycocotools 可直接读。
- images/<file_name>、MANIFEST.json。
"""

from __future__ import annotations

import json
from typing import Any

from src.data.constants import COCO_OVD_SUBSET_SIZE, PROBE_IMAGE_KEYS
from src.data.land import net
from src.data.land.common import LandContext, budget_gate, download_images, extract_member, open_remote_zip, write_json
from src.data.subsets import population_digest, seeded_sample

SOURCE = "coco_val2017"
SUBSET_FILE = "instances_val2017_e1sub500.json"
SAMPLE_PURPOSE = "coco_val2017_ovd500"

SPEC: dict[str, Any] = {
    "ann_zip_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "ann_member": "annotations/instances_val2017.json",
    "ann_sha256": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",  # P0，data_sources A §2
    "ann_bytes": 19_987_840,
    "mean_image_bytes": 162_941,  # P0 对 5,000 张 HEAD 的均值
    "license": {
        "annotations": "CC BY 4.0。COCO Terms of Use 原文：\"The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License.\"",
        "images": "\"The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use. The users of the images accept full responsibility for the use of the dataset…\" 逐图 Flickr 许可见 images[].license（val2017 中 NC 类占 69.5%）。https://cocodataset.org/#termsofuse",
        "project_terms": "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像",
    },
}


def select_subset(image_ids: list[int], k: int = COCO_OVD_SUBSET_SIZE) -> tuple[list[int], list[int]]:
    """返回 (子集 image id 升序, 被剔除的探针图 id)。先剔除再抽样（D20）。"""
    probe = {int(key.split(":")[1]) for key in PROBE_IMAGE_KEYS if key.startswith("coco:")}
    excluded = sorted(i for i in image_ids if i in probe)
    pool = [str(i) for i in image_ids if i not in probe]
    chosen = [int(s) for s in seeded_sample(pool, k, SAMPLE_PURPOSE)]
    return chosen, excluded


def subset_coco(coco: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    keep = set(ids)
    return {
        "info": coco["info"],
        "licenses": coco["licenses"],
        "images": sorted((im for im in coco["images"] if im["id"] in keep), key=lambda im: im["id"]),
        "annotations": sorted((a for a in coco["annotations"] if a["image_id"] in keep), key=lambda a: a["id"]),
        "categories": coco["categories"],
    }


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["license"] = spec["license"]
    est_images = spec["mean_image_bytes"] * (ctx.limit or COCO_OVD_SUBSET_SIZE)
    est = int(est_images * 1.25) + 5_000_000
    budget_gate(ctx, m, est, {"images": est_images, "annotations_subset": 5_000_000, "margin": "25%"})

    tmp = ctx.scratch(SOURCE) / "instances_val2017.json"
    zf, rf = open_remote_zip(spec["ann_zip_url"], ctx)
    info = zf.getinfo(spec["ann_member"])
    if not (tmp.exists() and net.sha256_file(tmp) == spec["ann_sha256"]):
        extract_member(zf, spec["ann_member"], tmp)
    sha = net.sha256_file(tmp)
    ok = sha == spec["ann_sha256"] and tmp.stat().st_size == spec["ann_bytes"]
    m.verify("instances_val2017.json 的 sha256 与字节数 = P0 记录", ok, got=sha, expected=spec["ann_sha256"], bytes=tmp.stat().st_size)
    if not ok:
        raise ValueError("instances_val2017.json 与 P0 记录不符")
    m.data["upstream"].append(
        {
            "url": spec["ann_zip_url"],
            "zip_bytes": rf.size,
            "member": spec["ann_member"],
            "member_sha256": sha,
            "member_bytes": info.file_size,
            "member_crc32": f"{info.CRC:08x}",
            "range_bytes_fetched": rf.bytes_fetched,
            "official_check": "官方 cocodataset.org 直链；zip CRC32 + sha256 = P0 记录",
        }
    )
    m.df("标注已取到 TMPDIR")
    coco = json.loads(tmp.read_text())
    all_ids = sorted(im["id"] for im in coco["images"])
    chosen, excluded = select_subset(all_ids)
    sub = subset_coco(coco, chosen)
    write_json(m, src_dir, src_dir / SUBSET_FILE, sub)
    tmp.unlink()

    land_ids = chosen if ctx.limit is None else chosen[: ctx.limit]
    by_id = {im["id"]: im for im in sub["images"]}
    jobs = [(by_id[i]["coco_url"], f"images/{by_id[i]['file_name']}", {"image_key": f"coco:{i}"}) for i in land_ids]
    download_images(ctx, m, src_dir, jobs, "val2017 图像")

    n_ann = {}
    for a in sub["annotations"]:
        n_ann[a["image_id"]] = n_ann.get(a["image_id"], 0) + 1
    m.data["selection"] = {
        "rule": f"剔除 D20 探针用图后，在其余 {len(all_ids) - len(excluded)} 张里按哈希排序取 {COCO_OVD_SUBSET_SIZE} 张（purpose={SAMPLE_PURPOSE}，src/data/subsets.py）",
        "probe_excluded_before_sampling": excluded,
        "n_population": len(all_ids) - len(excluded),
        "image_ids": chosen,
        "landed_image_ids": land_ids if ctx.limit is not None else "全部",
    }
    m.data["stats"] = {
        "n_images": len(chosen),
        "n_images_without_annotations": sum(1 for i in chosen if i not in n_ann),
        "n_annotations": len(sub["annotations"]),
        "n_crowd": sum(a["iscrowd"] for a in sub["annotations"]),
        "n_categories_present": len({a["category_id"] for a in sub["annotations"]}),
    }
    m.data["dedupe_facts"] = {
        "image_key_scheme": "coco:<image_id>",
        "known": "与 RefCOCO 系列（COCO train2014）图像交集为 0（data_sources §3）；CV-Bench 含 val2017 的 805 张（与本子集的交集见 cvbench 清单）",
        "subset_ids_sha256": population_digest(str(i) for i in chosen),
    }
    m.write()
