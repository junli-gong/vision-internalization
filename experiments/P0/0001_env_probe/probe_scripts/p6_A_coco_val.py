"""RUN-0001 §6 group A: schema/counts of the local COCO instances_val2017.json (read-only; owned by parent)
and URL lists for HEAD probing.
usage: python p6_A_coco_val.py <instances_val2017.json> <out_dir>"""
import sys, os, json, collections, random, hashlib

src, out = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
d = json.loads(raw)


def tname(v):
    if isinstance(v, list):
        return f"list[{tname(v[0]) if v else '?'}]"
    if isinstance(v, dict):
        return "dict{" + ",".join(f"{k}:{tname(x)}" for k, x in list(v.items())[:4]) + "}"
    return type(v).__name__


anns = d["annotations"]
seg_kind = collections.Counter(
    ("polygon" if isinstance(a["segmentation"], list) else
     ("RLE-uncompressed(counts=list)" if isinstance(a["segmentation"].get("counts"), list) else "RLE-compressed(counts=str)"))
    for a in anns)
crowd_vs_seg = collections.Counter((a["iscrowd"], "polygon" if isinstance(a["segmentation"], list) else "RLE") for a in anns)
imgs_with_ann = set(a["image_id"] for a in anns)
per_img = collections.Counter(a["image_id"] for a in anns)
lic_count = collections.Counter(im["license"] for im in d["images"])
lic_names = {l["id"]: (l["name"], l["url"]) for l in d["licenses"]}
summ = {
    "file": src, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    "top_level_keys": list(d.keys()),
    "info": d.get("info"),
    "licenses": d.get("licenses"),
    "image_license_distribution": {f"{k}: {lic_names[k][0]}": v for k, v in sorted(lic_count.items())},
    "n_images": len(d["images"]), "n_annotations": len(anns), "n_categories": len(d["categories"]),
    "n_images_with_annotations": len(imgs_with_ann),
    "n_images_without_annotations": len(d["images"]) - len(imgs_with_ann),
    "ann_per_image_mean_over_all_images": round(len(anns) / len(d["images"]), 3),
    "ann_per_image_max": max(per_img.values()),
    "n_supercategories": len(set(c["supercategory"] for c in d["categories"])),
    "category_id_range": [min(c["id"] for c in d["categories"]), max(c["id"] for c in d["categories"])],
    "image_fields_types": {k: tname(v) for k, v in d["images"][0].items()},
    "annotation_fields_types": {k: tname(v) for k, v in anns[0].items()},
    "category_fields_types": {k: tname(v) for k, v in d["categories"][0].items()},
    "example_image": d["images"][0],
    "example_annotation_truncated": {k: (v if k != "segmentation" else str(v)[:120] + "...") for k, v in anns[0].items()},
    "segmentation_kind_hist": dict(seg_kind),
    "iscrowd_x_segkind_hist": {f"iscrowd={a}|{b}": n for (a, b), n in sorted(crowd_vs_seg.items())},
    "coco_url_host_hist": dict(collections.Counter(im["coco_url"].split("/")[2] for im in d["images"])),
    "flickr_url_host_hist": dict(collections.Counter(im["flickr_url"].split("/")[2] for im in d["images"])),
    "image_wh_mean": [round(sum(im["width"] for im in d["images"]) / len(d["images"]), 1),
                      round(sum(im["height"] for im in d["images"]) / len(d["images"]), 1)],
    "categories": [(c["id"], c["name"], c["supercategory"]) for c in d["categories"]],
}
json.dump(summ, open(os.path.join(out, "coco_val2017_schema.json"), "w"), indent=1)
with open(os.path.join(out, "coco_val2017_coco_urls.txt"), "w") as f:
    for im in d["images"]: f.write(im["coco_url"] + "\n")
rng = random.Random(0)
samp = rng.sample(d["images"], 200)
with open(os.path.join(out, "coco_val2017_flickr_urls_seed0_200.txt"), "w") as f:
    for im in samp: f.write(im["flickr_url"] + "\n")
print(json.dumps({k: v for k, v in summ.items() if k not in ("categories", "licenses", "info")}, indent=1))
