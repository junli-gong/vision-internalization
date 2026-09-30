"""RUN-0001 §6 group A: indirect check of whether RefCOCO-family ann_ids are crowd (RLE) or polygon annotations,
using the local COCO instances_val2017.json (2017 reuses 2014 annotation ids; val2017 images come from train2014+val2014).
(1) id-range of crowd vs non-crowd annotations in val2017; (2) refs ann_ids that also appear in val2017 -> their iscrowd/seg type;
(3) image overlap refs images vs val2017 images.
usage: python p6_A_refs_crowd.py <instances_val2017.json> <refs_ann_ids.json> <refs_image_ids.json> <out.json>"""
import json, sys, collections
v = json.load(open(sys.argv[1]))
A = json.load(open(sys.argv[2])); I = json.load(open(sys.argv[3]))
crowd = [a["id"] for a in v["annotations"] if a["iscrowd"] == 1]; nonc = [a["id"] for a in v["annotations"] if a["iscrowd"] == 0]
byid = {a["id"]: a for a in v["annotations"]}
out = {"val2017_crowd_ann_id_range": [min(crowd), max(crowd)], "val2017_noncrowd_ann_id_range": [min(nonc), max(nonc)]}
thr = min(crowd)
for k, ids in list(A["per_dataset"].items()) + [("union", A["union_sorted"])]:
    inval = [i for i in ids if i in byid]
    out[k] = {"n_ann_ids": len(ids), "n_ann_id_ge_min_crowd_id": sum(1 for i in ids if i >= thr), "max_ann_id": max(ids),
              "n_in_val2017": len(inval),
              "in_val2017_iscrowd_hist": dict(collections.Counter(byid[i]["iscrowd"] for i in inval)),
              "in_val2017_segtype_hist": dict(collections.Counter("polygon" if isinstance(byid[i]["segmentation"], list) else "RLE" for i in inval)),
              "in_val2017_polygon_parts_hist": dict(sorted(collections.Counter(len(byid[i]["segmentation"]) for i in inval if isinstance(byid[i]["segmentation"], list)).items()))}
vimg = set(im["id"] for im in v["images"])
out["refs_union_images_also_in_val2017"] = len(set(I["union_sorted"]) & vimg)
out["refs_eval_images_also_in_val2017"] = {k: {s: len(set(ids) & vimg) for s, ids in sp.items()} for k, sp in I["per_dataset_split"].items()}
json.dump(out, open(sys.argv[4], "w"), indent=1); print(json.dumps(out, indent=1))
