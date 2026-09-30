"""RUN-0001 §6 group A: page datasets-server /rows (JSON, in memory) of a jxu124 RefCOCO mirror split, parse raw_anns,
count iscrowd / segmentation type / polygon parts, and compare ids with the official refs pickles (refs_image_ids.json).
usage: python p6_A_rows_crowd.py <hf_repo> <hf_split> <official_key> <official_split> <refs_image_ids.json> <refs_ann_ids.json> <downloads.tsv> <out.json>"""
import json, sys, time, urllib.request, collections
from huggingface_hub import get_token
repo, split, okey, osplit, idsf, annf, dl, outf = sys.argv[1:9]
tok = get_token(); nbytes = 0; nreq = 0
def g(u):
    global nbytes, nreq
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": f"Bearer {tok}"}), timeout=120) as r:
                b = r.read(); nbytes += len(b); nreq += 1; return json.loads(b)
        except Exception:
            if i == 4: raise
            time.sleep(3 * 2 ** i)
q = urllib.request.quote(repo, safe="")
first = g(f"https://datasets-server.huggingface.co/rows?dataset={q}&config=default&split={split}&offset=0&length=100")
total = first["num_rows_total"]; rows = [r["row"] for r in first["rows"]]; trunc = sum(1 for r in first["rows"] if r.get("truncated_cells"))
for o in range(100, total, 100):
    j = g(f"https://datasets-server.huggingface.co/rows?dataset={q}&config=default&split={split}&offset={o}&length=100")
    rows += [r["row"] for r in j["rows"]]; trunc += sum(1 for r in j["rows"] if r.get("truncated_cells"))
anns = [json.loads(r["raw_anns"]) for r in rows]
seg = collections.Counter("polygon" if isinstance(a["segmentation"], list) else "RLE" for a in anns)
off_img = set(json.load(open(idsf))["per_dataset_split"][okey][osplit]); off_ann = set(json.load(open(annf))["per_dataset"][okey])
out = {"repo": repo, "split": split, "compared_to_official": f"{okey}/{osplit}", "num_rows_total": total, "rows_fetched": len(rows),
       "rows_with_truncated_cells": trunc, "json_bytes_fetched": nbytes, "n_requests": nreq,
       "split_field_values": dict(collections.Counter(r["split"] for r in rows)),
       "iscrowd_hist": dict(collections.Counter(a["iscrowd"] for a in anns)), "segmentation_type_hist": dict(seg),
       "polygon_parts_hist": dict(sorted(collections.Counter(len(a["segmentation"]) for a in anns if isinstance(a["segmentation"], list)).items())),
       "raw_anns_keys": sorted(anns[0].keys()),
       "raw_ann_id_equals_ann_id": sum(1 for r, a in zip(rows, anns) if a["id"] == r["ann_id"]),
       "image_set_equals_official_split": set(r["image_id"] for r in rows) == off_img,
       "n_images": len(set(r["image_id"] for r in rows)), "n_official_images": len(off_img),
       "ann_ids_subset_of_official": set(r["ann_id"] for r in rows) <= off_ann,
       "n_sentences": sum(len(r["sentences"]) for r in rows),
       "image_path_example": rows[0].get("image_path"), "raw_image_info_coco_url_example": json.loads(rows[0]["raw_image_info"]).get("coco_url")}
json.dump(out, open(outf, "w"), indent=1); print(json.dumps(out, indent=1))
with open(dl, "a") as f:
    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\thttps://datasets-server.huggingface.co/rows?dataset={q}&split={split} (x{nreq})\t{nbytes}\t"
            f"iscrowd/polygon check of {repo}/{split}; JSON held in memory only (never written to disk)\tin-memory-only\n")
