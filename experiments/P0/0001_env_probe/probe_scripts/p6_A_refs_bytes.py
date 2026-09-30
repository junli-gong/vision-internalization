"""RUN-0001 §6 group A: bytes of COCO train2014 images used by RefCOCO/+/g, per dataset / split / union,
from HEAD Content-Length (head_train2014_union.tsv) and refs_image_ids.json.
usage: python p6_A_refs_bytes.py <dir>  -> <dir>/refcoco_images/refs_image_bytes.json"""
import json, sys, os, itertools, csv
D = sys.argv[1]
ids = json.load(open(os.path.join(D, "refs_image_ids.json")))
size = {}
for r in csv.DictReader(open(os.path.join(D, "refcoco_images", "head_train2014_union_compact.tsv")), delimiter="\t"):
    iid = int(r["image_id"]); size[iid] = int(r["content_length"]) if r["status"] == "200" else None
def tot(s):
    s = set(s); miss = [i for i in s if size.get(i) is None]
    return {"n_images": len(s), "bytes": sum(size[i] for i in s if size.get(i) is not None), "n_missing": len(miss)}
out = {"per_dataset_split": {}, "per_dataset_all_splits": {}, "per_family": {}, "eval_splits_only": {}}
for k, sp in ids["per_dataset_split"].items():
    out["per_dataset_split"][k] = {s: tot(v) for s, v in sp.items()}
    out["per_dataset_all_splits"][k] = tot(itertools.chain.from_iterable(sp.values()))
P = ids["per_dataset_split"]
fam = {"refcoco": set(itertools.chain.from_iterable(P["refcoco(unc)"].values())),
       "refcoco+": set(itertools.chain.from_iterable(P["refcoco+(unc)"].values())),
       "refcocog": set(itertools.chain.from_iterable(P["refcocog(umd)"].values()))}
out["per_family"] = {k: tot(v) for k, v in fam.items()}
out["union_all"] = tot(ids["union_sorted"])
ev = set(P["refcoco(unc)"]["val"] + P["refcoco(unc)"]["testA"] + P["refcoco(unc)"]["testB"]
         + P["refcoco+(unc)"]["val"] + P["refcoco+(unc)"]["testA"] + P["refcoco+(unc)"]["testB"]
         + P["refcocog(umd)"]["val"] + P["refcocog(umd)"]["test"])
out["eval_splits_only"]["union of refcoco(unc) val/testA/testB + refcoco+(unc) val/testA/testB + refcocog(umd) val/test"] = tot(ev)
tr = set(P["refcoco(unc)"]["train"] + P["refcoco+(unc)"]["train"] + P["refcocog(umd)"]["train"])
out["train_splits_only"] = {"union of refcoco(unc)/refcoco+(unc)/refcocog(umd) train": tot(tr),
                            "train-union minus eval-union images": tot(tr - ev)}
json.dump(out, open(os.path.join(D, "refcoco_images", "refs_image_bytes.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
