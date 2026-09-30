"""RUN-0001 §6 group A: image-level split cross-tabs between refer splitBys (from refs_image_ids.json / refs_ann_ids.json).
usage: python p6_A_refs_crosstab.py <dir>  -> writes <dir>/refs_crosstab_images.json"""
import json, itertools, sys, os
D = sys.argv[1]
d = json.load(open(os.path.join(D, 'refs_image_ids.json')))['per_dataset_split']
a = json.load(open(os.path.join(D, 'refs_ann_ids.json')))['per_dataset']
out = {}
for x, y in [("refcocog(umd)", "refcocog(google)"), ("refcoco(unc)", "refcoco(google)"), ("refcoco(unc)", "refcoco+(unc)")]:
    k = f"{x} vs {y}"
    out[k] = {"image_level_split_crosstab(x_split|y_split)": {f"{sx}|{sy}": len(set(d[x][sx]) & set(d[y][sy])) for sx in d[x] for sy in d[y]},
              "ann_id_sets_equal": set(a[x]) == set(a[y]),
              f"ann_ids_only_in_{x}": len(set(a[x]) - set(a[y])), f"ann_ids_only_in_{y}": len(set(a[y]) - set(a[x]))}
rc = set(itertools.chain.from_iterable(d["refcoco(unc)"].values())); rp = set(itertools.chain.from_iterable(d["refcoco+(unc)"].values()))
out["refcoco_images_not_in_refcoco+"] = sorted(rc - rp); out["refcoco+_images_not_in_refcoco"] = sorted(rp - rc)
out["refcoco(unc)_vs_refcoco+(unc)_same_image_set_per_split"] = {s: set(d["refcoco(unc)"][s]) == set(d["refcoco+(unc)"][s]) for s in d["refcoco(unc)"]}
json.dump(out, open(os.path.join(D, 'refs_crosstab_images.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
