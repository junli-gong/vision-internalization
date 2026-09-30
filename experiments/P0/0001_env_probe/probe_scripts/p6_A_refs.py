"""RUN-0001 §6 group A: RefCOCO/RefCOCO+/RefCOCOg official refs(*).p stats, read IN MEMORY via HTTP Range
from the Wayback Machine copies of the refer zips (official host bvisionweb1.cs.unc.edu does not resolve).
Nothing is written to disk except the stats/ids JSON under <out_dir>.
Pickles are loaded with a restricted Unpickler that refuses every global (refs are plain lists/dicts).
usage: python p6_A_refs.py <out_dir> <downloads.tsv>"""
import sys, io, os, json, time, zipfile, pickle, hashlib, urllib.request, collections, itertools

OUT, DL = sys.argv[1], sys.argv[2]
SRC = {  # dataset -> (wayback timestamp, [splitBy...])  from lichengunc/refer issue #14 comment 1258318183
    "refcoco": ("20220413011718", ["unc", "google"]),
    "refcoco+": ("20220413011656", ["unc"]),
    "refcocog": ("20220413012904", ["umd", "google"]),
}
URL = "https://web.archive.org/web/{ts}if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/{ds}.zip"


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.fetched, self.nreq = url, 0, 0, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"]); self.accept = r.headers.get("Accept-Ranges")
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        for i in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    assert r.status == 206, r.status
                    data = r.read()
                break
            except Exception:
                if i == 4: raise
                time.sleep(2 ** i * 2)
        self.pos += len(data); self.fetched += len(data); self.nreq += 1; return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global {module}.{name} refused")


def safe_loads(b):
    try:
        return SafeUnpickler(io.BytesIO(b)).load(), "default"
    except UnicodeDecodeError:
        return SafeUnpickler(io.BytesIO(b), encoding="latin1").load(), "latin1"


def tname(v):
    if isinstance(v, list):
        return f"list[{tname(v[0]) if v else '?'}]"
    return type(v).__name__


def log_dl(url, nbytes, purpose, action):
    with open(DL, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{url}\t{nbytes}\t{purpose}\t{action}\n")


ids_out, anns_out, refsets = {}, {}, {}
summary = {}
for ds, (ts, splitbys) in SRC.items():
    url = URL.format(ts=ts, ds=ds)
    for sb in splitbys:
        member = f"{ds}/refs({sb}).p"
        f = RangeFile(url)
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=4 << 20))
        info = z.getinfo(member)
        raw = z.read(member)
        sha = hashlib.sha256(raw).hexdigest()
        refs, enc = safe_loads(raw)
        del raw
        log_dl(url + f"#{member}", f.fetched,
               f"RefCOCO-family refs stats; member {member} uncompressed={info.file_size} compressed={info.compress_size}; "
               f"HTTP Range reads, held in memory only (never written to disk); n_http_range_requests={f.nreq}",
               "in-memory-only")
        per = collections.defaultdict(lambda: {"refs": 0, "sents": 0, "images": set(), "anns": set(), "cats": set()})
        for r in refs:
            p = per[r["split"]]
            p["refs"] += 1; p["sents"] += len(r["sentences"]); p["images"].add(r["image_id"])
            p["anns"].add(r["ann_id"]); p["cats"].add(r["category_id"])
        allimg = set(r["image_id"] for r in refs); allann = set(r["ann_id"] for r in refs)
        r0 = refs[0]
        s0 = r0["sentences"][0]
        key = f"{ds}({sb})"
        # image overlap between splits within this (dataset, splitBy)
        splits = sorted(per)
        overlap = {f"{a}&{b}": len(per[a]["images"] & per[b]["images"]) for a, b in itertools.combinations(splits, 2)}
        fn_prefixes = collections.Counter(r["file_name"].rsplit("_", 2)[0] for r in refs)
        stats = {
            "source_url": url, "member": member, "member_uncompressed_bytes": info.file_size,
            "member_compressed_bytes": info.compress_size, "member_sha256": sha, "bytes_fetched": f.fetched,
            "pickle_decode": enc, "container_type": tname(refs),
            "n_refs": len(refs), "n_sentences": sum(len(r["sentences"]) for r in refs),
            "n_images": len(allimg), "n_ann_ids": len(allann), "n_categories": len(set(r["category_id"] for r in refs)),
            "ref_ids_unique": len(set(r["ref_id"] for r in refs)),
            "sent_ids_unique": len(set(s for r in refs for s in r["sent_ids"])),
            "per_split": {s: {"refs": per[s]["refs"], "sentences": per[s]["sents"], "images": len(per[s]["images"]),
                              "ann_ids": len(per[s]["anns"]), "categories": len(per[s]["cats"])} for s in splits},
            "image_overlap_between_splits": overlap,
            "ref_keys_types": {k: tname(v) for k, v in r0.items()},
            "sentence_keys_types": {k: tname(v) for k, v in s0.items()},
            "file_name_prefix_counts": dict(fn_prefixes),
            "example_ref": {k: (v if k != "sentences" else v[:2]) for k, v in r0.items()},
            "all_refs_have_same_keys": all(set(r) == set(r0) for r in refs),
        }
        summary[key] = stats
        ids_out[key] = {s: sorted(per[s]["images"]) for s in splits}
        anns_out[key] = sorted(allann)
        refsets[key] = {r["ref_id"]: (r["ann_id"], r["image_id"], r["split"], tuple(r["sent_ids"])) for r in refs}
        print(f"{key:18s} refs={stats['n_refs']} sents={stats['n_sentences']} imgs={stats['n_images']} "
              f"anns={stats['n_ann_ids']} fetched={f.fetched} per_split={stats['per_split']}", flush=True)
        del refs

# cross-splitBy identity checks
cross = {}
for a, b in [("refcoco(unc)", "refcoco(google)"), ("refcocog(umd)", "refcocog(google)")]:
    A, B = refsets[a], refsets[b]
    common = set(A) & set(B)
    same_payload = sum(1 for k in common if A[k][:2] == B[k][:2] and A[k][3] == B[k][3])
    tab = collections.Counter((A[k][2], B[k][2]) for k in common)
    cross[f"{a} vs {b}"] = {"ref_ids_a": len(A), "ref_ids_b": len(B), "ref_ids_common": len(common),
                            "common_with_same_ann_image_sents": same_payload,
                            "images_a": len(set(v[1] for v in A.values())), "images_b": len(set(v[1] for v in B.values())),
                            "images_common": len(set(v[1] for v in A.values()) & set(v[1] for v in B.values())),
                            "split_crosstab(a_split,b_split)->refs": {f"{x}|{y}": n for (x, y), n in sorted(tab.items())}}
# dataset-level unions / overlaps (images and ann_ids)
img = {k: set(itertools.chain.from_iterable(v.values())) for k, v in ids_out.items()}
fam = {"refcoco": img["refcoco(unc)"] | img["refcoco(google)"], "refcoco+": img["refcoco+(unc)"],
       "refcocog": img["refcocog(umd)"] | img["refcocog(google)"]}
union = fam["refcoco"] | fam["refcoco+"] | fam["refcocog"]
annfam = {"refcoco": set(anns_out["refcoco(unc)"]) | set(anns_out["refcoco(google)"]), "refcoco+": set(anns_out["refcoco+(unc)"]),
          "refcocog": set(anns_out["refcocog(umd)"]) | set(anns_out["refcocog(google)"])}
annunion = set().union(*annfam.values())
evalsplits = {"refcoco(unc)": ["val", "testA", "testB"], "refcoco+(unc)": ["val", "testA", "testB"],
              "refcocog(umd)": ["val", "test"], "refcoco(google)": ["val", "test"], "refcocog(google)": ["val"]}
leak = {}
for tr in ["refcoco(unc)", "refcoco+(unc)", "refcocog(umd)", "refcoco(google)", "refcocog(google)"]:
    trimg = set(ids_out[tr].get("train", []))
    for ev, sps in evalsplits.items():
        if ev == tr: continue
        evimg = set(itertools.chain.from_iterable(ids_out[ev].get(s, []) for s in sps))
        leak[f"train[{tr}] & eval[{ev} {'+'.join(sps)}]"] = len(trimg & evimg)
xs = {
    "images_per_family": {k: len(v) for k, v in fam.items()},
    "images_union_all": len(union),
    "images_pairwise_intersections": {f"{a}&{b}": len(fam[a] & fam[b]) for a, b in itertools.combinations(fam, 2)},
    "images_triple_intersection": len(fam["refcoco"] & fam["refcoco+"] & fam["refcocog"]),
    "ann_ids_per_family": {k: len(v) for k, v in annfam.items()},
    "ann_ids_union_all": len(annunion),
    "ann_ids_pairwise_intersections": {f"{a}&{b}": len(annfam[a] & annfam[b]) for a, b in itertools.combinations(annfam, 2)},
    "train_image_vs_other_dataset_eval_image_overlap": leak,
    "cross_splitby": cross,
    "image_id_range": [min(union), max(union)],
}
summary["_cross"] = xs
print(json.dumps(xs, indent=1))
with open(os.path.join(OUT, "refs_stats.json"), "w") as f: json.dump(summary, f, indent=1, default=str)
with open(os.path.join(OUT, "refs_image_ids.json"), "w") as f:
    json.dump({"per_dataset_split": ids_out, "union_sorted": sorted(union)}, f, separators=(",", ":"))
with open(os.path.join(OUT, "refs_ann_ids.json"), "w") as f:
    json.dump({"per_dataset": anns_out, "union_sorted": sorted(annunion)}, f, separators=(",", ":"))
for fn in ["refs_stats.json", "refs_image_ids.json", "refs_ann_ids.json"]:
    print(fn, os.path.getsize(os.path.join(OUT, fn)))
