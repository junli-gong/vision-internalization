"""RUN-0001 §6 group A: datasets-server /statistics + /filter (server-side SQL, JSON only, no data download)
for jxu124 RefCOCO mirrors: 'split' column label frequencies, and counts of refs whose raw_anns has iscrowd 1 / 0
(positive control) and polygon ("segmentation": [[) vs RLE ("counts").
usage: python p6_A_dsfilter.py <out.json>"""
import json, sys, time, urllib.request, urllib.parse, urllib.error
from huggingface_hub import get_token
tok = get_token(); B = "https://datasets-server.huggingface.co"
def g(u):
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": f"Bearer {tok}"}), timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < 4: time.sleep(3 * 2 ** i); continue
            return {"HTTPError": e.code, "body": e.read()[:300].decode()}
        except Exception as e:
            if i < 4: time.sleep(3 * 2 ** i); continue
            return {"error": repr(e)}
W = {"iscrowd1": "\"raw_anns\" LIKE '%\"iscrowd\": 1%'", "iscrowd0": "\"raw_anns\" LIKE '%\"iscrowd\": 0%'",
     "seg_polygon": "\"raw_anns\" LIKE '{\"segmentation\": [[%'", "seg_rle_counts": "\"raw_anns\" LIKE '%\"counts\"%'"}
out = {}
for ds, splits in [("jxu124/refcoco", ["train", "validation", "test", "testB"]), ("jxu124/refcocoplus", ["train", "validation", "test", "testB"]),
                   ("jxu124/refcocog", ["train", "validation", "test"])]:
    for sp in splits:
        q = f"dataset={urllib.parse.quote(ds, safe='')}&config=default&split={sp}"
        st = g(f"{B}/statistics?{q}")
        sc = next((c["column_statistics"].get("frequencies") for c in st.get("statistics", []) if c["column_name"] == "split"), st if "statistics" not in st else None)
        rec = {"split_column_frequencies": sc, "statistics_partial": st.get("partial")}
        for k, w in W.items():
            fl = g(f"{B}/filter?{q}&where={urllib.parse.quote(w)}&offset=0&length=1")
            rec[f"filter_{k}_num_rows_total"] = fl.get("num_rows_total", fl)
        out[f"{ds}/{sp}"] = rec
        print(ds, sp, json.dumps(rec), flush=True)
        json.dump(out, open(sys.argv[1], "w"), indent=1)
