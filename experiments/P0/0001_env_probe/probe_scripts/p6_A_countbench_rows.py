"""RUN-0001 §6 group A: page through datasets-server /rows for nielsr/countbench (JSON only; images come back as
signed asset URLs, not downloaded) to count rows whose packaged image is null, and compare to official CountBench.json.
usage: python p6_A_countbench_rows.py <CountBench.json> <out.json>"""
import json, sys, time, urllib.request, urllib.error, collections
from huggingface_hub import get_token
tok = get_token()
def g(u):
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": f"Bearer {tok}"}), timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == 4: raise
            time.sleep(3 * 2 ** i)
off = json.load(open(sys.argv[1]))
rows = []
for o in range(0, 600, 100):
    j = g(f"https://datasets-server.huggingface.co/rows?dataset=nielsr%2Fcountbench&config=default&split=train&offset={o}&length=100")
    rows += [r["row"] for r in j["rows"]]
    if o == 0: total = j.get("num_rows_total")
null_img = [i for i, r in enumerate(rows) if r.get("image") is None]
wh = collections.Counter((r["image"]["width"], r["image"]["height"]) for r in rows if r.get("image"))
same_order = sum(1 for a, b in zip(rows, off) if a["image_url"] == b["image_url"] and a["text"] == b["text"] and a["number"] == b["number"])
out = {"num_rows_total": total, "rows_fetched": len(rows), "n_image_null": len(null_img), "n_image_present": len(rows) - len(null_img),
       "null_image_row_indices_first20": null_img[:20],
       "number_hist_image_present": dict(sorted(collections.Counter(r["number"] for r in rows if r.get("image")).items())),
       "rows_matching_official_json_same_index(url,text,number)": same_order,
       "n_distinct_image_sizes": len(wh), "most_common_sizes": wh.most_common(5)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
