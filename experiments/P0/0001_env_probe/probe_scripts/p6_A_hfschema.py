"""RUN-0001 §6 group A: print datasets-server features (compact types) + first row from hf_*.json.
usage: python p6_A_hfschema.py <hf_dir> <repo__id> ... """
import json, sys, os
def ft(f):
    if isinstance(f, list): return "list[" + ft(f[0]) + "]"
    if not isinstance(f, dict): return str(f)
    t = f.get("_type")
    if t == "Value": return f["dtype"]
    if t == "Image": return "Image"
    if t in ("Sequence", "List", "LargeList"): return f"{t}[{ft(f.get('feature'))}]"
    if t == "ClassLabel": return f"ClassLabel({len(f.get('names', []))})"
    if t is None: return "{" + ", ".join(f"{k}:{ft(v)}" for k, v in f.items()) + "}"
    return str(t)
for rid in sys.argv[2:]:
    d = json.load(open(os.path.join(sys.argv[1], f"hf_{rid}.json")))
    print("=====", rid)
    for k, v in d["ds_first_rows"].items():
        if "features" not in v: print("  ", k, "ERR", str(v)[:300]); continue
        print("  ", k, "; ".join(f"{x['name']}:{ft(x['type'])}" for x in v["features"]))
        r = v["rows_first2"][0] if v["rows_first2"] else None
        print("   row0:", json.dumps(r)[:1500])
