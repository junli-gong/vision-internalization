"""RUN-0034 §4: compare B200 train_exclusions.json with the A800 one, per training set (excluded keys same / different).
usage: python p34_exclusions_compare.py <B200 train_exclusions.json> <A800 train_exclusions.json> <out prefix>; exit 0 = identical sets, 1 = differ"""
import json, sys
from pathlib import Path


def main(bp, ap, prefix):
    b, a = json.loads(Path(bp).read_text()), json.loads(Path(ap).read_text())
    out = {"max_hamming": [b.get("max_hamming"), a.get("max_hamming")], "train_sets": [b.get("train_sets"), a.get("train_sets")],
           "eval_sets_same": b.get("eval_sets") == a.get("eval_sets"), "eval_sets": [b.get("eval_sets"), a.get("eval_sets")], "sets": {}}
    md = ["# RUN-0034 去重：与 A800 剔除清单的对照", "", f"- max_hamming：B200 {b.get('max_hamming')}，A800 {a.get('max_hamming')}",
          f"- eval_sets 相同：{out['eval_sets_same']}（B200 {len(b.get('eval_sets') or [])} 个，A800 {len(a.get('eval_sets') or [])} 个）", "",
          "| 训练集 | B200 剔除条数 | A800 剔除条数 | 两边都剔除的键 | 只在 B200 | 只在 A800 |", "|---|---|---|---|---|---|"]
    ok = out["eval_sets_same"]
    for s in sorted(set(b["sets"]) | set(a["sets"])):
        def keys(x):
            v = x["sets"].get(s)
            if v is None:
                return None
            return {(e["key"] if isinstance(e, dict) else e): (json.dumps(e, sort_keys=True) if isinstance(e, dict) else e) for e in v}
        kb, ka = keys(b), keys(a)
        if kb is None or ka is None:
            ok = False
            md.append(f"| {s} | {'缺' if kb is None else len(kb)} | {'缺' if ka is None else len(ka)} | | | |")
            out["sets"][s] = {"b200": kb is not None, "a800": ka is not None}
            continue
        both = set(kb) & set(ka)
        ob, oa = sorted(set(kb) - set(ka)), sorted(set(ka) - set(kb))
        same_entry = sum(1 for k in both if kb[k] == ka[k])
        ok = ok and not ob and not oa
        out["sets"][s] = {"b200": len(kb), "a800": len(ka), "both": len(both), "both_entry_identical": same_entry, "only_b200": ob, "only_a800": oa}
        md.append(f"| {s} | {len(kb)} | {len(ka)} | {len(both)}（整条记录相同 {same_entry}） | {len(ob)} | {len(oa)} |")
    out["identical"] = ok
    md += ["", f"**完全相同：{ok}**"]
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
