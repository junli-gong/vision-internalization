"""RUN-0034 §3: compare each source's B200 MANIFEST.json with the A800 one (file count, total bytes, per-file sha256 sets).
usage: python p33_manifest_compare.py <B200 train root> <repo root> <out prefix>"""
import gzip, json, sys
from pathlib import Path
A = "experiments/P0/0026_a800_delta/results_0026"
A800 = {"refcoco_train": f"{A}/recovery_refcoco_local_20261003T120728Z/MANIFEST.json.gz",
        "nyu_depth_train": f"{A}/land_logs/MANIFEST_nyu_depth_train.json.gz",
        "nyu_normals": f"{A}/recovery_nyu_normals_local_20261003T1046/MANIFEST.json",
        "ade20k": f"{A}/land_logs/MANIFEST_ade20k.json.gz"}


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    return json.load(gzip.open(p) if p.suffix == ".gz" else open(p))


def main(root, repo, prefix):
    out, md = {}, ["# RUN-0034 落地清单与 A800 的对照（逐文件 sha256 的集合）", "",
                   "| 源 | B200 文件数 | A800 文件数 | B200 总字节 | A800 总字节 | 失败成员 B200 / A800 | 两边都有且 sha256 相同 | 同路径 sha256 不同 | 只在 B200 | 只在 A800 |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
    for src, ap in A800.items():
        b, a = load(Path(root) / src / "MANIFEST.json"), load(Path(repo) / ap)
        if b is None or a is None:
            out[src] = {"b200": b is not None, "a800": a is not None}
            md.append(f"| {src} | {'缺' if b is None else b['n_files']} | {'缺' if a is None else a['n_files']} | | | | | | | |")
            continue
        bf = {k: v["sha256"] for k, v in b["files"].items()}
        af = {k: v["sha256"] for k, v in a["files"].items()}
        same = sum(1 for k in bf if k in af and af[k] == bf[k])
        diff = sorted(k for k in bf if k in af and af[k] != bf[k])
        ob, oa = sorted(set(bf) - set(af)), sorted(set(af) - set(bf))
        r = {"b200_n_files": b["n_files"], "a800_n_files": a["n_files"], "b200_total_bytes": b["total_bytes"], "a800_total_bytes": a["total_bytes"],
             "b200_failures": len(b.get("failures") or []), "a800_failures": len(a.get("failures") or []), "same": same,
             "sha_differs": diff, "only_b200": ob, "only_a800": oa}
        out[src] = r
        md.append(f"| {src} | {b['n_files']} | {a['n_files']} | {b['total_bytes']} | {a['total_bytes']} | {r['b200_failures']} / {r['a800_failures']} | "
                  f"{same} | {len(diff)} | {len(ob)} | {len(oa)} |")
        for lab, lst in (("同路径 sha256 不同", diff), ("只在 B200", ob), ("只在 A800", oa)):
            if lst:
                md.append(f"|   {lab}（前 10 个） | " + ", ".join(lst[:10]) + " |||||||||")
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
