"""RUN-0044 §4: compare each source's HPC MANIFEST.json with the B200 one from RUN-0034 (experiments/E3/0034_b200_prep/land_manifests/).
Same table as RUN-0034's p33_manifest_compare.py (which compares against A800 and is reused unchanged for that half).
usage: python p44_manifest_vs_b200.py <HPC train root> <repo root> <out prefix>; exit 0 = identical per-file sha256 sets for all four sources."""
import gzip, json, sys
from pathlib import Path
B = "experiments/E3/0034_b200_prep/land_manifests"
SOURCES = ["refcoco_train", "nyu_depth_train", "nyu_normals", "ade20k"]


def main(root, repo, prefix):
    out, ok = {}, True
    md = ["# RUN-0044 落地清单与 B200（RUN-0034）的对照（逐文件 sha256 的集合）", "",
          "| 源 | HPC 文件数 | B200 文件数 | HPC 总字节 | B200 总字节 | 失败成员 HPC / B200 | 两边都有且 sha256 相同 | 同路径 sha256 不同 | 只在 HPC | 只在 B200 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SOURCES:
        h = json.loads((Path(root) / s / "MANIFEST.json").read_text())
        b = json.load(gzip.open(Path(repo) / B / f"MANIFEST_{s}.json.gz"))
        hf = {k: v["sha256"] for k, v in h["files"].items()}
        bf = {k: v["sha256"] for k, v in b["files"].items()}
        same = sum(1 for k in hf if k in bf and bf[k] == hf[k])
        diff = sorted(k for k in hf if k in bf and bf[k] != hf[k])
        oh, ob = sorted(set(hf) - set(bf)), sorted(set(bf) - set(hf))
        out[s] = {"hpc_n_files": h["n_files"], "b200_n_files": b["n_files"], "hpc_total_bytes": h["total_bytes"], "b200_total_bytes": b["total_bytes"],
                  "hpc_failures": len(h.get("failures") or []), "b200_failures": len(b.get("failures") or []), "same": same,
                  "sha_differs": diff, "only_hpc": oh, "only_b200": ob}
        ok &= not diff and not oh and not ob
        md.append(f"| {s} | {h['n_files']} | {b['n_files']} | {h['total_bytes']} | {b['total_bytes']} | {out[s]['hpc_failures']} / {out[s]['b200_failures']} | "
                  f"{same} | {len(diff)} | {len(oh)} | {len(ob)} |")
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + f"\n\n完全相同：{ok}\n")
    print("\n".join(md)); print("identical", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
