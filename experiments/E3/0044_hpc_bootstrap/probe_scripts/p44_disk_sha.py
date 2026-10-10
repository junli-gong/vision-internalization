"""RUN-0044 §4: recompute sha256 of every file on disk under <root>/<source>/ and check it against that source's own MANIFEST.json
(do not trust a relayed MANIFEST). usage: python p44_disk_sha.py <train root> <out prefix> <source> [<source> ...]
Exit 0 = for every source, every MANIFEST file exists on disk with the same sha256 and bytes."""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest(), os.path.getsize(p)


def main(root, prefix, *sources):
    out, ok, md = {}, True, ["# RUN-0044 盘上文件重算 sha256，对照各源自己的 MANIFEST.json", "",
                             "| 源 | MANIFEST 文件数 | 盘上文件数（不含 MANIFEST.json） | sha256 与字节都相同 | 不同 | MANIFEST 有、盘上缺 | 盘上有、MANIFEST 没有 |",
                             "|---|---|---|---|---|---|---|"]
    for s in sources:
        d = Path(root) / s
        m = json.loads((d / "MANIFEST.json").read_text())
        disk = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
        with ProcessPoolExecutor(int(os.environ.get("NPROC", "16"))) as ex:
            got = dict(zip(disk, ex.map(sha, [d / k for k in disk], chunksize=64)))
        same = sum(1 for k, v in m["files"].items() if k in got and got[k] == (v["sha256"], v["bytes"]))
        diff = sorted(k for k, v in m["files"].items() if k in got and got[k] != (v["sha256"], v["bytes"]))
        missing = sorted(set(m["files"]) - set(got))
        extra = sorted(set(got) - set(m["files"]))
        out[s] = {"manifest_n_files": m["n_files"], "manifest_total_bytes": m["total_bytes"], "disk_n_files": len(disk),
                  "disk_total_bytes": sum(v[1] for v in got.values()), "same": same, "differ": diff, "missing_on_disk": missing,
                  "extra_on_disk": extra}
        ok &= not diff and not missing and same == m["n_files"]
        md.append(f"| {s} | {m['n_files']} | {len(disk)} | {same} | {len(diff)} | {len(missing)} | {len(extra)} |")
        if extra:
            md.append(f"|   盘上有、MANIFEST 没有（前 10 个） | " + ", ".join(extra[:10]) + " ||||||")
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + f"\n\n全部相同：{ok}\n")
    print("\n".join(md)); print("all_ok", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
