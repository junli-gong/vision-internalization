"""RUN-0095（CPU，只读）：从 HPC 上 RUN-0072 的 W2-a 逐句预测里抽出每句的交、并，写成小文件推回仓库（D152）。

- W2-a 的 predictions.jsonl 每个约 10 MB（含挑选时的长生成），超过同步上限，只留在 HPC（LARGE_FILES.manifest.tsv 记了 sha256）。
- 这里只读：先核对 sha256 与 manifest 逐字相同、metrics.json 的 complete 为 true，再逐行取 id 与 score 的 inter、union、iou。
- 不跑模型，不改任何结果目录；只写 --out。

用法：python extract_scores.py --run <RUN-0072 的结果目录> --setting w2a --out <仓库>/experiments/W2/0095_w2a_scores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

M = "qwen3.5-9b"
SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA",
          "test_refcoco+_unc_testB", "test_refcocog_umd_test"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--setting", default="w2a")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    if not out.name[:4].isdigit() or out.name[4] != "_":
        sys.exit(f"目录名没有运行编号：{out}")
    man = {}
    for line in (run / "LARGE_FILES.manifest.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            path, nbytes, sha, _ = line.split("\t")
            man[path] = (int(nbytes), sha)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"run": str(run), "setting": a.setting, "model": M, "splits": {}}
    for sp in SPLITS:
        rel = f"{M}/refseg/{a.setting}/{sp}/predictions.jsonl"
        p = run / rel
        if rel not in man:
            sys.exit(f"manifest 里没有 {rel}")
        got = sha256(p)
        if got != man[rel][1] or p.stat().st_size != man[rel][0]:
            sys.exit(f"{rel} 与 manifest 不符：sha256 {got}，字节 {p.stat().st_size}")
        m = json.loads((p.parent / "metrics.json").read_text(encoding="utf-8"))
        if m.get("complete") is not True:
            sys.exit(f"{sp} 的 metrics.json 不是 complete")
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                s = r["score"]
                rows.append({"id": r["id"], "inter": int(s["inter"]), "union": int(s["union"]), "iou": float(s["iou"])})
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            sys.exit(f"{sp} 有重复的 id")
        n_exp = m.get("n_expected", m.get("n"))
        if len(rows) != n_exp:
            sys.exit(f"{sp}：{len(rows)} 行，metrics 说 {n_exp}")
        dst = out / f"{sp}.jsonl"
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        ciou = sum(r["inter"] for r in rows) / sum(r["union"] for r in rows)
        head = next(h["value"] for h in m["headline"] if h["name"] == "cIoU")
        summary["splits"][sp] = {"n": len(rows), "source_sha256": got, "out_sha256": sha256(dst),
                                 "cIoU_from_rows": ciou, "cIoU_metrics": head, "match": abs(ciou - head) < 1e-12}
        print(sp, len(rows), f"{ciou:.6f}", f"{head:.6f}", "match" if abs(ciou - head) < 1e-12 else "MISMATCH")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0 if all(v["match"] for v in summary["splits"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
