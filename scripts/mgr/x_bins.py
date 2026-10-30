"""按划分、COCO 类别、形状复杂度分档（D154；预注册 U-6）。管理员的分析脚本，只读结果目录与 RUN-0099 的逐句元数据，不跑模型。

- 句子：W2 test 全部 5,000 句（RUN-0092 的 X-2 主设置与椭圆消融、RUN-0072 的 W2-c，逐 id 配对）。
- 分档（U-6，写于任何分档计算之前）：
  (i) 人（COCO 类别 1）与非人；
  (ii) COCO 大类，至少 150 句的单独成档，其余合为"其他"；
  (iii) 形状复杂度 c = P² / A 的四分位档：切点 = numpy.quantile(5,000 句真值的 c, [0.25, 0.5, 0.75])（linear），只由真值算出；
        档为 [最小, q1]、(q1, q2]、(q2, q3]、(q3, 最大]；A = 0 的句子单列一档；
  (iv) 附带：相对面积的四分位档，同样的切法，描述性。
- 读数：每档句数，X-2、W2-c、椭圆的 cIoU；X-2 − W2-c、X-2 − 椭圆的配对差，95%，10,000 次（口径同 scripts/mgr/x_test_table.py）。
  每个划分另报人的占比。只描述。

用法：python scripts/mgr/x_bins.py [--meta experiments/U/0099_test_metadata/test_metadata.jsonl] [--out docs/analysis/x_bins]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("xtt", ROOT / "scripts/mgr/x_test_table.py")
XT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XT)
W2T, SPLITS = XT.W2T, XT.SPLITS
X2_RUN = XT.CANDS["C1 (X-2)"]
MIN_SUPER = 150
PERSON = 1


def pool(run: str, setting: str) -> dict:
    out = {}
    for sp in SPLITS:
        p = W2T.preds(XT.leaf(run, setting, sp))
        if p is None:
            raise SystemExit(f"{run} {setting} {sp} 没有完整的预测")
        out.update(p)
    return out


def quartile_bins(vals: dict[str, float | None], label: str) -> tuple[dict[str, str], list[float]]:
    xs = np.array([v for v in vals.values() if v is not None], dtype=np.float64)
    q = [float(x) for x in np.quantile(xs, [0.25, 0.5, 0.75], method="linear")]
    names = [f"{label} Q1 (≤{q[0]:.3g})", f"{label} Q2 ({q[0]:.3g}, {q[1]:.3g}]", f"{label} Q3 ({q[1]:.3g}, {q[2]:.3g}]", f"{label} Q4 (>{q[2]:.3g})"]
    out = {}
    for i, v in vals.items():
        out[i] = f"{label} A = 0" if v is None else names[int(np.searchsorted(q, v, side="left"))]
    return out, q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", default="experiments/U/0099_test_metadata/test_metadata.jsonl")
    ap.add_argument("--out", default="docs/analysis/x_bins")
    a = ap.parse_args(argv)
    meta_path = ROOT / a.meta
    meta = {r["id"]: r for r in map(json.loads, meta_path.read_text(encoding="utf-8").splitlines()) if r}
    x2, abl, w2c = pool(X2_RUN, XT.MAIN), pool(X2_RUN, XT.ABL), pool(XT.REF, "w2c")
    if not (set(meta) == set(x2) == set(abl) == set(w2c)):
        raise SystemExit("元数据与三个系统的句子集合不同")
    sup_n = Counter(r["supercategory"] for r in meta.values())
    schemes: dict[str, dict[str, str]] = {
        "(i) 人与非人": {i: ("人" if r["category_id"] == PERSON else "非人") for i, r in meta.items()},
        "(ii) COCO 大类": {i: (r["supercategory"] if sup_n[r["supercategory"]] >= MIN_SUPER else "其他") for i, r in meta.items()}}
    cuts = {}
    schemes["(iii) 形状复杂度 c = P²/A"], cuts["complexity"] = quartile_bins(
        {i: (r["complexity"] if r["area"] > 0 else None) for i, r in meta.items()}, "c")
    schemes["(iv) 相对面积（附带）"], cuts["rel_area"] = quartile_bins({i: r["rel_area"] for i, r in meta.items()}, "面积")
    doc: dict = {"meta": a.meta, "n": len(meta), "n_resamples": XT.N_RES, "level": XT.LEVEL, "seed": C.BOOTSTRAP_SEED,
                 "cuts": cuts, "supercategory_counts": dict(sup_n), "person_share_by_split": {}, "schemes": {}}
    for sp in SPLITS:
        ids = [i for i, r in meta.items() if r["w2_split"] == sp]
        doc["person_share_by_split"][sp] = {"n": len(ids), "person": sum(meta[i]["category_id"] == PERSON for i in ids) / len(ids)}
    for sname, assign in schemes.items():
        rows = {}
        for b in sorted(set(assign.values()), key=lambda x: (x == "其他", x)):
            ids = sorted(i for i, v in assign.items() if v == b)
            sub = lambda d: {i: d[i] for i in ids}  # noqa: E731
            rows[b] = {"n": len(ids), "x2": XT.single_ci(sub(x2))["ciou"], "w2c": XT.single_ci(sub(w2c))["ciou"],
                       "ellipse": XT.single_ci(sub(abl))["ciou"],
                       "x2_minus_w2c": XT.paired(x2, w2c, ids), "x2_minus_ellipse": XT.paired(x2, abl, ids)}
        doc["schemes"][sname] = rows
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def d(x):
        return f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    L = ["# X-2 的增益按划分、类别、形状复杂度分档（预注册 U-6；`scripts/mgr/x_bins.py` 生成；只描述）", "",
         f"W2 test 5,000 句；cIoU = Σinter / Σunion；配对差 bootstrap {XT.N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}。元数据 `{a.meta}`（RUN-0099）。", "",
         "## 每个划分里人的占比", "", "| 划分 | 句数 | 人的占比 |", "|---|---|---|"]
    for sp, v in doc["person_share_by_split"].items():
        L.append(f"| {sp} | {v['n']} | {v['person']:.2f} |")
    for sname, rows in doc["schemes"].items():
        L += ["", f"## {sname}", "", "| 档 | 句数 | X-2 | W2-c | 椭圆 | X-2 − W2-c | X-2 − 椭圆 |", "|---|---|---|---|---|---|---|"]
        for b, r in rows.items():
            L.append(f"| {b} | {r['n']:,} | {r['x2']:.3f} | {r['w2c']:.3f} | {r['ellipse']:.3f} | {d(r['x2_minus_w2c'])} | {d(r['x2_minus_ellipse'])} |")
    L += ["", f"切点：c 的四分位 {', '.join(f'{x:.3g}' for x in cuts['complexity'])}；相对面积 {', '.join(f'{x:.3g}' for x in cuts['rel_area'])}。"
          f"大类至少 {MIN_SUPER} 句单独成档。"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
