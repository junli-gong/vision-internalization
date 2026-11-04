"""X-M 通过 G-XM 之后的结果汇总（D165 第五节；D177–D180）。管理员的分析脚本，只读结果目录，不跑模型，不改任何结果。

- 检测与实例分割（Rex-Omni 口径的类别宏平均 F1，IoU 0.5，XM-3a；src/xm/metrics 的 table / point / paired_bootstrap，按图像重抽）：
  - 划分：xm_decision（RUN-0106，07）、xm_final（RUN-0109，07）、lvis_sub（RUN-0112，07；每图最多 3 类，XM-2b）；
  - 系统：X-M 的 4 个设置、基座零样本的 2 个设置、Rex-Omni 的 2 个设置（RUN-0111，06；joint = 它自己的协议，标题读数；
    rexomni_det = 与 X-M 同协议的诊断；只做检测；xm_final 与 lvis_sub）；
  - 配对差：检测 X-M − 基座、X-M − Rex-Omni（joint、逐类）；实例分割 X-M 主设置 − X-M 的框交给 GrabCut、− 基座的框交给 GrabCut、
    − 框内椭圆、− 样条直接光栅化。
- RefCOCO 的 W2 test（5 个划分，各 1,000 句）：X-M（RUN-0111，06）对 X-2（RUN-0092，06，同节点），cIoU（ratio-of-sums）与 gIoU，
  逐句配对；主设置 − 椭圆。读法与 CI 同 scripts/mgr/x_test_table.py（同一组重抽下标）。
- ReasonSeg val（196 条，描述性）：X-M（RUN-0109）对 X-2（RUN-0105），同在 07，逐条配对 gIoU 与 cIoU（不扣忽略区域）。
- 重抽 10,000 次，95%，种子 src/constants.BOOTSTRAP_SEED。缺的叶目录（例如开发机上没有、只在 HPC 上的 predictions.jsonl）记为缺，不中断。

用法：python scripts/mgr/xm_results_table.py [--root <结果所在的仓库，缺省本仓库>] [--out docs/analysis/xm_results]  → <out>.json 与 <out>.md
（HPC 上：代码在稀疏的工作树里，结果在 $REPO，给 --root $REPO；--out 相对 --root。）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT  # 结果目录所在的仓库（--root）

from src import constants as C  # noqa: E402
from src.xm import metrics as MM  # noqa: E402

_spec = importlib.util.spec_from_file_location("x_test_table", ROOT / "scripts/mgr/x_test_table.py")
XT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(XT)

N_RES, LEVEL = 10_000, 0.95
M = "qwen3.5-9b"
XS = ("xm_spline_trimap", "xm_spline_raster", "xm_ellipse_trimap", "xm_box_grabcut")
BS = ("base_det", "base_box_grabcut")
RS = ("rexomni_det_joint", "rexomni_det")
MAIN = "xm_spline_trimap"
DEC = "experiments/XM/0106_xm_decision_hpcgpu07"
FIN = "experiments/XM/0109_xm_final_hpcgpu07"
LVS = "experiments/XM/0112_xm_lvis_hpcgpu07"
REX = "experiments/XM/0111_rexomni_xmtest_hpcgpu06"
LEAVES = {
    "xm_decision": {**{s: f"{DEC}/0106_xm_dec_r2/{M}/xm/{s}/xm_decision" for s in XS},
                    **{s: f"{DEC}/0106_base_dec/{M}/xm/{s}/xm_decision" for s in BS}},
    "xm_final": {**{s: f"{FIN}/0109_xm_final/{M}/xm/{s}/xm_final" for s in XS},
                 **{s: f"{FIN}/0109_base_final/{M}/xm/{s}/xm_final" for s in BS},
                 **{s: f"{REX}/0111_rexomni_final/rex-omni/xm/{s}/xm_final" for s in RS}},
    "lvis_sub": {**{s: f"{LVS}/0112_xm_lvis/{M}/xm/{s}/lvis_sub" for s in XS},
                 **{s: f"{LVS}/0112_base_lvis/{M}/xm/{s}/lvis_sub" for s in BS},
                 **{s: f"{REX}/0111_rexomni_lvis/rex-omni/xm/{s}/lvis_sub" for s in RS}},
}
DET_PAIRS = (("base_det", "X-M − 基座零样本"), ("rexomni_det_joint", "X-M − Rex-Omni（joint，它自己的协议）"),
             ("rexomni_det", "X-M − Rex-Omni（逐类，与 X-M 同协议）"))
SEG_PAIRS = (("xm_box_grabcut", "主设置 − X-M 的框交给 GrabCut"), ("base_box_grabcut", "主设置 − 基座的框交给 GrabCut"),
             ("xm_ellipse_trimap", "主设置 − 框内椭圆"), ("xm_spline_raster", "主设置 − 样条直接光栅化（不贴边）"))
XM_TEST = "experiments/XM/0111_rexomni_xmtest_hpcgpu06/0111_xm_test"
X2_TEST = "experiments/X/0092_x2_test_hpcgpu06"
RS_XM = "experiments/XM/0109_xm_final_hpcgpu07/0109_reasonseg_xm"
RS_X2 = "experiments/XM/0105_xm_tune_hpcgpu07/0105_reasonseg_x2"


def rows(leaf: str) -> list[dict] | None:
    p = DATA / leaf / "predictions.jsonl"
    if not p.exists():
        return None
    return sorted((json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()), key=lambda r: r["id"])


def metrics_point(leaf: str, kind: str) -> dict | None:
    p = DATA / leaf / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8")).get(kind)
    if not m:
        return None
    f = m["f1_rexomni"]
    return {"f1": f["value"], "p": f["precision"], "r": f["recall"], "ci": [f["ci"]["low"], f["ci"]["high"]] if f.get("ci") else None,
            "f1_miou": (m.get("f1_rexomni_miou") or {}).get("value"), "f1_exact": (m.get("f1_rexomni_exact") or {}).get("value"),
            "n_pairs": m.get("n_pairs"), "n_images": m.get("n_images"), "n_categories": m.get("n_categories")}


def paired(a_leaf: str, b_leaf: str, kind: str) -> dict:
    ra, rb = rows(a_leaf), rows(b_leaf)
    if ra is None or rb is None:
        return {"status": "缺", "missing": [x for x, r in ((a_leaf, ra), (b_leaf, rb)) if r is None]}
    ia, ib = [r["id"] for r in ra], [r["id"] for r in rb]
    if ia != ib:
        return {"status": "对的集合不同", "n_a": len(ia), "n_b": len(ib)}
    ta, tb = MM.table(ra, kind), MM.table(rb, kind)
    ci = MM.paired_bootstrap(ta, tb, MM.PRIMARY, n_resamples=N_RES, level=LEVEL, seed=C.BOOTSTRAP_SEED)
    return {"status": "完成", "n_pairs": len(ia), "a": MM.point(ta, MM.PRIMARY), "b": MM.point(tb, MM.PRIMARY),
            "diff": None if ci is None else ci.estimate, "ci": None if ci is None else [ci.low, ci.high]}


def det_seg(split: str) -> dict:
    lv = LEAVES[split]
    out = {"points": {s: {"det": metrics_point(p, "det"), "seg": metrics_point(p, "seg")} for s, p in lv.items()},
           "det_paired": {}, "seg_paired": {}}
    for other, name in DET_PAIRS:
        if other in lv:
            out["det_paired"][name] = paired(lv[MAIN], lv[other], "det")
    for other, name in SEG_PAIRS:
        if other in lv:
            out["seg_paired"][name] = paired(lv[MAIN], lv[other], "seg")
    return out


def refseg_pair(a_run: str, b_run: str, setting_a: str, setting_b: str, splits) -> dict:
    out, pa, pb = {}, {}, {}
    for sp in splits:
        a = XT.W2T.preds(XT.leaf(a_run, setting_a, sp))
        b = XT.W2T.preds(XT.leaf(b_run, setting_b, sp))
        if not a or not b:
            out[sp] = {"status": "缺"}
            continue
        out[sp] = {"status": "完成", "a": XT.single_ci(a), "b": XT.single_ci(b), "ciou_diff": XT.paired(a, b),
                   "a_giou": XT.single_g(a), "b_giou": XT.single_g(b), "giou_diff": XT.paired_g(a, b)}
        pa.update(a); pb.update(b)
    if len(splits) > 1 and all(out[sp]["status"] == "完成" for sp in splits):
        out["pooled"] = {"status": "完成", "a": XT.single_ci(pa), "b": XT.single_ci(pb), "ciou_diff": XT.paired(pa, pb),
                         "a_giou": XT.single_g(pa), "b_giou": XT.single_g(pb), "giou_diff": XT.paired_g(pa, pb)}
    return out


def fmt_ci(c) -> str:
    return "—" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def to_md(doc: dict) -> str:
    L = ["# X-M 的结果汇总（G-XM 通过之后；管理员的分析脚本 `scripts/mgr/xm_results_table.py` 生成，不要手改）", "",
         f"- 配对 bootstrap：{N_RES:,} 次、{LEVEL:.0%}、种子 {C.BOOTSTRAP_SEED}。检测与实例分割按图像重抽；指代分割逐句配对。",
         "- F1 是 Rex-Omni 口径的类别宏平均（XM-3a），IoU 0.5。", ""]
    names = {"xm_decision": "判定集 xm_decision（500 张，1,437 对；07）", "xm_final": "最终报告集 xm_final（1,000 张，2,940 对；07，Rex-Omni 在 06）",
             "lvis_sub": "LVIS 子集 lvis_sub（1,000 张，每图最多 3 类；07，Rex-Omni 在 06；描述性）"}
    for split, d in doc["detection_instance"].items():
        L += [f"## {names[split]}", "", "| 系统 / 设置 | 检测 F1 [95% CI] | 检测 P / R | 实例分割 F1 [95% CI] | 检测 F1@mIoU |", "|---|---|---|---|---|"]
        for s, p in d["points"].items():
            de, se = p["det"], p["seg"]
            dcell = "缺" if not de else f"{de['f1']:.3f}" + (f" [{de['ci'][0]:.3f}, {de['ci'][1]:.3f}]" if de["ci"] else "")
            pr = "—" if not de else f"{de['p']:.3f} / {de['r']:.3f}"
            scell = "—" if not se else f"{se['f1']:.3f}" + (f" [{se['ci'][0]:.3f}, {se['ci'][1]:.3f}]" if se["ci"] else "")
            mi = "—" if not de or de["f1_miou"] is None else f"{de['f1_miou']:.3f}"
            L.append(f"| {s} | {dcell} | {pr} | {scell} | {mi} |")
        L += ["", "| 配对差（X-M 主设置在前） | 读数 | 差 [95% CI] |", "|---|---|---|"]
        for kind, key in (("检测", "det_paired"), ("实例分割", "seg_paired")):
            for name, r in d[key].items():
                cell = f"{r['diff']:+.3f} {fmt_ci(r['ci'])}" if r.get("status") == "完成" and r.get("diff") is not None else r.get("status", "缺")
                L.append(f"| {name} | {kind} | {cell} |")
        L.append("")
    t = doc["refcoco_test"]
    L += ["## RefCOCO 的 W2 test（5 × 1,000 句）：X-M（RUN-0111）对 X-2（RUN-0092），同在 06，主设置 x2_spline_trimap", "",
          "| 划分 | X-M cIoU | X-2 cIoU | cIoU 差 [95% CI] | gIoU 差 [95% CI] |", "|---|---|---|---|---|"]
    for sp, r in t["xm_vs_x2"].items():
        if r["status"] != "完成":
            L.append(f"| {sp} | 缺 | | | |")
            continue
        L.append(f"| {sp} | {r['a']['ciou']:.3f} | {r['b']['ciou']:.3f} | {r['ciou_diff']['diff']:+.4f} {fmt_ci(r['ciou_diff']['ci'])} | "
                 f"{r['giou_diff']['diff']:+.4f} {fmt_ci(r['giou_diff']['ci'])} |")
    L += ["", "X-M 自己：主设置 − 框内椭圆（cIoU）：", ""]
    for sp, r in t["xm_spline_vs_ellipse"].items():
        if r["status"] == "完成":
            L.append(f"- {sp}：{r['ciou_diff']['diff']:+.4f} {fmt_ci(r['ciou_diff']['ci'])}")
    rs = doc["reasonseg"]
    L += ["", "## ReasonSeg val（196 条，描述性）：X-M（RUN-0109）对 X-2（RUN-0105），同在 07，不扣忽略区域", ""]
    r = rs.get("reasonseg_val", {})
    if r.get("status") == "完成":
        L += [f"- cIoU：X-M {r['a']['ciou']:.3f}，X-2 {r['b']['ciou']:.3f}，差 {r['ciou_diff']['diff']:+.4f} {fmt_ci(r['ciou_diff']['ci'])}",
              f"- gIoU：X-M {r['a_giou']['giou']:.3f}，X-2 {r['b_giou']['giou']:.3f}，差 {r['giou_diff']['diff']:+.4f} {fmt_ci(r['giou_diff']['ci'])}"]
    else:
        L.append("- 缺")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="结果所在的仓库（缺省本仓库）")
    ap.add_argument("--out", default="docs/analysis/xm_results")
    a = ap.parse_args(argv)
    global DATA
    if a.root:
        DATA = Path(a.root).resolve()
        XT.ROOT = DATA  # x_test_table.leaf 读它的模块全局 ROOT
    doc = {"n_resamples": N_RES, "level": LEVEL, "seed": C.BOOTSTRAP_SEED, "root": str(DATA), "leaves": LEAVES,
           "detection_instance": {sp: det_seg(sp) for sp in LEAVES},
           "refcoco_test": {"xm_vs_x2": refseg_pair(XM_TEST, X2_TEST, XT.MAIN, XT.MAIN, XT.SPLITS),
                            "xm_spline_vs_ellipse": refseg_pair(XM_TEST, XM_TEST, XT.MAIN, XT.ABL, XT.SPLITS)},
           "reasonseg": refseg_pair(RS_XM, RS_X2, XT.MAIN, XT.MAIN, ["reasonseg_val"])}
    out = DATA / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(to_md(doc), encoding="utf-8")
    print(out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
