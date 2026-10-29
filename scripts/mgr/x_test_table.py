"""通过 G-EX 的候选在 W2 test 上的汇总（预注册 X-6、X-6a；D147、D149）。管理员的分析脚本，只读结果目录，不跑模型。

- 候选：C1 = X-2、C2 = X-3（RUN-0092，06），设置 x2_spline_trimap（主）与 x2_ellipse_trimap（能力归属的消融）。
- 对照（同节点 06）：RUN-0072 的 W2-c、W1-F（同一份 test 清单、同一个取样函数，逐 id 配对）；外部基线 LLMSeg（RUN-0078，
  每个划分 300 句，修正后 = 主配置、修正前 = 阶段 2），在同样 1,500 句上配对。
- 读数：每个划分与合并（5,000 句；LLMSeg 1,500 句）的 cIoU（ratio-of-sums）与 gIoU（逐句 IoU 的均值）及 95% CI；配对差的 95% CI。
- 计算记账（X-6）：每句的模型调用、生成与输入 token、模型调用时间、墙钟，取各叶目录的 x2_summary / compute.json，与 W2-c、W1-F、
  LLMSeg 并列；合并为 5 个划分按句数加权的均值。
  重抽 10,000 次，种子 src/constants.BOOTSTRAP_SEED；与 src.stats 的参考实现同一组下标（同 scripts/mgr/w2_test_table.py 的
  fast_paired_ratio_ci，--check 核对过逐位相同）。
- 缺的候选（还没跑完）跳过并注明。

用法：python scripts/mgr/x_test_table.py [--out docs/analysis/x_test_table]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import constants as C  # noqa: E402
from src import stats as S  # noqa: E402
from src.stats import ratio_of_sums  # noqa: E402

_spec = importlib.util.spec_from_file_location("w2t", ROOT / "scripts/mgr/w2_test_table.py")
W2T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W2T)

SPLITS = W2T.SPLITS
M = "qwen3.5-9b"
CANDS = {"C1 (X-2)": "experiments/X/0092_x2_test_hpcgpu06", "C2 (X-3)": "experiments/X/0092_x3_test_hpcgpu06"}
MAIN, ABL = "x2_spline_trimap", "x2_ellipse_trimap"
REF = "experiments/W2/0072_w2_test_hpcgpu06"
LLMSEG = "experiments/W2/0078_llmseg_hpcgpu06"
N_RES, LEVEL = 10_000, 0.95


def leaf(run: str, setting: str, split: str) -> Path:
    return ROOT / run / M / "refseg" / setting / split


def llmseg_preds(split: str) -> tuple[dict, dict] | None:
    p = ROOT / LLMSEG / M / "refseg" / "llmseg" / split / "predictions.jsonl"
    if not p.exists():
        return None
    rec, seg = {}, {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rec[r["id"]] = {"inter": int(r["score"]["inter"]), "union": int(r["score"]["union"])}
        seg[r["id"]] = {"inter": int(r["stage2"]["score"]["inter"]), "union": int(r["stage2"]["score"]["union"])}
    return rec, seg


def arr(d: dict, ids: list[str]) -> np.ndarray:
    return np.array([[d[i]["inter"], d[i]["union"]] for i in ids], dtype=np.float64)


def ious(d: dict, ids: list[str]) -> np.ndarray:
    """逐句 IoU（gIoU 的样本）；并集为 0 的句子在这些划分里没有，出现就报错而不是猜。"""
    A = arr(d, ids)
    if (A[:, 1] <= 0).any():
        raise SystemExit("有并集为 0 的句子，gIoU 的口径要先定")
    return A[:, 0] / A[:, 1]


def single_g(d: dict) -> dict:
    ids = sorted(d)
    x = ious(d, ids)
    ci = S.bootstrap_ci(x, n_resamples=N_RES, level=LEVEL, seed=C.BOOTSTRAP_SEED)
    return {"n": len(ids), "giou": ci.estimate, "ci": [ci.low, ci.high]}


def paired_g(a: dict, b: dict, ids: list[str] | None = None) -> dict:
    ids = sorted(ids if ids is not None else a)
    if not set(ids) <= set(a) or not set(ids) <= set(b):
        raise SystemExit("配对的句子不全")
    ci = S.paired_bootstrap_ci(ious(a, ids), ious(b, ids), n_resamples=N_RES, level=LEVEL, seed=C.BOOTSTRAP_SEED)
    return {"n": len(ids), "diff": ci.estimate, "ci": [ci.low, ci.high]}


COMPUTE_KEYS = ("n_calls", "generated_tokens", "prompt_tokens", "call_time_s", "wall_time_s")


def compute_of(run: str, setting: str, sp: str) -> dict | None:
    """每句的计算量：候选取 x2_summary/<划分>.json 的 per_sample，其余取叶目录 compute.json 的 per_sample_mean。"""
    xs = ROOT / run / M / "refseg" / "x2_summary" / f"{sp}.json"
    if xs.exists() and setting in json.loads(xs.read_text(encoding="utf-8")).get("settings", {}):
        ps = json.loads(xs.read_text(encoding="utf-8"))["settings"][setting]["per_sample"]
        n = json.loads(xs.read_text(encoding="utf-8"))["n"]
    else:
        cj = ROOT / run / M / "refseg" / setting / sp / "compute.json"
        if not cj.exists():
            return None
        c = json.loads(cj.read_text(encoding="utf-8"))
        ps, n = c["per_sample_mean"], c["n_samples"]
    return {"n": n} | {k: float(ps[k]) for k in COMPUTE_KEYS}


def pooled_compute(run: str, setting: str) -> dict | None:
    rows = [compute_of(run, setting, sp) for sp in SPLITS]
    if not all(rows):
        return None
    n = sum(r["n"] for r in rows)
    return {"n": n} | {k: sum(r[k] * r["n"] for r in rows) / n for k in COMPUTE_KEYS}


def single_ci(d: dict) -> dict:
    ids = sorted(d)
    A = arr(d, ids)
    idx = S._resample_indices(len(ids), N_RES, C.BOOTSTRAP_SEED)
    boot = A[idx, 0].sum(1) / A[idx, 1].sum(1)
    ci = S._percentile_ci(float(ratio_of_sums(A)), boot, LEVEL, N_RES)
    return {"n": len(ids), "ciou": ci.estimate, "ci": [ci.low, ci.high]}


def paired(a: dict, b: dict, ids: list[str] | None = None) -> dict:
    ids = sorted(ids if ids is not None else a)
    if not set(ids) <= set(a) or not set(ids) <= set(b):
        raise SystemExit("配对的句子不全")
    ci = W2T.fast_paired_ratio_ci(arr(a, ids), arr(b, ids))
    return {"n": len(ids), "diff": ci.estimate, "ci": [ci.low, ci.high]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/x_test_table")
    a = ap.parse_args(argv)
    ref = {s: {sp: W2T.preds(leaf(REF, s, sp)) for sp in SPLITS} for s in ("w2c", "w1f")}
    llm = {sp: llmseg_preds(sp) for sp in SPLITS}
    doc: dict = {"n_resamples": N_RES, "level": LEVEL, "seed": C.BOOTSTRAP_SEED, "reference": {}, "candidates": {}}
    for s in ("w2c", "w1f"):
        if all(ref[s][sp] for sp in SPLITS):
            pool = {k: v for sp in SPLITS for k, v in ref[s][sp].items()}
            doc["reference"][s] = {sp: single_ci(ref[s][sp]) for sp in SPLITS} | {"pooled": single_ci(pool)}
            doc["reference"][s + "_giou"] = {sp: single_g(ref[s][sp]) for sp in SPLITS} | {"pooled": single_g(pool)}
    for name, run in CANDS.items():
        main_p = {sp: W2T.preds(leaf(run, MAIN, sp)) for sp in SPLITS}
        abl_p = {sp: W2T.preds(leaf(run, ABL, sp)) for sp in SPLITS}
        if not all(main_p.values()) or not all(abl_p.values()):
            doc["candidates"][name] = {"status": "未跑完", "splits_done": [sp for sp in SPLITS if main_p[sp]]}
            continue
        e: dict = {"status": "完成", "per_split": {}, "pooled": {}}
        pools = {k: {} for k in ("main", "abl", "w2c", "w1f")}
        llm_ids: list[str] = []
        llm_rec, llm_seg, main_llm = {}, {}, {}
        for sp in SPLITS:
            r = {"main": single_ci(main_p[sp]), "ablation": single_ci(abl_p[sp]),
                 "main_minus_w2c": paired(main_p[sp], ref["w2c"][sp]),
                 "main_minus_w1f": paired(main_p[sp], ref["w1f"][sp]),
                 "main_minus_ablation": paired(main_p[sp], abl_p[sp]),
                 "g_main": single_g(main_p[sp]), "g_ablation": single_g(abl_p[sp]),
                 "g_main_minus_w2c": paired_g(main_p[sp], ref["w2c"][sp]),
                 "g_main_minus_ablation": paired_g(main_p[sp], abl_p[sp])}
            if llm[sp]:
                rec, seg = llm[sp]
                ids = sorted(rec)
                r["llmseg"] = {"n": len(ids), "main_on_subset": single_ci({i: main_p[sp][i] for i in ids}),
                               "main_minus_llmseg_rec": paired(main_p[sp], rec, ids),
                               "main_minus_llmseg_seg": paired(main_p[sp], seg, ids)}
                llm_ids += ids
                llm_rec.update(rec); llm_seg.update(seg); main_llm.update({i: main_p[sp][i] for i in ids})
            e["per_split"][sp] = r
            pools["main"].update(main_p[sp]); pools["abl"].update(abl_p[sp])
            pools["w2c"].update(ref["w2c"][sp]); pools["w1f"].update(ref["w1f"][sp])
        e["pooled"] = {"main": single_ci(pools["main"]), "ablation": single_ci(pools["abl"]),
                       "main_minus_w2c": paired(pools["main"], pools["w2c"]),
                       "main_minus_w1f": paired(pools["main"], pools["w1f"]),
                       "main_minus_ablation": paired(pools["main"], pools["abl"]),
                       "g_main": single_g(pools["main"]), "g_ablation": single_g(pools["abl"]),
                       "g_main_minus_w2c": paired_g(pools["main"], pools["w2c"]),
                       "g_main_minus_w1f": paired_g(pools["main"], pools["w1f"]),
                       "g_main_minus_ablation": paired_g(pools["main"], pools["abl"])}
        e["compute"] = {"main": pooled_compute(run, MAIN), "ablation": pooled_compute(run, ABL)}
        if llm_ids:
            e["pooled"]["llmseg"] = {"n": len(llm_ids), "main_on_subset": single_ci(main_llm),
                                     "llmseg_rec": single_ci(llm_rec), "llmseg_seg": single_ci(llm_seg),
                                     "main_minus_llmseg_rec": paired(main_llm, llm_rec, llm_ids),
                                     "main_minus_llmseg_seg": paired(main_llm, llm_seg, llm_ids),
                                     "g_main_on_subset": single_g(main_llm),
                                     "g_llmseg_rec": single_g(llm_rec), "g_llmseg_seg": single_g(llm_seg),
                                     "g_main_minus_llmseg_rec": paired_g(main_llm, llm_rec, llm_ids),
                                     "g_main_minus_llmseg_seg": paired_g(main_llm, llm_seg, llm_ids)}
        doc["candidates"][name] = e
    doc["compute_reference"] = {"W2-c": pooled_compute(REF, "w2c"), "W1-F": pooled_compute(REF, "w1f"),
                                "LLMSeg（1,500 句）": pooled_compute(LLMSEG, "llmseg")}
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def c(x):
        return f"{x['ciou']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"

    def d(x):
        return f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"

    def g(x):
        return f"{x['giou']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
    L = ["# 探索候选在 W2 test 上的读数（预注册 X-6；`scripts/mgr/x_test_table.py` 生成；只描述）", "",
         f"cIoU（ratio-of-sums）与 95% CI；配对差为逐句配对 bootstrap，{N_RES:,} 次，种子 {C.BOOTSTRAP_SEED}。对照同节点 06：RUN-0072 的 W2-c、W1-F；"
         "LLMSeg 为 RUN-0078，每个划分 300 句。", ""]
    hdr = "| | " + " | ".join(SPLITS) + " | 合并 |"
    for name, e in doc["candidates"].items():
        L += [f"## {name}", ""]
        if e.get("status") != "完成":
            L += [f"未跑完（已完成：{e.get('splits_done')}）", ""]
            continue
        P = e["per_split"]
        rows = [("主设置 x2_spline_trimap", lambda r: c(r["main"]), c(e["pooled"]["main"])),
                ("消融 x2_ellipse_trimap", lambda r: c(r["ablation"]), c(e["pooled"]["ablation"])),
                ("主 − W2-c", lambda r: d(r["main_minus_w2c"]), d(e["pooled"]["main_minus_w2c"])),
                ("主 − W1-F", lambda r: d(r["main_minus_w1f"]), d(e["pooled"]["main_minus_w1f"])),
                ("主 − 椭圆消融", lambda r: d(r["main_minus_ablation"]), d(e["pooled"]["main_minus_ablation"])),
                ("gIoU 主设置", lambda r: g(r["g_main"]), g(e["pooled"]["g_main"])),
                ("gIoU 椭圆消融", lambda r: g(r["g_ablation"]), g(e["pooled"]["g_ablation"])),
                ("gIoU 主 − W2-c", lambda r: d(r["g_main_minus_w2c"]), d(e["pooled"]["g_main_minus_w2c"])),
                ("gIoU 主 − 椭圆消融", lambda r: d(r["g_main_minus_ablation"]), d(e["pooled"]["g_main_minus_ablation"]))]
        L += [hdr, "|---|" + "---|" * (len(SPLITS) + 1)]
        for lab, fn, pooled in rows:
            L.append(f"| {lab} | " + " | ".join(fn(P[sp]) for sp in SPLITS) + f" | {pooled} |")
        if "llmseg" in e["pooled"]:
            ll = e["pooled"]["llmseg"]
            L += ["", f"LLMSeg 子集（{ll['n']} 句）：主设置 {c(ll['main_on_subset'])}；LLMSeg 修正后 {c(ll['llmseg_rec'])}，修正前 {c(ll['llmseg_seg'])}；"
                  f"主 − LLMSeg 修正后 {d(ll['main_minus_llmseg_rec'])}，主 − LLMSeg 修正前 {d(ll['main_minus_llmseg_seg'])}。",
                  "", f"gIoU，同样 {ll['n']} 句：主设置 {g(ll['g_main_on_subset'])}；LLMSeg 修正后 {g(ll['g_llmseg_rec'])}，修正前 {g(ll['g_llmseg_seg'])}；"
                  f"主 − LLMSeg 修正后 {d(ll['g_main_minus_llmseg_rec'])}，主 − LLMSeg 修正前 {d(ll['g_main_minus_llmseg_seg'])}。"]
        L.append("")
    if doc["reference"]:
        L += ["## 对照（RUN-0072，06）", "", hdr, "|---|" + "---|" * (len(SPLITS) + 1)]
        for s, r in doc["reference"].items():
            fmt = g if s.endswith("_giou") else c
            L.append(f"| {s} | " + " | ".join(fmt(r[sp]) for sp in SPLITS) + f" | {fmt(r['pooled'])} |")
    comp = [(f"{n} {lab}", e["compute"][k]) for n, e in doc["candidates"].items() if "compute" in e
            for k, lab in (("main", "样条"), ("ablation", "椭圆消融"))] + list(doc["compute_reference"].items())
    L += ["", "## 计算记账（每句均值；合并为按句数加权）", "",
          "| 系统 | 句数 | 模型调用 | 生成 token | 输入 token | 模型调用时间 s | 墙钟 s |", "|---|---|---|---|---|---|---|"]
    for lab, x in comp:
        if x:
            L.append(f"| {lab} | {x['n']:,} | {x['n_calls']:.2f} | {x['generated_tokens']:.1f} | {x['prompt_tokens']:.0f} | "
                     f"{x['call_time_s']:.2f} | {'—' if lab.startswith('LLMSeg') else format(x['wall_time_s'], '.2f')} |")
    L += ["", "墙钟取各运行自己的记录：候选与 W2-c、W1-F 是每句的墙钟，含不学习的工具；LLMSeg 的 compute.json 记的墙钟是并发下的累计，"
          "与前几行不同口径，不列。"]
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
