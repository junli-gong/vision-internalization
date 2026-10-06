"""E2 汇总（管理员脚本，只读 run_e2 的 metrics.json；G2 门报告用）。

对每个模型 × 任务 × 探针种类（linear / conv）给出：
- 标题指标（metrics.json 的 headline，方向取自它）在第 0 层、最好的层、最后一层、L{N}_pre、pre_proj 上的读数与 95% CI；
- 最好的层在全部层里的相对位置（层号 / N）；
- 最好的层与最后一层相对第 0 层的变化（绝对与相对）；
- 按位置分列（上半 / 下半、光栅四分位）在第 0 层、最好的层、最后一层上的读数（D53-3）；
- post_proj 与第 0 层逐位相同的核对（metrics.json 的 post_proj_check）。
另给噪声底：同配置重跑（--noise 目录）与正式运行逐探针标题指标的绝对差，按任务 × 种类取最大与中位数。
E2 预注册 v1：标题指标上小于噪声底两倍的层间差异不作解读——表里标出 |最好 − 第 0 层| 是否 ≥ 2 × 噪声底最大值。

用法：python scripts/mgr/e2_summary.py <主运行目录> [--noise <噪声底目录>] [--out-md f.md] [--out-json f.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

TASKS = ("depth", "semseg", "normals", "refseg")
KINDS = ("linear", "conv")
LAYER = re.compile(r"L(\d+)")


def headline_of(m: dict) -> tuple[str, str]:
    h = m["headline"]
    if isinstance(h, dict):
        return h["metric"], h.get("direction", "↓")
    if isinstance(h, (list, tuple)):
        first = h[0]
        if isinstance(first, (list, tuple)):
            return first[0], first[1]
        if isinstance(first, dict):
            return first["metric"], first.get("better", first.get("direction", "↓"))
        return h[0], h[1]
    raise ValueError(f"看不懂 headline：{h!r}")


def val(ev: dict, metric: str):
    v = ev.get(metric)
    ci = ev.get(metric + "_ci") or {}
    return v, ci.get("low"), ci.get("high")


def probe_table(m: dict, metric: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {k: {} for k in KINDS}
    for name, p in m["probes"].items():
        kind, feat = p["kind"], p["feature"]
        if kind not in out:
            continue
        v, lo, hi = val(p["eval"], metric)
        pos = {}
        for split, regions in (p.get("eval_by_position") or {}).items():
            pos[split] = {r: (d or {}).get(metric) for r, d in regions.items()}
        out[kind][feat] = {"value": v, "lo": lo, "hi": hi, "pos": pos}
    return out


def summarize(run: Path, noise: Path | None) -> dict:
    res: dict = {"models": {}, "noise": {}}
    for mdir in sorted(p for p in run.iterdir() if p.is_dir() and (p / TASKS[0] / "metrics.json").exists()):
        mres = {}
        for t in TASKS:
            f = mdir / t / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text(encoding="utf-8"))
            metric, direction = headline_of(m)
            better = (lambda a, b: a < b) if direction == "↓" else (lambda a, b: a > b)
            tab = probe_table(m, metric)
            tres = {"metric": metric, "direction": direction, "post_proj_check": m.get("post_proj_check"), "kinds": {}}
            for kind, feats in tab.items():
                layers = sorted((int(LAYER.fullmatch(f).group(1)), f) for f in feats if LAYER.fullmatch(f))
                if not layers:
                    continue
                n_last = layers[-1][0]
                best = layers[0]
                for li, f in layers:
                    if feats[f]["value"] is not None and better(feats[f]["value"], feats[best[1]]["value"]):
                        best = (li, f)
                l0, lb, ll = feats["L0"], feats[best[1]], feats[f"L{n_last}"]
                pre_last = next((feats[f] for f in feats if f.endswith("_pre")), None)
                tres["kinds"][kind] = {
                    "curve": {f: feats[f]["value"] for _, f in layers},
                    "L0": l0, "best_layer": best[1], "best_frac": best[0] / n_last if n_last else 0.0, "best": lb,
                    "last_layer": f"L{n_last}", "last": ll, "last_pre": pre_last, "pre_proj": feats.get("pre_proj"),
                    "best_minus_L0": lb["value"] - l0["value"], "last_minus_L0": ll["value"] - l0["value"],
                    "best_rel_L0": (lb["value"] - l0["value"]) / abs(l0["value"]) if l0["value"] else None,
                }
            mres[t] = tres
        res["models"][mdir.name] = mres
    if noise is not None:
        for mdir in sorted(p for p in noise.iterdir() if p.is_dir()):
            main_dir = run / mdir.name
            for t in TASKS:
                fn, fm = mdir / t / "metrics.json", main_dir / t / "metrics.json"
                if not (fn.exists() and fm.exists()):
                    continue
                mn, mm = json.loads(fn.read_text()), json.loads(fm.read_text())
                metric, _ = headline_of(mm)
                for kind in KINDS:
                    diffs = []
                    for name, p in mm["probes"].items():
                        if p["kind"] != kind or name not in mn["probes"]:
                            continue
                        a, b = p["eval"].get(metric), mn["probes"][name]["eval"].get(metric)
                        if a is not None and b is not None:
                            diffs.append(abs(a - b))
                    if diffs:
                        res["noise"].setdefault(mdir.name, {}).setdefault(t, {})[kind] = {
                            "metric": metric, "n_probes": len(diffs), "max": max(diffs), "median": st.median(diffs)}
    return res


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    L = ["# E2 汇总（RUN-0029；管理员脚本 scripts/mgr/e2_summary.py，只读 metrics.json）", ""]
    noise_max = {}
    for model, tasks in res["noise"].items():
        for t, kinds in tasks.items():
            for k, d in kinds.items():
                noise_max[(t, k)] = d["max"]
    if res["noise"]:
        L += ["## 噪声底（0.8B 同配置重跑，逐探针标题指标的绝对差）", "", "| 任务 | 种类 | 指标 | 探针数 | 最大 | 中位数 |", "|---|---|---|---:|---:|---:|"]
        for model, tasks in res["noise"].items():
            for t, kinds in tasks.items():
                for k, d in kinds.items():
                    L.append(f"| {t} | {k} | {d['metric']} | {d['n_probes']} | {d['max']:.4f} | {d['median']:.4f} |")
        L.append("")
    L += ["## 逐层读数（标题指标；best 是全部层里最好的一层；frac = 层号 / N）", "",
          "| 模型 | 任务 | 种类 | 指标 | L0 | 最好的层（frac） | 最好 | 最后一层 | L_N_pre | pre_proj | 最好 − L0 | 相对 | ≥ 2×噪声底？ | post_proj 核对 |",
          "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for model, tasks in res["models"].items():
        for t, tr in tasks.items():
            ppc = tr.get("post_proj_check")
            ppc_s = "—" if ppc is None else (ppc.get("equal") if isinstance(ppc, dict) and "equal" in ppc else json.dumps(ppc, ensure_ascii=False)[:40])
            for k, d in tr["kinds"].items():
                nf = noise_max.get((t, k))
                sig = "—" if nf is None else ("是" if abs(d["best_minus_L0"]) >= 2 * nf else "否")
                rel_s = "—" if d["best_rel_L0"] is None else "{:+.1f}%".format(d["best_rel_L0"] * 100)
                L.append(f"| {model} | {t} | {k} | {tr['metric']}{tr['direction']} | {fmt(d['L0']['value'])} | {d['best_layer']}（{d['best_frac']:.2f}） | "
                         f"{fmt(d['best']['value'])} | {fmt(d['last']['value'])} | {fmt((d['last_pre'] or {}).get('value'))} | "
                         f"{fmt((d['pre_proj'] or {}).get('value'))} | {d['best_minus_L0']:+.3f} | "
                         f"{rel_s} | {sig} | {ppc_s} |")
    L += ["", "## 按位置分列（D53-3；linear 探针；L0 / 最好的层 / 最后一层）", "",
          "| 模型 | 任务 | 区域 | L0 | 最好 | 最后 |", "|---|---|---|---:|---:|---:|"]
    for model, tasks in res["models"].items():
        for t, tr in tasks.items():
            d = tr["kinds"].get("linear")
            if not d:
                continue
            for split in ("half", "raster_quartile"):
                for r in (d["L0"]["pos"].get(split) or {}):
                    L.append(f"| {model} | {t} | {split}:{r} | {fmt(d['L0']['pos'][split].get(r))} | "
                             f"{fmt(d['best']['pos'].get(split, {}).get(r))} | {fmt(d['last']['pos'].get(split, {}).get(r))} |")
    L += ["", "## 逐层曲线（linear；标题指标）", ""]
    for model, tasks in res["models"].items():
        for t, tr in tasks.items():
            d = tr["kinds"].get("linear")
            if d:
                L.append(f"- {model} / {t}：" + "，".join(f"{f}={fmt(v)}" for f, v in d["curve"].items()))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--noise")
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    a = ap.parse_args(argv)
    res = summarize(Path(a.run), Path(a.noise) if a.noise else None)
    md = markdown(res)
    if a.out_md:
        Path(a.out_md).write_text(md, encoding="utf-8")
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
