#!/usr/bin/env python
"""管理员用：E3 评测运行（eval_e3b、eval_e3a、run_e1 --checkpoint、抽查比对）的主读数汇总。只读各 metrics.json，不重算。

用法：python scripts/mgr/e3_summary.py <E3 评测运行目录> --model qwen3.5-2b --seed 0 --base-e1 experiments/E1/0027_e1_main/qwen3.5-2b
输出 markdown 到 stdout（--json 另写一份 json）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def fmt(v, nd=3):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def ci(m, nd=3):
    if not m or m.get("value") is None:
        return "—"
    c = m.get("ci") or {}
    return f"{fmt(m['value'], nd)} [{fmt(c.get('low'), nd)}, {fmt(c.get('high'), nd)}]" if c else fmt(m["value"], nd)


def grid_rows(run: Path, model: str, seed: int):
    out = []
    for v in ("e3b", "c1prime", "warmup"):
        for t in ("refseg", "depth"):
            d = load(run / "eval" / v / model / f"seed{seed}" / t / "metrics.json")
            if d is None:
                continue
            h = d["headline"]
            sg = d.get("single", {})
            valid = None
            for key in ("parse", "validity"):
                if isinstance(sg.get(key), dict):
                    valid = sg[key].get("valid_rate")
            preds = run / "eval" / v / model / f"seed{seed}" / t / "predictions.jsonl"
            if valid is None and preds.is_file():
                rows = [json.loads(l) for l in preds.open(encoding="utf-8")]
                valid = sum(1 for r in rows if r.get("failure") is None) / max(len(rows), 1)
            bil = (sg.get("bilinear") or {})
            if t == "refseg":
                out.append((v, t, f"cIoU {ci(h.get('cIoU'))}", f"gIoU {ci(h.get('gIoU'))}",
                            f"bilinear cIoU {ci(bil.get('cIoU'))}", fmt(valid)))
            else:
                out.append((v, t, f"AbsRel {ci(h.get('abs_rel'))}", f"δ1 {ci(h.get('d1'))}",
                            f"δ1 strict {ci(h.get('d1_strict'))}", fmt(valid)))
    return out


def dense_rows(run: Path, model: str, seed: int):
    keys = {"refseg": ("cIoU", "gIoU"), "depth": ("abs_rel", "d1"), "semseg": ("miou", "pixel_acc"),
            "normals": ("mean", "within_11.25")}
    out = []
    for v in ("full", "c1", "c2"):
        for t, (a, b) in keys.items():
            d = load(run / "eval" / "e3a" / v / model / f"seed{seed}" / t / "metrics.json")
            if d is None:
                continue
            h = d["headline"]
            leak = d.get("leak_checks") or {}
            lk = []
            for name, rec in leak.items():
                pm = (rec or {}).get("paired_normal_minus_check") or {}
                first = next(iter(pm.values()), None) if pm else None
                if first:
                    lk.append(f"{name}: {fmt(first.get('estimate'))} [{fmt(first.get('low'))}, {fmt(first.get('high'))}]")
            out.append((v, t, f"{a} {ci(h.get(a))}", f"{b} {ci(h.get(b))}", "; ".join(lk) or "—"))
    return out


def e1_headline(d):
    """E1 metrics.json → (指标名, 严格值, 有效率)。三列：strict / valid / conditional。"""
    if d is None:
        return None
    h = d.get("headline") or []
    first = h[0] if isinstance(h, list) and h else (h if isinstance(h, dict) else None)
    if not first:
        return None
    valid = (d.get("valid") or {}).get("rate") if isinstance(d.get("valid"), dict) else None
    return first.get("name"), first.get("value"), valid


def ttext_rows(run: Path, model: str, seed: int, base: Path | None):
    tasks = ["refexp_box", "refexp_poly", "grid16", "count", "ovd", "rel_depth_nyu", "rel_depth_blink", "spatial_cvbench",
             "spatial_embspatial", "rel_depth_nyu__letter_swap", "rel_depth_nyu__debiased", "rel_depth_blink__debiased",
             "spatial_cvbench__debiased", "spatial_embspatial__debiased", "rel_depth_nyu__free", "rel_depth_blink__free",
             "spatial_cvbench__free", "spatial_embspatial__free"]
    variants = [v for v in ("e3b", "c3", "e3a") if (run / "t_text" / v / f"seed{seed}" / model).is_dir()]
    out = []
    for t in tasks:
        b = e1_headline(load(base / t / "metrics.json")) if base else None
        row = [t, (b[0] if b else "—"), fmt(b[1]) if b else "—"]
        for v in variants:
            r = e1_headline(load(run / "t_text" / v / f"seed{seed}" / model / t / "metrics.json"))
            if r is None:
                row.append("—")
            else:
                delta = (r[1] - b[1]) if (b and r[1] is not None and b[1] is not None) else None
                row.append(f"{fmt(r[1])} ({'+' if delta is not None and delta >= 0 else ''}{fmt(delta)})")
        out.append(row)
    return variants, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-e1", help="RUN-0027 的 <模型> 目录（T 文本的基座读数）")
    a = ap.parse_args()
    run = Path(a.run)
    base = Path(a.base_e1) if a.base_e1 else None
    print(f"# E3 评测汇总：{run.name}，{a.model}，种子 {a.seed}\n")
    print("## 网格单轮（E3b / C1′；guided 上采样）\n")
    print("| 变体 | 任务 | 主指标 | 次指标 | 另一读法 | 有效率 |\n|---|---|---|---|---|---|")
    for r in grid_rows(run, a.model, a.seed):
        print("| " + " | ".join(r) + " |")
    print("\n## 稠密（E3a full / C1 / C2；guided）\n")
    print("| 变体 | 任务 | 主指标 | 次指标 | 泄漏检查（正常 − 检查，配对差） |\n|---|---|---|---|---|")
    for r in dense_rows(run, a.model, a.seed):
        print("| " + " | ".join(r) + " |")
    variants, rows = ttext_rows(run, a.model, a.seed, base)
    print("\n## T 文本（严格读数；括号里是减基座 RUN-0027）\n")
    print("| 任务 | 指标 | 基座 | " + " | ".join(variants) + " |\n|---|---|---|" + "---|" * len(variants))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    sc = run / "spotcheck"
    if sc.is_dir():
        print("\n## 冻结对照的抽查\n")
        for f in sorted(sc.glob("*.md")):
            print(f"- {f.stem}：{f.read_text(encoding='utf-8').splitlines()[0].lstrip('# ')}")


if __name__ == "__main__":
    main()
