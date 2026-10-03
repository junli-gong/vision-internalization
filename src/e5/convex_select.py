"""凸起先验基线的幅度 A：只在 val 划分上网格搜索选定（D59-4；DEV-0028），以及用冻结的 A 复算项目负责人的表。

凸起先验的定义、冻结的 A（src/synth/relief.CONVEX_PRIOR_AMPLITUDE）与派生基线 copy_convex 都在 src/synth/relief.py；这里只有选法。
只用生成器给出的基线（copy 的网格与目标的起伏），不涉及任何模型，也不读外部预测。

选法（select）
- 样本：val 划分（不看 test），序号 0..B·n − 1、档 = 序号 mod B（与 E5 评测集同一规则，src/e5/protocol.py），n = SELECTION_N_PER_BIN；
  只留标题口径的样本（protocol.in_headline：a 为 30° 以上三档，b 为环绕移动）。
- 对 CONVEX_AMPLITUDE_SEARCH 里的每个 A，算 copy_convex 的起伏误差，口径与门槛完全相同（scoring.score → relief_population →
  relief_strict_value：目标起伏全平的样本不计，没有公共前景格记 RELIEF_FAIL_VALUE）。
- 选标题口径上严格均值最小的 A；同分取较小的 A。a、b 各选各的（两类的物体大小、起伏分布不同；一个幅度参数是指每个任务一个）。

报告（report）：用冻结的 A，在指定划分上按档列平面（copy_flat）、真值仿射（gt_affine_upper）、凸起先验（copy_convex）的
起伏误差均值与中位数，以及"凸起先验胜过平面的样本比例"（逐样本严格小于）。项目负责人的表是 a、test 划分、每档 80 条。

用法（仓库根；纯 CPU，几分钟）：
    python -m src.e5.convex_select select [--task a b] [--n-per-bin 200]
    python -m src.e5.convex_select report [--task a b] [--split test] [--n-per-bin 80]
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable, Sequence

import numpy as np

from src.e5 import scoring as SC
from src.e5.protocol import groups_of, in_headline, num_bins, sample_id
from src.synth import generate
from src.synth.relief import (
    CONVEX_AMPLITUDE_SEARCH,
    CONVEX_BASELINE,
    CONVEX_PRIOR_AMPLITUDE,
    RELIEF_TASKS,
    derived_baselines,
)
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

SELECTION_SPLIT = "val"
SELECTION_N_PER_BIN = 200
LEAD_TABLE_SPLIT = "test"
LEAD_TABLE_N_PER_BIN = 80
FLAT, GT_AFFINE = SC.FLAT_BASELINE, SC.GT_AFFINE


def samples(task: str, split: str, n_per_bin: int, *, headline_only: bool = True) -> list[Sample]:
    """划分 split 的序号 0..B·n − 1（档 = 序号 mod B）上的样本；headline_only 时只留标题口径的。"""
    t = canonical_task(task)
    b = num_bins(t)
    out = []
    for i in range(b * n_per_bin):
        if headline_only and t == "rotation3d" and not in_headline(t, i % b, {}):
            continue  # a 的标题口径只看档，不必生成
        s = generate(t, split, i, bin=i % b)
        if headline_only and not in_headline(t, i % b, groups_of(t, s.meta)):
            continue
        out.append(s)
    return out


def _record(sample: Sample, pred: SC.Pred) -> dict[str, Any]:
    return {"id": sample_id(sample.task, sample.split, sample.index), "bin": int(sample.meta["difficulty"]["bin"]),
            "score": SC.score(sample, pred, relief_expected=True)}


def convex_records(ss: Sequence[Sample], amplitude: int) -> list[dict[str, Any]]:
    out = []
    for s in ss:
        src, codes = derived_baselines(s, amplitude)[CONVEX_BASELINE]
        out.append(_record(s, SC.Pred(np.asarray(src.grid, np.uint8), np.asarray(src.mask, bool), codes)))
    return out


def baseline_relief_records(ss: Sequence[Sample], name: str) -> list[dict[str, Any]]:
    return [_record(s, SC.baseline_pred(s, name)) for s in ss]


def strict_values(recs: Iterable[dict[str, Any]]) -> dict[str, float]:
    """id → 严格读数（只含起伏总体里的样本，口径同门槛）。"""
    return {r["id"]: SC.relief_strict_value(r) for r in SC.relief_population(list(recs))}


def select(task: str, n_per_bin: int = SELECTION_N_PER_BIN, grid: Sequence[int] = CONVEX_AMPLITUDE_SEARCH,
           split: str = SELECTION_SPLIT) -> dict[str, Any]:
    """在 split（默认 val）的标题口径上网格搜索 A（模块说明）。返回每个 A 的严格均值（另按档）与选中的 A。"""
    t = canonical_task(task)
    if t not in RELIEF_TASKS:
        raise ValueError(f"{t} 没有起伏")
    if split == "test":
        raise ValueError("A 只在 val 上选（D59-4）")
    ss = samples(t, split, n_per_bin)
    bins = {s.index: int(s.meta["difficulty"]["bin"]) for s in ss}
    table, by_bin = {}, {}
    ids = None
    for a in grid:
        vals = strict_values(convex_records(ss, a))
        if ids is None:
            ids = sorted(vals)
        elif sorted(vals) != ids:  # pragma: no cover - 总体只取决于目标
            raise RuntimeError("不同 A 的起伏总体不同")
        table[int(a)] = float(np.mean(list(vals.values())))
        per: dict[int, list[float]] = {}
        for sid, v in vals.items():
            per.setdefault(bins[int(sid.rsplit("/", 1)[1])], []).append(v)
        by_bin[int(a)] = {str(b): float(np.mean(v)) for b, v in sorted(per.items())}
    chosen = min(table, key=lambda a: (table[a], a))
    return {"task": t, "split": split, "n_per_bin": n_per_bin, "n_samples": len(ss), "n_population": len(ids or []),
            "criterion": "标题口径上 copy_convex 的起伏误差严格均值最小（同分取较小的 A）",
            "table": table, "by_bin": by_bin, "chosen": chosen, "frozen": CONVEX_PRIOR_AMPLITUDE[t]}


def report(task: str, split: str = LEAD_TABLE_SPLIT, n_per_bin: int = LEAD_TABLE_N_PER_BIN,
           amplitude: int | None = None) -> dict[str, Any]:
    """用冻结的 A（或给定的 amplitude）在 split 上按档复算：平面、真值仿射、凸起先验的起伏误差，及凸起先验胜过平面的比例。
    a 列全部四档；b 只列环绕移动（标题口径），也按档。"""
    t = canonical_task(task)
    a = CONVEX_PRIOR_AMPLITUDE[t] if amplitude is None else int(amplitude)
    ss = samples(t, split, n_per_bin, headline_only=(t == "viewpoint"))
    flat = strict_values(baseline_relief_records(ss, FLAT))
    ga = strict_values(baseline_relief_records(ss, GT_AFFINE))
    cv = strict_values(convex_records(ss, a))
    rows = []
    for b in range(num_bins(t)):
        ids = [sample_id(t, split, s.index) for s in ss if int(s.meta["difficulty"]["bin"]) == b]
        ids = [i for i in ids if i in cv]
        if not ids:
            continue
        f, g, c = (np.array([d[i] for i in ids]) for d in (flat, ga, cv))
        rows.append({"bin": b, "range": ss[0].meta["difficulty"]["bins"][b], "n": len(ids),
                     "flat_mean": float(f.mean()), "gt_affine_mean": float(g.mean()), "convex_mean": float(c.mean()),
                     "flat_median": float(np.median(f)), "gt_affine_median": float(np.median(g)),
                     "convex_median": float(np.median(c)), "convex_beats_flat": float(np.mean(c < f))})
    return {"task": t, "split": split, "n_per_bin": n_per_bin, "amplitude": a, "rows": rows}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "report"])
    ap.add_argument("--task", nargs="+", default=list(RELIEF_TASKS))
    ap.add_argument("--n-per-bin", type=int)
    ap.add_argument("--split", default=None, help="report 的划分（默认 test）；select 固定 val")
    a = ap.parse_args(argv)
    for t in a.task:
        if a.mode == "select":
            out = select(t, a.n_per_bin or SELECTION_N_PER_BIN)
        else:
            out = report(t, a.split or LEAD_TABLE_SPLIT, a.n_per_bin or LEAD_TABLE_N_PER_BIN)
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
