#!/usr/bin/env python
"""管理员用：E5 主比较的 H5 分析（E5 预注册 §4、§8 第 2、4 项、§8a、§8b、§8c 的 D99-2 补充、§8d；N39）。只供数，结论文字由管理员写。

读的东西（都不重算模型输出）：
- 各运行叶目录 <根>/trained/<起点>/<模型>/<档>/seed<k>/<任务>__structured[__unmerged]/{predictions.jsonl, metrics.json, config.yaml}；
  档是 n0、n1000、n4000、n16000、n64000、n4000_abcd。同一个（起点、档、种子）只能出现在一个根里（D91-1：同组同机）。
- 平凡基线的逐样本记录：RUN-0032 的 <基线根>/<基线>/<任务>/predictions.jsonl（生成器自身的性质，与系统无关）。
  逐个运行核对它与基线根的 test 集相同（config.yaml 的 sample_ids_sha256、task_generator_version、synth_generator_version），
  并核对按它算出的标题口径基线均值与运行 metrics.json 的 baselines_headline 相同。

算的东西（§8d，写于读到任何 E5 训练后 test 分数之前）：
1. 显著：每类每档 src.stats.is_significant（D10(9) 三条规则），A = E3b+C3，B = C3+预热，输入 (样本, 种子)，统计量取均值；
   读数是标题指标（a：严格起伏误差，越低越好，只在目标起伏有定义的样本上；b–e：网格 mask IoU，越高越好），样本取标题口径。
2. 先决条件与判定条件（按类；种子之间怎么合见 N39，三种口径都算）：
   (a) seed_mean：每条样本对种子取平均，再用 src.e5.scoring.gate 与基线配对；
   (b) all_seeds：每个运行自己的 gate 都通过（metrics.json 的 gate.imagining 为 true）；
   (c) majority：至少 2 个种子通过。
   c、d、e 的"通过" = 先决条件；a、b = 先决条件与判定条件都通过（gate.imagining）。
3. 每档：同时满足 1 与 2 的类 ≥ 3 → 这一档满足。两档都满足 = 成立，一档 = 部分支持，都不满足 = 不成立（§8 第 4 项）。
4. 截距与斜率：D = 种子平均[(E − C)@N − (E − C)@0]，按越大越好取向（a 取起伏误差的相反数）；同一组样本下标同时用于 4 个读数与全部种子
   的配对 bootstrap，95%，下界 > 0 算斜率的证据。
5. 未合并（D99-2）：12 个运行换成 __structured__unmerged 叶目录，1–4 照算。
6. 噪声底：每个运行、每类，Δ = 未合并 − 合并 的标题读数；起伏取 a、b 的 |Δ| 最大值，IoU 取五类的 |Δ| 最大值。
7. 描述：对基座+预热；E3b 起点（种子 0、2）；a–d → e；16K；64K（种子 0，单种子配对 bootstrap）。

用法：
  python scripts/mgr/e5_h5.py --roots experiments/E5/0051_e5_main_hpcgpu07 experiments/E5/0051_e5_main_hpcgpu06 \\
      --baselines experiments/E5/0032_e5_test/baseline --model qwen3.5-2b --out docs/analysis/e5_h5
  输出 <out>.json 与 <out>.md。--n-resamples 默认 src/constants 的 BOOTSTRAP_N_RESAMPLES。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED  # noqa: E402
from src.e5.protocol import E5_TASKS, TASK_LETTER  # noqa: E402
from src.e5.scoring import RELIEF_FAIL_VALUE, gate, relief_strict_value  # noqa: E402
from src.e5.val_select import HEADLINE_METRIC  # noqa: E402
from src.stats import _percentile_ci, _resample_indices, is_significant, paired_bootstrap_ci  # noqa: E402

E, C, B, E3B = "e3b_c3", "c3_warmup", "base_warmup", "e3b"
H5_TIERS = ("n1000", "n4000")
SEEDS = (0, 1, 2)
E3B_SEEDS = (0, 2)  # §8b：E3b 种子 1 在 A100 上没过起点的合并核对（RUN-0048）
RULES = ("seed_mean", "all_seeds", "majority")
UNMERGED = "__unmerged"
DATA_KEYS = ("sample_ids_sha256", "task_generator_version", "synth_generator_version")


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(np.asarray(xs, np.float64)))


def load_yaml(p: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(p.read_text())


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class Leaf:
    """一个运行在一个任务上的评测叶目录。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.metrics = json.loads((path / "metrics.json").read_text())
        if not self.metrics.get("complete", False):
            raise ValueError(f"{path}：metrics.json 的 complete 不是 true")
        self.recs = {r["id"]: r for r in read_jsonl(path / "predictions.jsonl")}
        if len(self.recs) != int(self.metrics["n"]):
            raise ValueError(f"{path}：predictions.jsonl 有 {len(self.recs)} 条，metrics.json 的 n 是 {self.metrics['n']}")
        data = load_yaml(path / "config.yaml")["data"]
        self.data = {k: data.get(k) for k in DATA_KEYS}

    @property
    def imagining(self) -> bool | None:
        return (self.metrics.get("gate") or {}).get("imagining")


def find_leaves(roots: Sequence[Path], model: str) -> dict[tuple[str, str, int, str, str], Leaf]:
    """(起点, 档, 种子, 任务, 口径) → 叶目录。口径 '' = 正式（合并），'__unmerged' = D99-2。"""
    out: dict[tuple[str, str, int, str, str], Leaf] = {}
    where: dict[tuple[str, str, int], Path] = {}
    for root in roots:
        for seed_dir in sorted((root / "trained").glob(f"*/{model}/*/seed*")):
            start, tag, seed = seed_dir.parts[-4], seed_dir.parts[-2], int(seed_dir.name.removeprefix("seed"))
            key3 = (start, tag, seed)
            if key3 in where and where[key3] != root:
                raise ValueError(f"{key3} 同时出现在 {where[key3]} 与 {root}（D91-1：同组同机）")
            where[key3] = root
            for leaf in sorted(seed_dir.glob("*__structured*")):
                name = leaf.name
                task, _, rest = name.partition("__structured")
                if rest not in ("", UNMERGED) or task not in E5_TASKS or not (leaf / "metrics.json").exists():
                    continue
                out[(start, tag, seed, task, rest)] = Leaf(leaf)
    return out


class Baselines:
    """RUN-0032 的平凡基线逐样本记录；按任务取，与运行的 test 集逐项核对。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache: dict[str, tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]] = {}

    def for_task(self, task: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        if task not in self.cache:
            recs, data = {}, None
            for d in sorted(self.root.glob(f"*/{task}")):
                recs[d.parent.name] = read_jsonl(d / "predictions.jsonl")
                dd = load_yaml(d / "config.yaml")["data"]
                dd = {k: dd.get(k) for k in DATA_KEYS}
                if data is not None and dd != data:
                    raise ValueError(f"{task}：基线 {d.parent.name} 的 test 集与其他基线不同：{dd} 对 {data}")
                data = dd
            if not recs:
                raise ValueError(f"{self.root} 下没有 {task} 的基线")
            self.cache[task] = (recs, data)
        return self.cache[task]

    def check(self, leaf: Leaf, task: str) -> None:
        recs, data = self.for_task(task)
        if leaf.data != data:
            raise ValueError(f"{leaf.path}：test 集与基线不同：{leaf.data} 对 {data}")
        head_ids = {i for i, r in leaf.recs.items() if r["headline"]}
        want = leaf.metrics.get("baselines_headline") or {}
        roles = leaf.metrics.get("baseline_roles") or {}
        for name in roles:
            if name not in recs:
                raise ValueError(f"{task}：运行的基线角色里有 {name}，基线根里没有")
        if not want:
            raise ValueError(f"{leaf.path}：metrics.json 没有 baselines_headline，核对不了基线")
        for name, block in want.items():
            head = [r for r in recs[name] if r["id"] in head_ids]
            if len(head) != len(head_ids):
                raise ValueError(f"{leaf.path}：基线 {name} 的标题样本 {len(head)} 条，运行是 {len(head_ids)} 条")
            ref = _find_number(block, ("mask_iou_grid", "strict"))
            got = mean([float(r["score"]["iou_grid"]) for r in head])
            if ref is None or abs(got - ref) > 1e-9:
                raise ValueError(f"{leaf.path}：基线 {name} 的标题口径网格 IoU {got} 与 metrics.json 的 {ref} 不同")
            ref = _find_number(block, ("relief_error", "strict"))
            if ref is not None:
                pop = [r for r in head if r["score"].get("relief") is not None and not r["score"]["relief"]["target_flat"]]
                got = mean([relief_strict_value(r) for r in pop])
                if abs(got - ref) > 1e-9:
                    raise ValueError(f"{leaf.path}：基线 {name} 的标题口径起伏误差 {got} 与 metrics.json 的 {ref} 不同")


def _find_number(block: Any, path: Sequence[str]) -> float | None:
    cur = block
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return None if cur is None else float(cur)


def headline_ids(leaf: Leaf, task: str) -> list[str]:
    """标题口径的样本 id（按 id 排序）；a 只取目标起伏有定义的（目标的性质，与系统无关）。"""
    ids = [i for i, r in leaf.recs.items() if r["headline"]]
    if HEADLINE_METRIC[task] == "relief":
        ids = [i for i in ids if not leaf.recs[i]["score"]["relief"]["target_flat"]]
    return sorted(ids)


def value(rec: Mapping[str, Any], task: str) -> float:
    """标题读数：a 是严格起伏误差（无效记 RELIEF_FAIL_VALUE），其余是网格 mask IoU（严格）。"""
    if HEADLINE_METRIC[task] == "relief":
        return relief_strict_value(rec) if rec["score"].get("relief") is not None else RELIEF_FAIL_VALUE
    return float(rec["score"]["iou_grid"])


class Analysis:
    def __init__(self, leaves: Mapping[tuple, Leaf], baselines: Baselines, n_resamples: int) -> None:
        self.leaves = leaves
        self.baselines = baselines
        self.n_resamples = n_resamples
        self.checked: set[Path] = set()
        self.notes: list[str] = []

    def leaf(self, start: str, tag: str, seed: int, task: str, merge: str = "") -> Leaf | None:
        lf = self.leaves.get((start, tag, seed, task, merge))
        if lf is not None and lf.path not in self.checked:
            self.baselines.check(lf, task)
            self.checked.add(lf.path)
        return lf

    def matrix(self, start: str, tag: str, seeds: Sequence[int], task: str, ids: Sequence[str], merge: str = "") -> np.ndarray | None:
        """(样本, 种子) 的标题读数；任何一个运行缺失就返回 None。"""
        cols = []
        for s in seeds:
            lf = self.leaf(start, tag, s, task, merge if tag != "n0" else "")
            if lf is None:
                return None
            missing = [i for i in ids if i not in lf.recs]
            if missing:
                raise ValueError(f"{lf.path} 缺少 {len(missing)} 条标题样本，例如 {missing[:3]}")
            cols.append([value(lf.recs[i], task) for i in ids])
        return np.asarray(cols, np.float64).T

    def ids_for(self, task: str, tag: str, seeds: Sequence[int], starts: Sequence[str], merge: str = "") -> list[str] | None:
        """所有参与比较的运行共有的标题样本；各运行的标题集合必须相同。"""
        sets = []
        for st in starts:
            for s in seeds:
                lf = self.leaf(st, tag, s, task, merge if tag != "n0" else "")
                if lf is None:
                    return None
                sets.append(tuple(headline_ids(lf, task)))
        if len(set(sets)) != 1:
            raise ValueError(f"{task} {tag}：各运行的标题样本集合不同")
        return list(sets[0])

    def compare(self, a_start: str, b_start: str, tag: str, task: str, seeds: Sequence[int], merge: str = "") -> dict[str, Any] | None:
        """A 对 B 的显著性（§8d 第 1 项）；种子 < 2 时只给单种子的配对 bootstrap。"""
        ids = self.ids_for(task, tag, seeds, (a_start, b_start), merge)
        if ids is None:
            return None
        a = self.matrix(a_start, tag, seeds, task, ids, merge)
        b = self.matrix(b_start, tag, seeds, task, ids, merge)
        hib = HEADLINE_METRIC[task] != "relief"
        out: dict[str, Any] = {"task": task, "letter": TASK_LETTER[task], "tag": tag, "seeds": list(seeds), "n": len(ids),
                               "metric": HEADLINE_METRIC[task], "higher_is_better": hib, "a": a_start, "b": b_start,
                               "merge": "unmerged" if merge else "merged",
                               "seed_a": [float(x) for x in a.mean(axis=0)], "seed_b": [float(x) for x in b.mean(axis=0)],
                               "mean_a": float(a.mean()), "mean_b": float(b.mean())}
        if len(seeds) >= 2:
            r = is_significant(a, b, statistic=lambda x: float(np.mean(x)), higher_is_better=hib, n_resamples=self.n_resamples)
            out.update(diff=r.mean_diff, ci=[r.paired_ci.low, r.paired_ci.high], pooled_std=r.pooled_std,
                       exceeds_std_rule=r.exceeds_std_rule, ci_excludes_zero=r.ci_excludes_zero, same_sign=r.same_sign,
                       significant=r.significant, better=r.significantly_better, worse=r.significantly_worse)
        else:
            ci = paired_bootstrap_ci(a[:, 0], b[:, 0], n_resamples=self.n_resamples)
            out.update(diff=float(a[:, 0].mean() - b[:, 0].mean()), ci=None if ci is None else [ci.low, ci.high],
                       significant=None, better=None, worse=None, note="单种子，只作描述")
        return out

    def gates(self, start: str, tag: str, task: str, seeds: Sequence[int], merge: str = "") -> dict[str, Any] | None:
        """先决条件与判定条件的三种口径（N39）。"""
        lfs = [self.leaf(start, tag, s, task, merge) for s in seeds]
        if any(lf is None for lf in lfs):
            return None
        per_seed = [lf.imagining for lf in lfs]
        recs, _ = self.baselines.for_task(task)
        roles = lfs[0].metrics["baseline_roles"]
        ids = sorted(i for i, r in lfs[0].recs.items() if r["headline"])
        avg = []
        for i in ids:
            r0 = lfs[0].recs[i]
            sc = {"iou_grid": mean([float(lf.recs[i]["score"]["iou_grid"]) for lf in lfs]), "relief": None}
            if r0["score"].get("relief") is not None:
                errs = [relief_strict_value(lf.recs[i]) for lf in lfs]
                sc["relief"] = {"error": mean(errs), "target_flat": r0["score"]["relief"]["target_flat"], "reason": None}
            avg.append({"id": i, "headline": True, "score": sc})
        g = gate(task, avg, {n: recs[n] for n in roles}, roles, n_resamples=self.n_resamples)
        pre, jud = g.get("prerequisite") or {}, g.get("judgment") or {}
        return {"seed_mean": g["imagining"], "all_seeds": all(p is True for p in per_seed),
                "majority": sum(p is True for p in per_seed) >= 2, "per_seed": per_seed,
                "seed_mean_detail": {"prerequisite_pass": pre.get("pass"), "prerequisite_diff": pre.get("diff"),
                                     "prerequisite_ci": _ci_pair(pre.get("diff_ci")), "strongest": pre.get("strongest"),
                                     "judgment_pass": jud.get("pass"),
                                     "judgment": {n: {"diff": b.get("diff"), "ci": _ci_pair(b.get("diff_ci")), "beats": b.get("beats")}
                                                  for n, b in (jud.get("baselines") or {}).items()}}}

    def slope(self, tag: str, task: str, merge: str = "") -> dict[str, Any] | None:
        """截距与斜率（§8d 第 4 项）。"""
        ids = self.ids_for(task, tag, SEEDS, (E, C), merge)
        ids0 = self.ids_for(task, "n0", SEEDS, (E, C))
        if ids is None or ids0 is None:
            return None
        if ids != ids0:
            raise ValueError(f"{task}：{tag} 与 n0 的标题样本不同")
        sign = 1.0 if HEADLINE_METRIC[task] != "relief" else -1.0
        eN, cN = self.matrix(E, tag, SEEDS, task, ids, merge), self.matrix(C, tag, SEEDS, task, ids, merge)
        e0, c0 = self.matrix(E, "n0", SEEDS, task, ids), self.matrix(C, "n0", SEEDS, task, ids)

        def stat(sel: np.ndarray | slice) -> float:
            dN = eN[sel].mean(axis=0) - cN[sel].mean(axis=0)
            d0 = e0[sel].mean(axis=0) - c0[sel].mean(axis=0)
            return float(sign * np.mean(dN - d0))

        point = stat(slice(None))
        idx = _resample_indices(len(ids), self.n_resamples, BOOTSTRAP_SEED)
        boot = np.array([stat(i) for i in idx], np.float64)
        ci = _percentile_ci(point, boot, BOOTSTRAP_CI_LEVEL, self.n_resamples)
        d0 = float(sign * np.mean(e0.mean(axis=0) - c0.mean(axis=0)))
        dN = float(sign * np.mean(eN.mean(axis=0) - cN.mean(axis=0)))
        return {"task": task, "tag": tag, "merge": "unmerged" if merge else "merged", "delta_0": d0, "delta_N": dN,
                "D": point, "ci": [ci.low, ci.high], "slope_evidence": bool(ci.low > 0), "orientation": "越大越好（a 取起伏误差的相反数）"}

    def h5(self, merge: str = "") -> dict[str, Any]:
        tiers: dict[str, Any] = {}
        for tag in H5_TIERS:
            rows = []
            for task in E5_TASKS:
                cmp = self.compare(E, C, tag, task, SEEDS, merge)
                g = self.gates(E, tag, task, SEEDS, merge)
                if cmp is None or g is None:
                    rows.append({"task": task, "letter": TASK_LETTER[task], "missing": True})
                    continue
                row = {**cmp, "gates": g, "counts": {r: bool(cmp["better"] and g[r] is True) for r in RULES}}
                rows.append(row)
            complete = all(not r.get("missing") for r in rows)
            counts = {r: sum(1 for x in rows if not x.get("missing") and x["counts"][r]) for r in RULES}
            tiers[tag] = {"rows": rows, "complete": complete, "n_classes": counts,
                          "satisfied": {r: (counts[r] >= 3) if complete else None for r in RULES},
                          "slopes": [self.slope(tag, t, merge) for t in E5_TASKS]}
        verdict = {}
        for r in RULES:
            sat = [tiers[t]["satisfied"][r] for t in H5_TIERS]
            verdict[r] = None if any(x is None for x in sat) else ("成立" if all(sat) else "部分支持" if any(sat) else "不成立")
        return {"merge": "unmerged" if merge else "merged", "tiers": tiers, "verdict": verdict}

    def noise_floor(self) -> dict[str, Any]:
        rows = []
        for (start, tag, seed, task, merge), lf in sorted(self.leaves.items()):
            if merge != UNMERGED:
                continue
            base = self.leaf(start, tag, seed, task)
            un = self.leaf(start, tag, seed, task, UNMERGED)
            if base is None:
                raise ValueError(f"{un.path} 没有对应的正式叶目录")
            ids = headline_ids(base, task)
            if ids != headline_ids(un, task):
                raise ValueError(f"{un.path}：标题样本与正式评测不同")
            rows.append({"start": start, "tag": tag, "seed": seed, "task": task, "metric": HEADLINE_METRIC[task],
                         "merged": mean([value(base.recs[i], task) for i in ids]),
                         "unmerged": mean([value(un.recs[i], task) for i in ids]),
                         "iou_merged": mean([float(base.recs[i]["score"]["iou_grid"]) for i in sorted(i for i, r in base.recs.items() if r["headline"])]),
                         "iou_unmerged": mean([float(un.recs[i]["score"]["iou_grid"]) for i in sorted(i for i, r in un.recs.items() if r["headline"])])})
        for r in rows:
            r["delta"] = r["unmerged"] - r["merged"]
            r["iou_delta"] = r["iou_unmerged"] - r["iou_merged"]
        relief = [abs(r["delta"]) for r in rows if r["task"] in ("rotation3d",)]
        relief += [abs(self._relief_delta(r)) for r in rows if r["task"] == "viewpoint"]
        iou = [abs(r["iou_delta"]) for r in rows]
        return {"rows": rows, "n_runs": len({(r["start"], r["tag"], r["seed"]) for r in rows}),
                "relief_floor": max(relief) if relief else None, "n_relief": len(relief),
                "iou_floor": max(iou) if iou else None, "n_iou": len(iou)}

    def _relief_delta(self, r: Mapping[str, Any]) -> float:
        """b 的标题读数是 IoU；噪声底的起伏一项另算 b 的严格起伏误差（标题口径、目标起伏有定义的样本）。"""
        base = self.leaf(r["start"], r["tag"], r["seed"], "viewpoint")
        un = self.leaf(r["start"], r["tag"], r["seed"], "viewpoint", UNMERGED)
        ids = sorted(i for i, x in base.recs.items() if x["headline"] and not x["score"]["relief"]["target_flat"])
        return mean([relief_strict_value(un.recs[i]) for i in ids]) - mean([relief_strict_value(base.recs[i]) for i in ids])

    def descriptive(self) -> dict[str, Any]:
        out: dict[str, Any] = {"vs_base_warmup": [], "e3b": [], "transfer_e": [], "n16000": [], "n64000": []}
        for tag in ("n0",) + H5_TIERS:
            for task in E5_TASKS:
                for a in (E, C):
                    r = self.compare(a, B, tag, task, SEEDS)
                    if r:
                        out["vs_base_warmup"].append(r)
                for b in (C, E):
                    r = self.compare(E3B, b, tag, task, E3B_SEEDS)
                    if r:
                        out["e3b"].append(r)
        r = self.compare(E, C, "n4000_abcd", "imagery", SEEDS)
        if r:
            out["transfer_e"].append(r)
        for task in E5_TASKS:
            r = self.compare(E, C, "n16000", task, SEEDS)
            if r:
                out["n16000"].append(r)
            for a, b in ((E, C), (E, B), (C, B)):
                r = self.compare(a, b, "n64000", task, (0,))
                if r:
                    out["n64000"].append(r)
        return out


def _ci_pair(ci: Mapping[str, Any] | None) -> list[float] | None:
    return None if not ci else [float(ci["low"]), float(ci["high"])]


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if nd else f"{x}"
    return str(x)


def fci(ci: Sequence[float] | None) -> str:
    return "—" if not ci else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def render(doc: Mapping[str, Any]) -> str:
    L = ["<!-- 由 scripts/mgr/e5_h5.py 生成；口径见 E5 预注册 §8d 与 N39 -->", "", "# E5 主比较：H5（E3b+C3 对 C3+预热）", ""]
    for h in (doc["h5"], doc.get("h5_unmerged")):
        if not h:
            continue
        L += [f"## H5（{'合并' if h['merge'] == 'merged' else '未合并，D99-2'}）", "",
              "结论（三种口径，N39）：" + "；".join(f"{r} = {h['verdict'][r] or '不完整'}" for r in RULES), ""]
        for tag in H5_TIERS:
            t = h["tiers"][tag]
            L += [f"### {tag}（计入的类数：" + "，".join(f"{r} {t['n_classes'][r]}" for r in RULES) + "）", "",
                  "| 类 | 指标 | n | E3b+C3 | C3+预热 | 差 | 95% CI | 显著更好 | 门槛 seed_mean / all_seeds / majority | 逐种子 gate |",
                  "|---|---|---:|---:|---:|---:|---|---|---|---|"]
            for r in t["rows"]:
                if r.get("missing"):
                    L.append(f"| {r['letter']} | — | — | — | — | — | — | — | 缺运行 | — |")
                    continue
                g = r["gates"]
                L.append(f"| {r['letter']} | {r['metric']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {fmt(r['diff'])} | {fci(r['ci'])} | "
                         f"{fmt(r['better'])} | {fmt(g['seed_mean'])} / {fmt(g['all_seeds'])} / {fmt(g['majority'])} | "
                         f"{' '.join(fmt(p) for p in g['per_seed'])} |")
            L += ["", "截距与斜率（D = 种子平均[(E − C)@N − (E − C)@0]，越大越好）：", "",
                  "| 类 | Δ_0 | Δ_N | D | 95% CI | 斜率的证据 |", "|---|---:|---:|---:|---|---|"]
            for s in t["slopes"]:
                if s:
                    L.append(f"| {TASK_LETTER[s['task']]} | {fmt(s['delta_0'])} | {fmt(s['delta_N'])} | {fmt(s['D'])} | {fci(s['ci'])} | {fmt(s['slope_evidence'])} |")
            L.append("")
    nf = doc.get("noise_floor")
    if nf and nf["rows"]:
        L += ["## 合并与未合并的噪声底（test 规模，§8d 第 6 项）", "",
              f"- 运行数 {nf['n_runs']}；起伏误差 |Δ| 最大值 {fmt(nf['relief_floor'], 4)}（{nf['n_relief']} 个）；"
              f"网格 IoU |Δ| 最大值 {fmt(nf['iou_floor'], 4)}（{nf['n_iou']} 个）", ""]
    d = doc.get("descriptive") or {}
    for key, title in (("vs_base_warmup", "对基座+预热（描述）"), ("e3b", "E3b 起点（§8b，种子 0、2，描述）"),
                       ("transfer_e", "a–d → e 的迁移（4K，只看 e）"), ("n16000", "16K（描述）"), ("n64000", "64K（种子 0，描述）")):
        rows = d.get(key) or []
        if not rows:
            continue
        L += [f"## {title}", "", "| 档 | 类 | A | B | 种子 | A | B | 差 | 95% CI | 显著更好 |", "|---|---|---|---|---|---:|---:|---:|---|---|"]
        for r in rows:
            L.append(f"| {r['tag']} | {r['letter']} | {r['a']} | {r['b']} | {','.join(map(str, r['seeds']))} | {r['mean_a']:.3f} | "
                     f"{r['mean_b']:.3f} | {fmt(r['diff'])} | {fci(r['ci'])} | {fmt(r['better'])} |")
        L.append("")
    if doc.get("notes"):
        L += ["## 备注", ""] + [f"- {n}" for n in doc["notes"]] + [""]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True, type=Path)
    p.add_argument("--baselines", required=True, type=Path)
    p.add_argument("--model", default="qwen3.5-2b")
    p.add_argument("--out", required=True, type=Path, help="输出前缀：写 <out>.json 与 <out>.md")
    p.add_argument("--n-resamples", type=int, default=BOOTSTRAP_N_RESAMPLES)
    a = p.parse_args(argv)
    leaves = find_leaves(a.roots, a.model)
    an = Analysis(leaves, Baselines(a.baselines), a.n_resamples)
    doc: dict[str, Any] = {"roots": [str(r) for r in a.roots], "baselines": str(a.baselines), "model": a.model,
                           "n_resamples": a.n_resamples, "n_leaves": len(leaves)}
    doc["h5"] = an.h5()
    doc["h5_unmerged"] = an.h5(UNMERGED)
    doc["noise_floor"] = an.noise_floor()
    doc["descriptive"] = an.descriptive()
    doc["notes"] = an.notes
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    a.out.with_suffix(".md").write_text(render(doc))
    print(f"写入 {a.out.with_suffix('.json')}、{a.out.with_suffix('.md')}；叶目录 {len(leaves)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
