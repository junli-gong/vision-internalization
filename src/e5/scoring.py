"""E5 的逐条解析与评分、汇总（三列）、两级门槛（DEV-0018；D28-13、D36-6）。指标一律在这里与 src/synth/relief.py 定义。

逐条
- 预测统一成 Pred：网格 (N, N) 0/1、全分辨率 mask (H, W)、可选的起伏代码 (N, N)。来源：
  * 零样本：b–e 用 parse_grid_text(n=32)（E1 的数字网格），a 用 parse_relief_grid_text(n=32)（33 个符号）；全分辨率 mask =
    grid_to_mask 回填到画布；
  * 思维链：parse_polygons（0–1000 坐标的 JSON 多边形）→ 顶点四舍五入到 1/256 px → src/synth/raster.py 的整数光栅化
    （与生成器的真值同一套规则），多个多边形取并集；网格 = mask_to_grid；
  * 平凡基线：生成器给出的 mask 与网格；a 的起伏代码取 extras["baseline_relief_<名>"]；
  * 外部预测：给出的全分辨率 mask；网格 = mask_to_grid。
  模型输出先过 parse_generation（思考段未闭合、截断、空回答先判；截断一律判失败，同 E1）。
- mask IoU 两种口径：全分辨率（预测 mask 对 target_mask）与网格（预测网格对 target_grid），单样本 IoU 同 src.metrics.seg
  （解析失败记 0，两者都空记 1）。
- a 的起伏误差：src/synth/relief.relief_score（RMSE / 目标起伏的标准差，只在公共前景格上算）。只对给出起伏的系统
  （零样本、平凡基线）有定义；思维链与外部预测只给 mask，起伏记"不适用"。

汇总的三列（同 E1，D36-6）
- 严格（strict，标题）：全部样本；mask IoU 解析失败记 0；起伏误差在"目标起伏有定义"（目标不是全平）的样本上算，
  预测无效或与目标没有公共前景格时记 RELIEF_FAIL_VALUE = 2.0：与目标起伏完全反相关（深度前后翻转）的预测的误差
  （p − μ = −(t − μ) ⇒ RMSE = 2σ）。它比"预测一个平面"（≈ 1.0）、"与目标同分布但独立的猜测"（√2）都差，也比 30° 以上
  gt_affine_upper 的均值（生成器上实测 1.1–1.4）差，所以失败在判定条件里只会拖后腿（取 √2 时，全部失败的系统在 60–90° 档
  就可能"低于"真值仿射）。
- 有效率（valid_rate）：mask：解析成功的比例（拒答一定解析不出网格或多边形）；起伏：起伏误差有定义的比例。
- 条件读数（conditional）：只在有效的样本上算的均值（另给中位数，起伏误差的分布有长尾）。
- 每列都给样本数；严格读数给 bootstrap 95% CI（按样本重采样，src.stats）。空集合一律 null。
- 分列：标题口径（protocol.in_headline）、全部样本、每档、各分列维度（protocol.groups_of）。

两级门槛（D28-13；gate）
- 先决条件：标题口径上，系统的网格 mask IoU（严格）高于最强的非真值基线（角色 trivial 的基线里严格读数最高者；
  a 的 gt_affine_upper 是参照线，不算）。点估计严格大于即"胜过"；另给与最强基线逐样本配对的差值 95% CI，只报告。
- 判定条件（a）：标题口径上，系统的起伏误差（严格）低于真值拟合的二维仿射（gt_affine_upper）的起伏误差。
  * 另报 "vs_flat"：同时低于 copy_flat（"预测一个平面"）的起伏误差。生成器上实测 copy_flat ≈ 1.02，而 30° 以上
    gt_affine_upper ≈ 1.1–1.4：照字面的判定条件可以被"给一个平面"满足（见完成总结的待裁项）。
  * b 的环绕移动：生成器没有 b 的起伏与真值仿射，判定条件记 unavailable（待裁）。c、d、e：不适用。
  * 不给起伏的系统（思维链、外部预测）在 a 上记 not_applicable。
- imagining = 先决条件通过，且判定条件通过（或不适用）。unavailable 时为 null。
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.constants import BOOTSTRAP_N_RESAMPLES
from src.e5.protocol import E5_GRID_SIZE, groups_of, headline_scope_text, in_headline, sample_id
from src.executors.raster import mask_to_rle
from src.grid.codec import (
    GridParseFailure,
    grid_to_mask,
    grid_to_text,
    mask_to_grid,
    parse_grid_text,
    parse_relief_grid_text,
    relief_grid_to_text,
)
from src.metrics.seg import mask_inter_union, sample_iou
from src.parsing import ParseFailure, ParseTally, parse_generation, parse_polygons
from src.parsing.refusal import DIGIT_RE as _DIGIT
from src.parsing.refusal import REFUSAL_RE
from src.stats import bootstrap_ci, paired_bootstrap_ci
from src.synth.raster import UNITS, rasterize_units
from src.synth.relief import RELIEF_TARGET_KEY, baseline_relief_key, relief_score
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

RELIEF_FAIL_VALUE = 2.0
GATE_IOU_KEY = "iou_grid"  # 门槛用网格口径：模型给出的是 32×32 网格，基线也取它们的网格，同一分辨率比较
REFERENCE_ROLE = "reference_upper_bound"
GT_AFFINE = "gt_affine_upper"
FLAT_BASELINE = "copy_flat"

class E5Failure(str, enum.Enum):
    """E5 自己的失败原因（ParseFailure、GridParseFailure 之外）：外部预测没有给出 mask。"""

    EXTERNAL_MISSING = "external_missing"  # 外部预测文件里没有这个 id
    EXTERNAL_FAILURE = "external_failure"  # 外部预测明确记了失败


_FAILURES = {e.value: e for e in list(ParseFailure) + list(GridParseFailure) + list(E5Failure)}
if len(_FAILURES) != len(ParseFailure) + len(GridParseFailure) + len(E5Failure):  # pragma: no cover
    raise ImportError("失败原因有重名的取值")
# 拒答的诊断（只作诊断，不改失败原因）：失败、回答里没有阿拉伯数字（_DIGIT）、且 REFUSAL_RE 能匹配。两个正则与 E1 同一来源
# src/parsing/refusal.py（DEV-0019），本模块顶部 import。


def failure_enum(value: str | None):
    if value is None:
        return None
    return _FAILURES[value]


# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pred:
    grid: np.ndarray | None  # (N, N) uint8 0/1
    mask: np.ndarray | None  # (H, W) bool
    relief: np.ndarray | None = None  # (N, N) uint8 起伏代码（0 背景）
    failure: str | None = None
    polygons: list | None = None  # 思维链解析出的多边形（像素坐标，量化前）

    @property
    def valid(self) -> bool:
        return self.failure is None


def failed(reason: str) -> Pred:
    return Pred(None, None, None, reason)


def from_grid(grid: np.ndarray, height: int, width: int) -> Pred:
    g = np.asarray(grid, dtype=np.uint8)
    return Pred(g, grid_to_mask(g, height, width))


def from_relief(codes: np.ndarray, height: int, width: int) -> Pred:
    c = np.asarray(codes, dtype=np.uint8)
    g = (c > 0).astype(np.uint8)
    return Pred(g, grid_to_mask(g, height, width), c)


def from_mask(mask: np.ndarray, n: int = E5_GRID_SIZE, relief: np.ndarray | None = None) -> Pred:
    m = np.asarray(mask, dtype=bool)
    return Pred(mask_to_grid(m, n), m, None if relief is None else np.asarray(relief, dtype=np.uint8))


def rasterize_polygons(polygons: Sequence[Sequence[tuple[float, float]]], height: int, width: int) -> np.ndarray:
    """像素坐标的多边形（parse_polygons 的输出）→ (H, W) bool：顶点四舍五入到 1/256 px，src/synth/raster.py 的整数规则，取并集。"""
    out = np.zeros((height, width), dtype=bool)
    for poly in polygons:
        q = np.rint(np.asarray(poly, dtype=np.float64) * UNITS).astype(np.int64)
        out |= rasterize_units(q, height, width)
    return out


def parse_answer(kind: str, result: Any, height: int, width: int, n: int = E5_GRID_SIZE) -> Pred:
    """一次生成（GenerationResult）→ Pred。kind 取 prompts.FORMAT_KINDS 的值。"""
    if kind == "binary_grid":
        p = parse_generation(result, parse_grid_text, n=n)
        return failed(p.failure.value) if p.failure else from_grid(p.grid, height, width)
    if kind == "relief_grid":
        p = parse_generation(result, parse_relief_grid_text, n=n)
        return failed(p.failure.value) if p.failure else from_relief(p.grid, height, width)
    if kind == "polygon":
        p = parse_generation(result, parse_polygons, width=width, height=height, axis_order="xy")
        if p.failure:
            return failed(p.failure.value)
        polys = [list(poly) for poly in p.value]
        m = rasterize_polygons(polys, height, width)
        return Pred(mask_to_grid(m, n), m, None, None, polys)
    raise ValueError(f"未知的解析方式 {kind!r}")


def baseline_pred(sample: Sample, name: str) -> Pred:
    """生成器给出的基线 → Pred（a 的起伏代码取 extras；其他任务没有起伏）。"""
    b = sample.baselines[name]
    relief = sample.extras.get(baseline_relief_key(name))
    return Pred(np.asarray(b.grid, dtype=np.uint8), np.asarray(b.mask, dtype=bool),
                None if relief is None else np.asarray(relief, dtype=np.uint8))


def baseline_roles(sample: Sample) -> dict[str, str]:
    """基线名 → 角色（trivial / reference_upper_bound）。c、d、e 的元数据里没有角色，全部是 trivial。"""
    roles = sample.meta.get("baseline_roles") or {}
    return {name: roles.get(name, "trivial") for name in sorted(sample.baselines)}


# ---------------------------------------------------------------------------
# 逐条评分
# ---------------------------------------------------------------------------


def score(sample: Sample, pred: Pred, *, relief_expected: bool) -> dict[str, Any]:
    """一条预测的得分（写进 predictions.jsonl 的 score）。relief_expected：这个系统在这个任务上是否应给出起伏。"""
    if pred.valid:
        full = mask_inter_union(pred.mask, sample.target_mask)
        grid = mask_inter_union(pred.grid, sample.target_grid)
    else:
        full = mask_inter_union(None, sample.target_mask)
        grid = mask_inter_union(None, sample.target_grid)
    out: dict[str, Any] = {
        "iou_full": sample_iou(*full),
        "iou_grid": sample_iou(*grid),
        "inter_full": full.inter, "union_full": full.union,
        "inter_grid": grid.inter, "union_grid": grid.union,
        "pred_cells": None if pred.grid is None else int(np.count_nonzero(pred.grid)),
        "target_cells": int(np.count_nonzero(sample.target_grid)),
        "relief": None,
    }
    if relief_expected and canonical_task(sample.task) == "rotation3d":
        target = sample.extras[RELIEF_TARGET_KEY]
        t = relief_score(target, target)
        if not pred.valid or pred.relief is None:
            out["relief"] = {"error": None, "reason": "invalid_prediction" if t.reason is None else t.reason,
                             "n_eval_cells": 0, "target_flat": t.reason == "flat_target"}
        else:
            sc = relief_score(pred.relief, target)
            out["relief"] = {"error": sc.error, "reason": sc.reason, "n_eval_cells": sc.n_eval_cells,
                             "target_flat": sc.reason == "flat_target"}
    return out


def pred_record(pred: Pred, *, store_rle: bool) -> dict[str, Any]:
    """预测本身（predictions.jsonl 的 pred）：网格文本、起伏文本、多边形、可选的全分辨率 RLE。"""
    return {
        "grid": None if pred.grid is None else grid_to_text(pred.grid).split("\n"),
        "relief": None if pred.relief is None else relief_grid_to_text(pred.relief).split("\n"),
        "polygons": None if pred.polygons is None else [[list(map(float, p)) for p in poly] for poly in pred.polygons],
        "rle": mask_to_rle(pred.mask) if (store_rle and pred.mask is not None) else None,
    }


def sample_fields(sample: Sample) -> dict[str, Any]:
    """每条记录都带的样本字段：id、档、分列维度、是否计入标题。"""
    g = groups_of(sample.task, sample.meta)
    b = int(sample.meta["difficulty"]["bin"])
    return {"id": sample_id(sample.task, sample.split, sample.index), "task": canonical_task(sample.task),
            "split": sample.split, "index": int(sample.index), "bin": b,
            "difficulty_value": sample.meta["difficulty"]["value"], "groups": g,
            "headline": in_headline(sample.task, b, g)}


def is_refusal(rec: Mapping[str, Any]) -> bool:
    if rec.get("failure") is None:
        return False
    text = rec.get("text") or ""
    return _DIGIT.search(text) is None and REFUSAL_RE.search(text) is not None


# ---------------------------------------------------------------------------
# 汇总：三列
# ---------------------------------------------------------------------------


def _ci(values: Sequence[float], n_resamples: int) -> dict[str, float] | None:
    if not values or n_resamples <= 0:
        return None
    ci = bootstrap_ci(np.asarray(values, dtype=np.float64), n_resamples=n_resamples)
    return None if ci is None else {"low": ci.low, "high": ci.high, "level": ci.level, "n_resamples": ci.n_resamples}


def _mean(v: Sequence[float]) -> float | None:
    return float(np.mean(v)) if len(v) else None


def _median(v: Sequence[float]) -> float | None:
    return float(np.median(v)) if len(v) else None


def iou_block(recs: Sequence[Mapping[str, Any]], key: str, n_resamples: int) -> dict[str, Any]:
    strict = [float(r["score"][key]) for r in recs]
    valid = [float(r["score"][key]) for r in recs if r["valid"]]
    n = len(recs)
    return {"n": n, "n_valid": len(valid), "valid_rate": (len(valid) / n) if n else None,
            "strict": _mean(strict), "strict_ci": _ci(strict, n_resamples),
            "conditional": _mean(valid), "conditional_median": _median(valid)}


def relief_population(recs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """起伏误差的总体：给了起伏记录、且目标起伏有定义（不是全平）的样本。"""
    return [r for r in recs if r["score"].get("relief") is not None and not r["score"]["relief"]["target_flat"]]


def relief_strict_value(r: Mapping[str, Any]) -> float:
    e = r["score"]["relief"]["error"]
    return RELIEF_FAIL_VALUE if e is None else float(e)


def relief_block(recs: Sequence[Mapping[str, Any]], n_resamples: int) -> dict[str, Any] | None:
    applicable = [r for r in recs if r["score"].get("relief") is not None]
    if not applicable:
        return None
    pop = relief_population(applicable)
    strict = [relief_strict_value(r) for r in pop]
    valid = [float(r["score"]["relief"]["error"]) for r in pop if r["score"]["relief"]["error"] is not None]
    n = len(pop)
    reasons = Counter(r["score"]["relief"]["reason"] or "ok" for r in applicable)
    return {"n": n, "n_target_flat": len(applicable) - n, "n_valid": len(valid),
            "valid_rate": (len(valid) / n) if n else None, "strict": _mean(strict),
            "strict_ci": _ci(strict, n_resamples), "conditional": _mean(valid), "conditional_median": _median(valid),
            "fail_value": RELIEF_FAIL_VALUE, "reasons": dict(sorted(reasons.items()))}


def blocks(recs: Sequence[Mapping[str, Any]], n_resamples: int) -> dict[str, Any]:
    out = {"mask_iou_grid": iou_block(recs, "iou_grid", n_resamples),
           "mask_iou_full": iou_block(recs, "iou_full", n_resamples)}
    rb = relief_block(recs, n_resamples)
    if rb is not None:
        out["relief_error"] = rb
    return out


def aggregate(task: str, recs: Sequence[Mapping[str, Any]], *, n_resamples: int = BOOTSTRAP_N_RESAMPLES,
              enable_thinking: bool | None = None) -> dict[str, Any]:
    """predictions.jsonl 的记录 → metrics 的主体（不含门槛；门槛见 gate）。只读记录，续跑后再算逐字节相同。"""
    t = canonical_task(task)
    head = [r for r in recs if r["headline"]]
    by_bin = {}
    for b in sorted({r["bin"] for r in recs}):
        by_bin[str(b)] = blocks([r for r in recs if r["bin"] == b], n_resamples)
    by_group: dict[str, dict[str, Any]] = {}
    dims = sorted({d for r in recs for d in r["groups"]})
    for d in dims:
        vals = sorted({r["groups"][d] for r in recs})
        by_group[d] = {v: blocks([r for r in recs if r["groups"][d] == v], n_resamples) for v in vals}
    tally = ParseTally(failure_enum(r["failure"]) for r in recs)
    gen = [r for r in recs if "n_generated_tokens" in r]
    out: dict[str, Any] = {
        "task": t,
        "n": len(recs),
        "headline_scope": headline_scope_text(t),
        "headline": blocks(head, n_resamples),
        "overall": blocks(recs, n_resamples),
        "by_bin": by_bin,
        "by_group": by_group,
        "parse": tally.to_dict(),
        "refusal": {"n": sum(1 for r in recs if is_refusal(r))},
    }
    if gen:
        toks = [int(r["n_generated_tokens"]) for r in gen]
        out["truncation"] = {"n": sum(1 for r in gen if r.get("truncated")), "rate": sum(1 for r in gen if r.get("truncated")) / len(gen)}
        out["thinking_unclosed"] = {"n": sum(1 for r in gen if r.get("thinking_unclosed")),
                                    "rate": sum(1 for r in gen if r.get("thinking_unclosed")) / len(gen)}
        out["generated_tokens"] = {"mean": float(np.mean(toks)), "max": int(max(toks)), "p50": float(np.median(toks))}
        pix = [int(r["n_image_pixels"]) for r in gen]
        out["image_pixels"] = {"mean": float(np.mean(pix)), "min": int(min(pix)), "max": int(max(pix))}
        out["enable_thinking"] = enable_thinking
    return out


# ---------------------------------------------------------------------------
# 两级门槛
# ---------------------------------------------------------------------------


def _paired(a: Sequence[float], b: Sequence[float], n_resamples: int) -> dict[str, float] | None:
    if not a or n_resamples <= 0:
        return None
    ci = paired_bootstrap_ci(np.asarray(a, np.float64), np.asarray(b, np.float64), n_resamples=n_resamples)
    return None if ci is None else {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "level": ci.level}


def gate(task: str, system: Sequence[Mapping[str, Any]], baselines: Mapping[str, Sequence[Mapping[str, Any]]],
         roles: Mapping[str, str], *, n_resamples: int = BOOTSTRAP_N_RESAMPLES, iou_key: str = GATE_IOU_KEY) -> dict[str, Any]:
    """两级门槛（模块说明）。system 与各基线的记录按 id 对齐，只取标题口径的样本。基线记录用 score() 算出（与系统同一个评分）。"""
    t = canonical_task(task)
    head = [r for r in system if r["headline"]]
    ids = [r["id"] for r in head]
    base = {}
    for name, recs in baselines.items():
        m = {r["id"]: r for r in recs}
        missing = [i for i in ids if i not in m]
        if missing:
            raise ValueError(f"基线 {name} 缺少 {len(missing)} 条样本，例如 {missing[:3]}")
        base[name] = [m[i] for i in ids]
    sys_iou = [float(r["score"][iou_key]) for r in head]
    trivial = sorted(n for n, role in roles.items() if role != REFERENCE_ROLE and n in base)
    out: dict[str, Any] = {"task": t, "iou_key": iou_key, "n": len(head), "scope": headline_scope_text(t)}
    if not head or not trivial:
        out.update(prerequisite=None, judgment=None, imagining=None)
        return out
    b_iou = {n: float(np.mean([float(r["score"][iou_key]) for r in base[n]])) for n in trivial}
    strongest = max(trivial, key=lambda n: (b_iou[n], -trivial.index(n)))  # 同分时取名字排序靠前的（copy 先于 copy_flat）
    s_mean = float(np.mean(sys_iou))
    pre = {"system": s_mean, "baselines": dict(sorted(b_iou.items())), "strongest": strongest,
           "strongest_value": b_iou[strongest], "diff": s_mean - b_iou[strongest],
           "diff_ci": _paired(sys_iou, [float(r["score"][iou_key]) for r in base[strongest]], n_resamples),
           "pass": s_mean > b_iou[strongest]}
    out["prerequisite"] = pre
    if t == "rotation3d":
        has_relief = all(r["score"].get("relief") is not None for r in head)
        if not has_relief:
            jud = {"status": "not_applicable", "note": "这个系统不给起伏（只给 mask）", "pass": None}
        else:
            pop_ids = [r["id"] for r in relief_population(head)]
            sv = {r["id"]: relief_strict_value(r) for r in head if r["id"] in set(pop_ids)}
            refs = {}
            for name in (GT_AFFINE, FLAT_BASELINE):
                if name in base:
                    bm = {r["id"]: r for r in base[name]}
                    refs[name] = [relief_strict_value(bm[i]) for i in pop_ids]
            s_vals = [sv[i] for i in pop_ids]
            s_rel = _mean(s_vals)
            jud = {"status": "evaluated", "n": len(pop_ids), "system": s_rel}
            for name, vals in refs.items():
                jud[name] = {"value": _mean(vals), "diff": None if s_rel is None else s_rel - _mean(vals),
                             "diff_ci": _paired(s_vals, vals, n_resamples)}
            ga = refs.get(GT_AFFINE)
            jud["pass"] = None if (s_rel is None or ga is None) else s_rel < _mean(ga)
            fl = refs.get(FLAT_BASELINE)
            jud["pass_vs_flat"] = None if (s_rel is None or fl is None or ga is None) else (s_rel < _mean(ga) and s_rel < _mean(fl))
            jud["flat_beats_gt_affine"] = None if (fl is None or ga is None) else _mean(fl) < _mean(ga)
    elif t == "viewpoint":
        jud = {"status": "unavailable", "pass": None,
               "note": "D28-13 的判定条件也适用于 b 的环绕移动，但生成器没有 b 的起伏与真值仿射（待裁）"}
    else:
        jud = {"status": "not_applicable", "pass": None}
    out["judgment"] = jud
    if jud["status"] == "unavailable":
        out["imagining"] = None
    elif jud["status"] == "evaluated":
        out["imagining"] = bool(pre["pass"] and jud["pass"])
    else:
        out["imagining"] = bool(pre["pass"])
    return out


def baseline_records(samples: Iterable[Sample]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """全部样本上每个基线的记录（与系统的记录同形）与角色。"""
    out: dict[str, list[dict[str, Any]]] = {}
    roles: dict[str, str] = {}
    for s in samples:
        for name, role in baseline_roles(s).items():
            roles.setdefault(name, role)
            p = baseline_pred(s, name)
            rec = {**sample_fields(s), "valid": True, "failure": None,
                   "score": score(s, p, relief_expected=p.relief is not None)}
            out.setdefault(name, []).append(rec)
    return out, roles
