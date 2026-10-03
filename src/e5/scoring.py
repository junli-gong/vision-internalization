"""E5 的逐条解析与评分、汇总（三列）、两级门槛（DEV-0018；D28-13、D36-6；DEV-0028：D59-4、D59-5）。指标一律在这里与
src/synth/relief.py 定义。

逐条
- 预测统一成 Pred：网格 (N, N) 0/1、全分辨率 mask (H, W)、可选的起伏代码 (N, N)。来源：
  * 零样本：c–e 用 parse_grid_text(n=32)（E1 的数字网格），a、b 用 parse_relief_grid_text(n=32)（33 个符号；b 自 DEV-0028）；
    全分辨率 mask = grid_to_mask 回填到画布；
  * 思维链（DEV-0028 起预算强制收尾，D59-5）：只解析作答段 answer_text（budget_view；思考段的结束方式按 thinking_end 四类，
    DEV-0029）。a、b 同零样本的 33 符号网格；
    c–e 是 parse_polygons（0–1000 坐标的 JSON 多边形）→ 顶点四舍五入到 1/256 px → src/synth/raster.py 的整数光栅化
    （与生成器的真值同一套规则），多个多边形取并集；网格 = mask_to_grid；
  * 平凡基线：生成器给出的 mask 与网格，a、b 的起伏代码取 extras["baseline_relief_<名>"]；派生基线（copy_convex，
    relief.derived_baselines）现算；
  * 外部预测：给出的全分辨率 mask；网格 = mask_to_grid。
  模型输出先过 parse_generation（思考段未闭合、截断、空回答先判；截断一律判失败，同 E1）。
- mask IoU 两种口径：全分辨率（预测 mask 对 target_mask）与网格（预测网格对 target_grid），单样本 IoU 同 src.metrics.seg
  （解析失败记 0，两者都空记 1）。
- 外部预测另算对"当前 mask"的 IoU（D59-5）：当前 mask = 输入图里目标在变换之前的 mask（src.synth.api.current_mask：
  a、b、d 为 copy 基线，c 为可见部分 visible；e 没有图像，不算），两种口径，键 iou_current_full / iou_current_grid。
- a、b 的起伏误差：src/synth/relief.relief_score（RMSE / 目标起伏的标准差，只在公共前景格上算）。只对给出起伏的系统有定义
  （零样本、思维链、平凡基线）；外部预测只给 mask，起伏记"不适用"。

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
- 预算强制收尾的系统（思维链）另按思考段的结束方式 thinking_end 四类（natural 自然收尾、forced 强制收尾、not_opened 没写思考段、
  unclosed 思考段里就停了）分开汇总，给条数、解析失败、截断、思考 / 作答 / 补入的 token 数与读数（thinking_end_block；
  口径同 E1 的 thinking_end 一节，DEV-0029）。

两级门槛（D59-4；gate）。"胜过"一律用与基线逐样本配对的 bootstrap 95% CI（src.stats.paired_bootstrap_ci，差 = 系统 − 基线）判：
- 先决条件：标题口径上，系统的网格 mask IoU（严格）胜过最强的非真值基线（角色 trivial 的基线里严格均值最高者，同分取名字
  排序靠前的；gt_affine_upper 是参照线，不算）：差的 CI 下界 > 0。
- 判定条件（a，以及 b 的环绕移动 = b 的标题口径）：标题口径上，系统的起伏误差（严格）同时胜过三条基线
  JUDGMENT_BASELINES = 平面 copy_flat、真值仿射 gt_affine_upper、凸起先验 copy_convex：每一条的差的 CI 上界 < 0，三条都胜过
  才通过。每条另报均值、中位数、差的均值与 CI、逐样本差的中位数、系统更低的样本比例。
  * c、d、e：没有起伏，不适用。不给起伏的系统（外部预测）在 a、b 上记 not_applicable。
- imagining：c、d、e = 先决条件；a、b = 先决条件与判定条件都通过才为 true，有一条不通过为 false，判不了（CI 缺失、
  系统不给起伏而先决条件通过）为 null。
- 作废：D28-13 照字面的判法（只比真值仿射、按点估计判）与 DEV-0018 的 pass_vs_flat 不再输出，README 注明已作废（D59-4）。
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.adapters.base import THINKING_ENDS
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
from src.synth.api import current_mask
from src.synth.relief import (
    CONVEX_BASELINE,
    DERIVED_BASELINE_ROLES,
    RELIEF_TARGET_KEY,
    RELIEF_TASKS,
    baseline_relief_key,
    derived_baseline_names,
    derived_baselines,
    relief_score,
)
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

RELIEF_FAIL_VALUE = 2.0
GATE_IOU_KEY = "iou_grid"  # 门槛用网格口径：模型给出的是 32×32 网格，基线也取它们的网格，同一分辨率比较
REFERENCE_ROLE = "reference_upper_bound"
GT_AFFINE = "gt_affine_upper"
FLAT_BASELINE = "copy_flat"
# 判定条件的三条基线（D59-4）：起伏误差要同时胜过它们
JUDGMENT_BASELINES = (FLAT_BASELINE, GT_AFFINE, CONVEX_BASELINE)
CURRENT_KEYS = {"iou_current_full": "mask_iou_current_full", "iou_current_grid": "mask_iou_current_grid"}

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
    """基线 → Pred。生成器给出的基线：a、b 的起伏代码取 extras（其他任务没有起伏）；派生基线（copy_convex）现算。"""
    if name in derived_baseline_names(sample.task):
        b, relief = derived_baselines(sample)[name]
    else:
        b = sample.baselines[name]
        relief = sample.extras.get(baseline_relief_key(name))
    return Pred(np.asarray(b.grid, dtype=np.uint8), np.asarray(b.mask, dtype=bool),
                None if relief is None else np.asarray(relief, dtype=np.uint8))


def baseline_names(sample: Sample) -> list[str]:
    """样本的全部基线名（生成器给出的 + 派生的），排序。"""
    return sorted(set(sample.baselines) | set(derived_baseline_names(sample.task)))


def baseline_roles(sample: Sample) -> dict[str, str]:
    """基线名 → 角色（trivial / reference_upper_bound）。c、d、e 的元数据里没有角色，全部是 trivial；派生基线见
    relief.DERIVED_BASELINE_ROLES。"""
    roles = {**(sample.meta.get("baseline_roles") or {}), **DERIVED_BASELINE_ROLES}
    return {name: roles.get(name, "trivial") for name in baseline_names(sample)}


def budget_view(result: Any) -> Any:
    """预算强制收尾的一次生成（generate_batch_budgeted 的 GenerationResult）→ 给 parse_generation 的视图：只看作答段。
    按 thinking_end 的四类（DEV-0027 的定义见 src/adapters/budget.py；DEV-0029）：

    - text = answer_text：natural、forced 是收尾之后的部分，not_opened 是全部生成（没有思考段），unclosed 为空串；
    - stopped = 适配器的 stopped：natural、forced、not_opened 指作答段是否以停止 token 结束（False = 写满 answer_budget，判截断）；
      unclosed 是思考段里写出了停止 token，为 True；
    - thinking_unclosed = (thinking_end == "unclosed")：思考段里就停了、没有作答段，判 thinking_unclosed（同 E1）。
      适配器的 thinking_unclosed 是 split_answer 对完整序列的判定（作答段里又写了思考段开头也算），E5 只解析作答段，不用它。"""
    end = getattr(result, "thinking_end", None)
    if end not in THINKING_ENDS:
        raise ValueError(f"预算强制收尾的结果 thinking_end 必须是 {THINKING_ENDS} 之一，收到 {end!r}")
    return _AnswerView(result.answer_text, result.stopped, end == "unclosed")


@dataclass(frozen=True)
class _AnswerView:
    text: str
    stopped: bool | None
    thinking_unclosed: bool


# ---------------------------------------------------------------------------
# 逐条评分
# ---------------------------------------------------------------------------


def score(sample: Sample, pred: Pred, *, relief_expected: bool, current: bool = False) -> dict[str, Any]:
    """一条预测的得分（写进 predictions.jsonl 的 score）。relief_expected：这个系统在这个任务上是否应给出起伏。
    current：另算对"当前 mask"的 IoU（外部预测，D59-5；没有当前 mask 的任务 e 记 None）。"""
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
    if current:
        cm = current_mask(sample)
        if cm is None:
            out.update(iou_current_full=None, iou_current_grid=None)
        else:
            cf = mask_inter_union(pred.mask if pred.valid else None, cm.mask)
            cg = mask_inter_union(pred.grid if pred.valid else None, cm.grid)
            out.update(iou_current_full=sample_iou(*cf), iou_current_grid=sample_iou(*cg))
    if relief_expected and canonical_task(sample.task) in RELIEF_TASKS:
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
    for key, name in CURRENT_KEYS.items():  # 外部预测对"当前 mask"的 IoU（D59-5）
        cur = [r for r in recs if r["score"].get(key) is not None]
        if cur:
            out[name] = iou_block(cur, key, n_resamples)
    rb = relief_block(recs, n_resamples)
    if rb is not None:
        out["relief_error"] = rb
    return out


def _tok_stats(v: Sequence[int]) -> dict[str, Any] | None:
    return {"mean": float(np.mean(v)), "max": int(max(v)), "p50": float(np.median(v)), "total": int(sum(v))} if v else None


def thinking_end_block(recs: Sequence[Mapping[str, Any]], n_resamples: int) -> dict[str, Any] | None:
    """预算强制收尾（D59-5）：按思考段的结束方式（THINKING_ENDS：natural 自然收尾、forced 强制收尾、not_opened 没写思考段（Gemma、
    InternVL 第一个 token 不是思考开头，直接作答）、unclosed 思考段里就停了（写出停止 token，没有作答段））分开报，口径同 E1 的
    thinking_end 一节（src/e1/scoring.thinking_end_breakdown；DEV-0029）：
    - counts：各类条数，四类都列，没有的记 0；
    - by_kind：每个出现的类给条数、标题口径条数、解析失败率与分原因、截断条数、思考段与作答段的 token 数，以及读数
      （标题口径与全部样本各一份 blocks：严格 / 有效率 / 条件三列）；
    - inserted_tokens：补入的 token 数（只有 forced 有，其余记 0）。
    记录里没有 thinking_end（不是预算强制收尾的运行）时返回 None；有的记录有、有的没有，或取值不在四类里，报错。"""
    have = [r.get("thinking_end") is not None for r in recs]
    if not any(have):
        return None
    if not all(have):
        raise ValueError("一次运行里有的记录有 thinking_end、有的没有")
    counts = Counter(r["thinking_end"] for r in recs)
    bad = sorted(set(counts) - set(THINKING_ENDS))
    if bad:
        raise ValueError(f"未知的 thinking_end {bad}（应为 {THINKING_ENDS} 之一）")
    out: dict[str, Any] = {"n": len(recs), "counts": {k: counts.get(k, 0) for k in THINKING_ENDS}, "by_kind": {}}
    for kind in THINKING_ENDS:
        sub = [r for r in recs if r["thinking_end"] == kind]
        if not sub:
            continue
        tally = ParseTally(failure_enum(r["failure"]) for r in sub)
        head = [r for r in sub if r["headline"]]
        out["by_kind"][kind] = {"n": len(sub), "n_headline": len(head), "parse_failure_rate": tally.failure_rate,
                                "parse": tally.to_dict(), "n_truncated": sum(1 for r in sub if r["truncated"]),
                                "thinking_tokens": _tok_stats([int(r["n_thinking_tokens"]) for r in sub]),
                                "answer_tokens": _tok_stats([int(r["n_answer_tokens"]) for r in sub]),
                                "headline": blocks(head, n_resamples), "overall": blocks(sub, n_resamples)}
    out["inserted_tokens"] = _tok_stats([int(r["n_inserted_tokens"]) for r in recs])
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
    te = thinking_end_block(recs, n_resamples)
    if te is not None:
        out["thinking_end"] = te
    return out


# ---------------------------------------------------------------------------
# 两级门槛
# ---------------------------------------------------------------------------


def _paired(a: Sequence[float], b: Sequence[float], n_resamples: int) -> dict[str, float] | None:
    if not a or n_resamples <= 0:
        return None
    ci = paired_bootstrap_ci(np.asarray(a, np.float64), np.asarray(b, np.float64), n_resamples=n_resamples)
    return None if ci is None else {"estimate": ci.estimate, "low": ci.low, "high": ci.high, "level": ci.level,
                                    "n_resamples": ci.n_resamples}


def _compare(sys_vals: Sequence[float], base_vals: Sequence[float], n_resamples: int, *, lower_is_better: bool) -> dict[str, Any]:
    """系统与一条基线的逐样本配对比较（差 = 系统 − 基线）。beats：越低越好时 CI 上界 < 0，越高越好时 CI 下界 > 0；
    CI 算不了（空集合、n_resamples = 0）时为 None。"""
    sv, bv = np.asarray(sys_vals, np.float64), np.asarray(base_vals, np.float64)
    ci = _paired(list(sv), list(bv), n_resamples)
    d = sv - bv
    beats = None if ci is None else (ci["high"] < 0 if lower_is_better else ci["low"] > 0)
    return {"n": int(sv.size), "system": _mean(sv), "system_median": _median(sv), "value": _mean(bv), "median": _median(bv),
            "diff": _mean(d), "diff_ci": ci, "diff_median": _median(d),
            "system_better_rate": (float(np.mean(d < 0 if lower_is_better else d > 0)) if d.size else None),
            "beats": beats}


def gate(task: str, system: Sequence[Mapping[str, Any]], baselines: Mapping[str, Sequence[Mapping[str, Any]]],
         roles: Mapping[str, str], *, n_resamples: int = BOOTSTRAP_N_RESAMPLES, iou_key: str = GATE_IOU_KEY) -> dict[str, Any]:
    """两级门槛（模块说明，D59-4）。system 与各基线的记录按 id 对齐，只取标题口径的样本。基线记录用 score() 算出
    （与系统同一个评分）。"""
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
    out: dict[str, Any] = {"task": t, "iou_key": iou_key, "n": len(head), "scope": headline_scope_text(t),
                           "rule": "配对 bootstrap 95% CI（差 = 系统 − 基线）：IoU 下界 > 0、起伏误差上界 < 0 才算胜过（D59-4）",
                           "n_resamples": n_resamples}
    if not head or not trivial:
        out.update(prerequisite=None, judgment=None, imagining=None)
        return out
    b_iou = {n: float(np.mean([float(r["score"][iou_key]) for r in base[n]])) for n in trivial}
    strongest = max(trivial, key=lambda n: (b_iou[n], -trivial.index(n)))  # 同分时取名字排序靠前的（copy 先于 copy_convex、copy_flat）
    cmp_iou = _compare(sys_iou, [float(r["score"][iou_key]) for r in base[strongest]], n_resamples, lower_is_better=False)
    pre = {"baselines": dict(sorted(b_iou.items())), "strongest": strongest, "strongest_value": b_iou[strongest],
           **cmp_iou, "pass": cmp_iou["beats"]}
    out["prerequisite"] = pre
    if t in RELIEF_TASKS:
        if not all(r["score"].get("relief") is not None for r in head):
            jud: dict[str, Any] = {"status": "not_applicable", "note": "这个系统不给起伏（只给 mask）", "pass": None}
        else:
            absent = [n for n in JUDGMENT_BASELINES if n not in base]
            if absent:
                raise ValueError(f"{t} 的判定条件缺少基线 {absent}（要 {JUDGMENT_BASELINES}）")
            pop_ids = [r["id"] for r in relief_population(head)]
            sv = {r["id"]: relief_strict_value(r) for r in head}
            s_vals = [sv[i] for i in pop_ids]
            jud = {"status": "evaluated", "n": len(pop_ids), "system": _mean(s_vals), "system_median": _median(s_vals),
                   "baselines": {}}
            for name in JUDGMENT_BASELINES:
                bm = {r["id"]: r for r in base[name]}
                jud["baselines"][name] = _compare(s_vals, [relief_strict_value(bm[i]) for i in pop_ids], n_resamples,
                                                  lower_is_better=True)
            beats = [jud["baselines"][n]["beats"] for n in JUDGMENT_BASELINES]
            jud["pass"] = None if (not pop_ids or any(b is None for b in beats)) else all(beats)
    else:
        jud = {"status": "not_applicable", "pass": None}
    out["judgment"] = jud
    if t not in RELIEF_TASKS:
        out["imagining"] = pre["pass"]
    elif pre["pass"] is False or jud["pass"] is False:
        out["imagining"] = False
    elif pre["pass"] is True and jud["pass"] is True:
        out["imagining"] = True
    else:
        out["imagining"] = None
    return out


def baseline_records(samples: Iterable[Sample]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """全部样本上每个基线（含派生基线）的记录（与系统的记录同形）与角色。"""
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
