"""W1 的重放与错误分析（DEV-0057；D113-4；docs/w2_plan.md §4）。只用 CPU，不加载模型。

W1 的记录（predictions.jsonl）只存了最终提交的 mask。工作台逐位确定（DEV-0050 的测试钉住），所以按记录里的调用序列重放，就能得到
每一步的 mask，与真值算 IoU。这里只读 W1 的代码，不改它。

重放（replay_episode）：
- 用原图新建一个工作台，构造方式与运行时相同：workbench_factory(原图副本, disabled_groups=…, **工作台选项)，参数取运行身份
  config.yaml（protocol.disabled_groups、workbench.options 里与默认值不同的项，同 W1Config.workbench_options()）。工具清单的
  sha256 必须等于记录里的 tools_sha256，原图尺寸必须等于记录的 image_size，否则抛 ReplayError（不是这一版工作台，重放没有意义）。
- 按记录的顺序（逐轮、轮内按出现顺序），把 executed = true 的调用逐个 Workbench.call(name, arguments) 重放；被拒的（refused）与
  解析失败的（tool_call_errors，本来就不在 tool_calls 里）不重放。
- 一致性核对（不一致的条目标 replay.mismatch = true，原因写进 replay.reasons，不进分类）：
  - 每次调用的返回文字与记录的 result.text 逐字相同（"text@k:工具名"，k 是重放的第几次调用，从 0 起）；
  - 提交的 mask 与记录的 RLE 逐位相同（"submission"）：记录有 mask ⇔ 重放有 mask 提交，且逐位相等；
  - 最终得分的交、并与记录的 score 相同（"score"；顺带核对了真值是同一份）。
  - 返回图的像素 sha256 只计数（replay.n_image_mismatch），不算不一致：图不进任何 mask。
- 每次调用之后，记下工作台里每个 mask 的 IoU（与真值，src/metrics/seg 的同一口径）与面积。

每条样本的量（replay.jsonl 的一行，见 replay_episode）：
- final_iou：最终提交的 IoU；失败码不为空（没有有效提交）时为 0。
- best_iou：整个过程中任何一个 mask 的最高 IoU，以及它第一次出现在第几次调用之后（best_call）、是哪个 mask（best_mask）。
  从来没有 mask 时为 0（空 mask）。
- seed_box（原图 0–1000）：第一次**成功**、且用 box 初始化的 grabcut 调用的 box 参数（seed_box_source = "grabcut_box"）。没有这样的
  调用时，取第一个被写出的非空 mask 的外接框（seed_box_source = "mask_bbox:<写它的工具>"）；都没有时为 None。
  view_frames 开、调用给了 view 时，box 先按那张视图换回原图 0–1000（RUN-0063 的 view_frames 是关的）。
- seed_box_iou_gt：seed_box 与真值框（sample.gt["box"]）的框 IoU；seed_box_iou_own：与"自己的框"（W1-0 refexp_box 的 parsed.box）
  的框 IoU。都换成绝对像素、连续坐标（src/metrics/box 的同一口径）。
- first_snap_iou：第一次成功的贴边调用（SNAP_TOOLS：grabcut、watershed、select_superpixels）写出的 mask 的 IoU。
- w1f_iou、ownbox_iou：取自基线记录的 score（另记 W1-F 的交、并，算同一批句子上的 cIoU）。
- 失败码、结束方式、轮数、调用序列（tools，只记工具名）。
- last_mask：最后一次被写出、到结束时还在的 mask（"没有提交"时若自动交它，不用真值就能做到）。

IoU 的变化（每次写出 mask 的调用，每个被写的 mask 一条 delta）：Δ = 写出后的 IoU − 参照 mask 在调用前的 IoU。参照按先后取：
1. input：工具的输入 mask（INPUT_MASK_ARGS：refine / morph / components 的 name，grabcut 的 mask）；
2. same：同名的 mask 在调用前已存在（被覆盖或合并）；
3. working：调用前最后一次被写出、仍存在的 mask（换了名字重做时，与之前的"当前结果"比）；
4. 都没有：ref = None（这条样本的第一个 mask），不计升降。

分类（classify）：失败样本两种定义都报，F1：final_iou < 0.5；F2：final_iou < w1f_iou − 0.05（比 W1-F 差；没有 W1-F 记录的不计）。
每条失败样本按先后归入唯一一类（CLASSES）：
1. no_submission（没有提交）：失败码不为空；
2. degraded（修改把好的结果改坏）：best_iou − final_iou ≥ 0.10；
3. bad_seed（种子点放错）：seed_box_iou_gt < 0.5；没有种子（seed_box 为 None）的也归这一类，单独计数；
4. snap（贴边算法本身的问题）：其余。
阈值是 Thresholds 的字段（scripts/w1_replay.py 的命令行参数）。边界：比较时留 BOUNDARY_TOL 的余量，"x ≥ t"按 x ≥ t − tol 判，
"x < t"按 x < t − tol 判，所以恰好落在阈值上（含浮点误差，如 0.6 − 0.5 = 0.0999…）算达到阈值。
"""

from __future__ import annotations

import importlib
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.constants import NORM1000_SCALE
from src.executors.geometry import box_iou, norm1000_to_abs
from src.executors.raster import rle_to_mask
from src.metrics.seg import mask_inter_union, sample_iou
from src.w1.config import WORKBENCH_OPTIONS
from src.w1.loop import image_digest, tools_digest

REPLAY_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1
SNAP_TOOLS: tuple[str, ...] = ("grabcut", "watershed", "select_superpixels")  # 任务书定义的"贴边调用"
SEED_TOOL = "grabcut"
INPUT_MASK_ARGS: Mapping[str, str] = {"refine": "name", "morph": "name", "components": "name", "grabcut": "mask"}
REF_KINDS: tuple[str, ...] = ("input", "same", "working")
CLASSES: tuple[str, ...] = ("no_submission", "degraded", "bad_seed", "snap")
CLASS_LABELS: Mapping[str, str] = {"no_submission": "没有提交", "degraded": "修改把好的结果改坏", "bad_seed": "种子点放错",
                                   "snap": "贴边算法本身的问题"}
FAILURE_DEFS: tuple[str, ...] = ("F1", "F2")
BOUNDARY_TOL = 1e-9
SEED_SOURCE_GRABCUT = "grabcut_box"
SEED_SOURCE_MASK = "mask_bbox"
HIST_EDGES: tuple[float, ...] = tuple(i / 10 for i in range(11))  # 0, 0.1, …, 1.0（最后一格含 1.0）
QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


class ReplayError(RuntimeError):
    """重放的前提不成立（工具清单、原图尺寸、记录格式）：整条运行都不该继续。"""


# ---------------------------------------------------------------------------
# 重放
# ---------------------------------------------------------------------------


def load_factory(spec: str) -> Callable[..., Any]:
    """"MODULE:ATTR"（运行身份 config.yaml 的 workbench.factory）→ 工作台类。"""
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise ValueError(f"工作台应写成 MODULE:ATTR，收到 {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def workbench_kwargs(run_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """运行身份 → 构造工作台的关键字参数（与 run_episodes 相同：disabled_groups，加上与默认值不同的工作台选项）。"""
    opts = dict((run_cfg.get("workbench") or {}).get("options") or {})
    unknown = sorted(set(opts) - set(WORKBENCH_OPTIONS))
    if unknown:
        raise ReplayError(f"运行身份里有未知的工作台选项 {unknown}")
    kw: dict[str, Any] = {"disabled_groups": frozenset(run_cfg["protocol"]["disabled_groups"])}
    kw.update({k: v for k, v in opts.items() if v != WORKBENCH_OPTIONS[k]})
    return kw


def executed_calls(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """记录里按顺序执行过的调用（逐轮、轮内按出现顺序；只取 executed = true）。"""
    out = []
    for rd in record["rounds"]:
        for j, c in enumerate(rd["tool_calls"]):
            if c["executed"]:
                if c.get("result") is None:
                    raise ReplayError(f"{record['id']} 第 {rd['round']} 轮第 {j} 个调用 executed = true 却没有 result")
                out.append({"round": int(rd["round"]), "pos": j, "name": c["name"], "arguments": c["arguments"], "result": c["result"]})
    return out


def _parse_arg(value: Any, schema: Mapping[str, Any], pname: str) -> Any:
    """按工作台自己的解析规则读一个参数（src.w1.workbench._parse_value；成功的调用里它一定解析得了）。"""
    from src.w1.workbench import _parse_value

    return _parse_value(value, dict(schema), pname)


def _spec_param(wb: Any, tool: str, pname: str) -> Mapping[str, Any]:
    for s in wb.specs():
        if s.name == tool:
            return s.parameters["properties"][pname]
    raise ReplayError(f"工作台里没有工具 {tool!r}")


def seed_from_grabcut(wb: Any, arguments: Mapping[str, Any]) -> list[float] | None:
    """成功的 grabcut 调用的 box 参数 → 原图 0–1000（给了 view 且 view_frames 开时按那张视图换算）。用 mask 初始化的返回 None。"""
    if "box" not in arguments:
        return None
    box = [float(v) for v in _parse_arg(arguments["box"], _spec_param(wb, SEED_TOOL, "box"), "box")]
    if "view" in arguments and getattr(wb, "view_frames", False):
        from src.w1.workbench import view_point_to_full

        region = wb.views[int(_parse_arg(arguments["view"], _spec_param(wb, SEED_TOOL, "view"), "view"))]
        box = [*view_point_to_full(region, box[0], box[1]), *view_point_to_full(region, box[2], box[3])]
    return box


def mask_bbox_abs(m: np.ndarray) -> list[float] | None:
    """mask 的外接框（绝对像素，按像素边缘：占列 c0..c1 → [c0, c1 + 1)）；空 mask 为 None。"""
    m = np.asarray(m)
    if not m.any():
        return None
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def abs_to_norm1000(box_abs: Sequence[float], w: int, h: int) -> list[float]:
    """绝对像素 xyxy → 原图 0–1000（与工作台 _px_to_norm 同一换算）。"""
    return [box_abs[0] / w * NORM1000_SCALE, box_abs[1] / h * NORM1000_SCALE, box_abs[2] / w * NORM1000_SCALE,
            box_abs[3] / h * NORM1000_SCALE]


def safe_box_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    """两个绝对像素框的 IoU（src.executors.geometry.box_iou）；任一为 None 或框不合法时为 None。"""
    if a is None or b is None:
        return None
    try:
        return float(box_iou(a, b))
    except ValueError:
        return None


@dataclass
class _MaskStat:
    arr: np.ndarray  # 工作台里的那个数组本身（工作台每写一次换一个新数组，用 is 判断是否被写过）
    inter: int
    union: int
    area: int

    @property
    def iou(self) -> float:
        return sample_iou(self.inter, self.union)


def _snapshot(wb: Any, gt: np.ndarray, prev: Mapping[str, _MaskStat]) -> tuple[dict[str, _MaskStat], list[str]]:
    """工作台当前的 mask → 统计（没变的复用上一次的）；返回 (统计, 这次被写的 mask 名，按工作台的顺序)。"""
    cur: dict[str, _MaskStat] = {}
    written: list[str] = []
    for name, arr in wb.masks.items():
        old = prev.get(name)
        if old is not None and old.arr is arr:
            cur[name] = old
            continue
        st = mask_inter_union(np.asarray(arr), gt)
        cur[name] = _MaskStat(arr, st.inter, st.union, int(np.count_nonzero(arr)))
        written.append(name)
    return cur, written


def replay_episode(record: Mapping[str, Any], image: Any, gt: np.ndarray, *, gt_box: Sequence[float],
                   workbench_factory: Callable[..., Any], workbench_kw: Mapping[str, Any],
                   own_box: Sequence[float] | None = None, own_box_iou_gt: float | None = None,
                   w1f_score: Mapping[str, Any] | None = None, ownbox_score: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """一条 W1 回路记录（predictions.jsonl 的一行）重放 → replay.jsonl 的一行（键的含义见模块说明）。

    image：原图（PIL）；gt：真值 mask（(H, W) bool）；gt_box：真值框（绝对像素 xyxy）；own_box：自己的框（绝对像素 xyxy，框解析失败时
    None）；own_box_iou_gt：自己的框与真值框的 IoU（W1-0 refexp_box 的 score.iou）；w1f_score、ownbox_score：基线记录的 score。
    calls[k]：第 k 次重放的调用之后的状态：tool、ok、written（被写的 mask）、deltas（见模块说明）、masks（mask 名 → [IoU, 面积]）。
    """
    rid = record["id"]
    w, h = image.size
    if list(record["image_size"]) != [w, h]:
        raise ReplayError(f"{rid}：记录的 image_size {record['image_size']} 与原图 {[w, h]} 不同")
    if gt.shape != (h, w):
        raise ReplayError(f"{rid}：真值 mask 形状 {gt.shape} 与原图 (H, W) = {(h, w)} 不同")
    wb = workbench_factory(image.copy(), **dict(workbench_kw))
    sha = tools_digest(wb.specs())
    if sha != record["tools_sha256"]:
        raise ReplayError(f"{rid}：工作台的工具清单 sha256 {sha[:12]}… 与记录的 {record['tools_sha256'][:12]}… 不同（不是同一版工作台）")
    gt_area = int(np.count_nonzero(gt))
    reasons: list[str] = []
    n_image_mismatch = 0
    stats: dict[str, _MaskStat] = {}
    calls_out: list[dict[str, Any]] = []
    working: str | None = None  # 最后一次被写出、仍存在的 mask
    seed: dict[str, Any] | None = None  # grabcut 的框
    first_mask: dict[str, Any] | None = None  # 第一个被写出的非空 mask
    first_snap: dict[str, Any] | None = None
    best = {"iou": 0.0, "call": None, "mask": None, "inter": 0, "union": gt_area}
    for k, c in enumerate(executed_calls(record)):
        name, args = c["name"], c["arguments"]
        before = stats
        out = wb.call(name, dict(args))
        if out.text != c["result"]["text"]:
            reasons.append(f"text@{k}:{name}")
        if [image_digest(im) for im in out.images] != list(c["result"].get("images") or []):
            n_image_mismatch += 1
        stats, written = _snapshot(wb, gt, before)
        ok = out.error is None
        deltas = []
        for m in written:
            ref = kind = None
            in_arg = INPUT_MASK_ARGS.get(name)
            if in_arg and in_arg in args:
                in_name = _parse_arg(args[in_arg], {"type": "string"}, in_arg)
                if in_name in before:
                    ref, kind = in_name, "input"
            if ref is None and m in before:
                ref, kind = m, "same"
            if ref is None and working is not None and working in before:
                ref, kind = working, "working"
            deltas.append({"mask": m, "ref": ref, "ref_kind": kind, "iou_before": before[ref].iou if ref is not None else None,
                           "iou_after": stats[m].iou})
            if first_mask is None and stats[m].area > 0:
                first_mask = {"call": k, "tool": name, "box_abs": mask_bbox_abs(stats[m].arr)}
        if written:
            working = written[-1]
        if working is not None and working not in stats:  # 被 delete_mask 删了
            working = None
        if ok and name == SEED_TOOL and seed is None:
            sb = seed_from_grabcut(wb, args)
            if sb is not None:
                seed = {"call": k, "box": sb}
        if ok and first_snap is None and name in SNAP_TOOLS and written:
            first_snap = {"call": k, "tool": name, "iou": stats[written[-1]].iou}
        for m, st in stats.items():
            if best["call"] is None or st.iou > best["iou"]:
                best = {"iou": st.iou, "call": k, "mask": m, "inter": st.inter, "union": st.union}
        calls_out.append({"k": k, "round": c["round"], "tool": name, "ok": ok, "written": written, "deltas": deltas,
                          "masks": {m: [st.iou, st.area] for m, st in stats.items()}})
    if seed is not None:
        seed_box, seed_source, seed_call = seed["box"], SEED_SOURCE_GRABCUT, seed["call"]
    elif first_mask is not None:
        seed_box = abs_to_norm1000(first_mask["box_abs"], w, h)
        seed_source, seed_call = f"{SEED_SOURCE_MASK}:{first_mask['tool']}", first_mask["call"]
    else:
        seed_box = seed_source = seed_call = None
    # 提交核对
    ans = record.get("answer") or {}
    rec_mask = rle_to_mask(ans["mask"]["rle"]) if ans.get("mask") else None
    sub = wb.submission
    rep_mask = getattr(sub, "mask", None) if sub is not None else None
    if (rec_mask is None) != (rep_mask is None) or (rec_mask is not None and not np.array_equal(rec_mask, np.asarray(rep_mask))):
        reasons.append("submission")
    failure = record["failure"]
    if failure is None and rec_mask is None:
        raise ReplayError(f"{rid}：没有失败码却没有提交的 mask")
    fst = mask_inter_union(rec_mask if failure is None else None, gt)
    sc = record.get("score") or {}
    if (sc.get("inter"), sc.get("union")) != (fst.inter, fst.union):
        reasons.append("score")
    last = stats.get(working) if working is not None else None
    seed_abs = norm1000_to_abs(np.asarray(seed_box, dtype=np.float64), width=w, height=h).tolist() if seed_box is not None else None
    return {
        "schema": REPLAY_SCHEMA_VERSION, "id": rid, "failure": failure, "ended_by": record["ended_by"], "n_rounds": record["n_rounds"],
        "n_executed": len(calls_out), "n_refused": record.get("n_refused", 0), "n_tool_call_errors": record.get("n_tool_call_errors", 0),
        "tools": [c["tool"] for c in calls_out], "image_size": [w, h], "gt_area": gt_area,
        "replay": {"mismatch": bool(reasons), "reasons": reasons, "n_calls": len(calls_out), "n_image_mismatch": n_image_mismatch},
        "final_iou": sample_iou(*fst), "final_inter": fst.inter, "final_union": fst.union,
        "submitted_mask": getattr(sub, "source", None) if rep_mask is not None else None,
        "best_iou": best["iou"], "best_call": best["call"], "best_mask": best["mask"], "best_inter": best["inter"],
        "best_union": best["union"],
        "last_mask": working if last is not None else None, "last_iou": last.iou if last is not None else 0.0,
        "last_inter": last.inter if last is not None else 0, "last_union": last.union if last is not None else gt_area,
        "seed_box": seed_box, "seed_box_source": seed_source, "seed_call": seed_call,
        "seed_box_iou_gt": safe_box_iou(seed_abs, gt_box), "seed_box_iou_own": safe_box_iou(seed_abs, own_box),
        "own_box": list(own_box) if own_box is not None else None, "own_box_iou_gt": own_box_iou_gt,
        "first_snap_iou": first_snap["iou"] if first_snap else None, "first_snap_call": first_snap["call"] if first_snap else None,
        "first_snap_tool": first_snap["tool"] if first_snap else None,
        "w1f_iou": None if w1f_score is None else float(w1f_score["iou"]),
        "w1f_inter": None if w1f_score is None else int(w1f_score["inter"]),
        "w1f_union": None if w1f_score is None else int(w1f_score["union"]),
        "ownbox_iou": None if ownbox_score is None else float(ownbox_score["iou"]),
        "calls": calls_out,
    }


def replay_job(job: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """scripts/w1_replay.py 的一条任务（可 pickle，--workers 的子进程里跑）：读原图与真值（E1 的 load_images、gt_mask，同运行时），
    构造工作台类，重放。返回 (replay.jsonl 的一行, 墙钟秒)。"""
    import time

    from src.e1.runner import load_images
    from src.e1.scoring import gt_mask

    t0 = time.perf_counter()
    s = job["sample"]
    ims = load_images(s)
    if len(ims) != 1:
        raise ReplayError(f"{s.id}：W1 的任务都是单图，收到 {len(ims)} 张")
    row = replay_episode(job["record"], ims[0], gt_mask(s), gt_box=s.gt["box"], workbench_factory=load_factory(job["factory"]),
                         workbench_kw=job["workbench_kw"], own_box=job.get("own_box"), own_box_iou_gt=job.get("own_box_iou_gt"),
                         w1f_score=job.get("w1f_score"), ownbox_score=job.get("ownbox_score"))
    return row, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """分类的阈值（scripts/w1_replay.py 的命令行参数，默认值即任务书 DEV-0057 的定义）。"""

    f1_iou: float = 0.5  # F1：final_iou < f1_iou
    f2_margin: float = 0.05  # F2：final_iou < w1f_iou − f2_margin
    degrade: float = 0.10  # 修改把好的结果改坏：best_iou − final_iou ≥ degrade
    seed_iou: float = 0.5  # 种子点放错：seed_box_iou_gt < seed_iou
    delta_eps: float = 0.0  # IoU 的变化：|Δ| ≤ delta_eps（再加 BOUNDARY_TOL）记"不变"

    def __post_init__(self) -> None:
        for k, v in asdict(self).items():
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"阈值 {k} 必须是非负有限数，收到 {v!r}")


def _ge(x: float, t: float) -> bool:
    return x >= t - BOUNDARY_TOL


def _lt(x: float, t: float) -> bool:
    return x < t - BOUNDARY_TOL


def is_failure(row: Mapping[str, Any], definition: str, th: Thresholds) -> bool | None:
    """失败与否；F2 没有 W1-F 记录时为 None（不计）。"""
    if definition == "F1":
        return _lt(row["final_iou"], th.f1_iou)
    if definition == "F2":
        if row.get("w1f_iou") is None:
            return None
        return _lt(row["final_iou"], row["w1f_iou"] - th.f2_margin)
    raise ValueError(f"未知的失败定义 {definition!r}；可用 {FAILURE_DEFS}")


def classify_row(row: Mapping[str, Any], th: Thresholds) -> str:
    """一条失败样本 → 唯一一类（先后见模块说明）。"""
    if row["failure"] is not None:
        return "no_submission"
    if _ge(row["best_iou"] - row["final_iou"], th.degrade):
        return "degraded"
    if row.get("seed_box_iou_gt") is None or _lt(row["seed_box_iou_gt"], th.seed_iou):
        return "bad_seed"
    return "snap"


def _median(xs: Iterable[float | None]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return statistics.median(v) if v else None


def _ratio(inter: int, union: int) -> float | None:
    return inter / union if union else None


def summarize(values: Sequence[float | None]) -> dict[str, Any]:
    """分布：条数、缺失数、均值、最小、分位数（numpy linear）、最大、直方图（HIST_EDGES，最后一格含右端）。"""
    v = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
    out: dict[str, Any] = {"n": int(v.size), "n_missing": sum(1 for x in values if x is None)}
    if v.size == 0:
        return out
    out.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
               quantiles={f"p{int(q * 100)}": float(np.quantile(v, q)) for q in QUANTILES})
    counts, _ = np.histogram(np.clip(v, HIST_EDGES[0], HIST_EDGES[-1]), bins=np.asarray(HIST_EDGES))
    out["hist"] = {"edges": list(HIST_EDGES), "counts": [int(c) for c in counts]}
    return out


def _class_block(rows: Sequence[Mapping[str, Any]], th: Thresholds) -> dict[str, Any]:
    by: dict[str, list[Mapping[str, Any]]] = {c: [] for c in CLASSES}
    for r in rows:
        by[classify_row(r, th)].append(r)
    n = len(rows)
    out = {}
    for c in CLASSES:
        rs = by[c]
        out[c] = {"label": CLASS_LABELS[c], "n": len(rs), "share": len(rs) / n if n else None,
                  "median_best_iou": _median(r["best_iou"] for r in rs),
                  "median_seed_box_iou_gt": _median(r.get("seed_box_iou_gt") for r in rs),
                  "median_w1f_iou": _median(r.get("w1f_iou") for r in rs),
                  "median_final_iou": _median(r["final_iou"] for r in rs),
                  "n_seed_missing": sum(1 for r in rs if r.get("seed_box") is None),
                  "ids": [r["id"] for r in rs]}
    return out


def _fallback_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一批样本上：实际 cIoU、交过程中最好的 mask（要真值挑）、交最后写出的 mask（不要真值）、W1-F 的 cIoU。"""
    s = lambda a, b: _ratio(sum(int(r[a]) for r in rows), sum(int(r[b]) for r in rows))  # noqa: E731
    w1f = [r for r in rows if r.get("w1f_inter") is not None]
    return {"n": len(rows), "ciou_actual": s("final_inter", "final_union"), "ciou_if_best": s("best_inter", "best_union"),
            "ciou_if_last": s("last_inter", "last_union"),
            "ciou_w1f": _ratio(sum(r["w1f_inter"] for r in w1f), sum(r["w1f_union"] for r in w1f)) if len(w1f) == len(rows) else None,
            "n_with_any_mask": sum(1 for r in rows if r["best_mask"] is not None)}


def tool_iou_changes(rows: Sequence[Mapping[str, Any]], th: Thresholds) -> dict[str, Any]:
    """每种工具：执行次数、成功次数、写出 mask 的次数；Δ 的升、降、不变（|Δ| ≤ delta_eps）、无参照的条数，Δ 的中位数与均值，
    以及按参照种类（input / same / working）分开的条数与中位 Δ。"""
    acc: dict[str, dict[str, Any]] = {}
    for r in rows:
        for c in r["calls"]:
            a = acc.setdefault(c["tool"], {"n_executed": 0, "n_ok": 0, "n_wrote": 0, "deltas": [], "no_ref": 0, "after": [],
                                           "by_ref": {k: [] for k in REF_KINDS}})
            a["n_executed"] += 1
            a["n_ok"] += int(bool(c["ok"]))
            a["n_wrote"] += int(bool(c["written"]))
            for d in c["deltas"]:
                a["after"].append(d["iou_after"])
                if d["ref"] is None:
                    a["no_ref"] += 1
                    continue
                dv = d["iou_after"] - d["iou_before"]
                a["deltas"].append(dv)
                a["by_ref"][d["ref_kind"]].append(dv)
    out = {}
    eps = th.delta_eps + BOUNDARY_TOL
    for tool in sorted(acc, key=lambda t: (-acc[t]["n_executed"], t)):
        a = acc[tool]
        ds = a["deltas"]
        out[tool] = {"n_executed": a["n_executed"], "n_ok": a["n_ok"], "n_wrote": a["n_wrote"], "n_masks_written": len(a["after"]),
                     "up": sum(1 for d in ds if d > eps), "down": sum(1 for d in ds if d < -eps),
                     "same": sum(1 for d in ds if abs(d) <= eps), "no_ref": a["no_ref"],
                     "median_delta": _median(ds), "mean_delta": float(np.mean(ds)) if ds else None,
                     "median_iou_after": _median(a["after"]),
                     "by_ref": {k: {"n": len(v), "median_delta": _median(v)} for k, v in a["by_ref"].items() if v}}
    return out


def classify(rows: Sequence[Mapping[str, Any]], th: Thresholds = Thresholds(), *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """replay.jsonl 的行 → error_analysis.json 的内容。重放不一致的行不进分类（只报条数与 id）。"""
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("replay.jsonl 里有重复的 id")
    bad = [r for r in rows if r["replay"]["mismatch"]]
    good = [r for r in rows if not r["replay"]["mismatch"]]
    doc: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA_VERSION, "source": dict(source or {}), "thresholds": asdict(th), "boundary_tol": BOUNDARY_TOL,
        "rules": {
            "F1": f"final_iou < {th.f1_iou:g}", "F2": f"final_iou < w1f_iou − {th.f2_margin:g}（没有 W1-F 记录的不计）",
            "order": [f"1 no_submission：失败码不为空", f"2 degraded：best_iou − final_iou ≥ {th.degrade:g}",
                      f"3 bad_seed：seed_box_iou_gt < {th.seed_iou:g}（没有种子的也算）", "4 snap：其余"],
            "boundary": f"比较留 {BOUNDARY_TOL:g} 的余量：恰好落在阈值上算达到阈值"},
        "n_rows": len(rows), "n_replay_mismatch": len(bad),
        "replay_mismatch": [{"id": r["id"], "reasons": r["replay"]["reasons"]} for r in bad],
        "n_image_mismatch_calls": sum(r["replay"]["n_image_mismatch"] for r in rows),
        "n_analyzed": len(good),
        "overall": {**_fallback_block(good), "n_valid_submission": sum(1 for r in good if r["failure"] is None),
                    "failures": dict(sorted(Counter("ok" if r["failure"] is None else r["failure"] for r in good).items())),
                    "seed_box_source": dict(sorted(Counter(
                        "none" if r["seed_box_source"] is None else r["seed_box_source"] for r in good).items()))},
    }
    defs: dict[str, Any] = {}
    for d in FAILURE_DEFS:
        flags = [(r, is_failure(r, d, th)) for r in good]
        failed = [r for r, f in flags if f]
        defs[d] = {"rule": doc["rules"][d], "n_failed": len(failed), "n_excluded": sum(1 for _, f in flags if f is None),
                   "share_of_analyzed": len(failed) / len(good) if good else None, "classes": _class_block(failed, th),
                   "no_submission_fallback": _fallback_block([r for r in failed if classify_row(r, th) == "no_submission"])}
    doc["failure_defs"] = defs
    with_seed = [r for r in good if r.get("seed_box_iou_own") is not None]
    doc["seed_box"] = {
        "seed_box_iou_own": summarize([r.get("seed_box_iou_own") for r in good]),
        "seed_box_iou_gt": summarize([r.get("seed_box_iou_gt") for r in good]),
        "own_box_iou_gt": summarize([r.get("own_box_iou_gt") for r in good]),
        "paired": {"n": len(with_seed),
                   "median_seed_box_iou_gt": _median(r.get("seed_box_iou_gt") for r in with_seed),
                   "median_own_box_iou_gt": _median(r.get("own_box_iou_gt") for r in with_seed),
                   "median_diff_seed_minus_own": _median(r["seed_box_iou_gt"] - r["own_box_iou_gt"] for r in with_seed
                                                         if r.get("seed_box_iou_gt") is not None and r.get("own_box_iou_gt") is not None),
                   "n_seed_worse": sum(1 for r in with_seed if r.get("seed_box_iou_gt") is not None and r.get("own_box_iou_gt") is not None
                                       and r["seed_box_iou_gt"] < r["own_box_iou_gt"]),
                   "acc50_seed": _acc50([r.get("seed_box_iou_gt") for r in with_seed]),
                   "acc50_own": _acc50([r.get("own_box_iou_gt") for r in with_seed])},
    }
    doc["no_submission_all"] = _fallback_block([r for r in good if r["failure"] is not None])
    doc["tool_iou_change"] = tool_iou_changes(good, th)
    return doc


def _acc50(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(1 for x in v if _ge(x, 0.5)) / len(v) if v else None


# ---------------------------------------------------------------------------
# error_analysis.md
# ---------------------------------------------------------------------------


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def render_md(doc: Mapping[str, Any]) -> str:
    th = doc["thresholds"]
    src = doc.get("source") or {}
    L = ["# W1 错误分析（DEV-0057；D113-4）", ""]
    if src:
        lim = src.get("limit")
        L += [f"- 来源：`{src.get('run', '—')}`，模型 {src.get('model', '—')}，{src.get('task', 'refseg')}/{src.get('setting', '—')}/"
              f"{src.get('split', '—')}；回路记录 sha256 {str(src.get('predictions_sha256', '—'))[:16]}…"
              + (f"；**只重放了前 {lim} 条（--limit，调试用）**" if lim else "")]
    L += [f"- 阈值：F1 final_iou < {th['f1_iou']:g}；F2 final_iou < w1f_iou − {th['f2_margin']:g}；改坏 best − final ≥ {th['degrade']:g}；"
          f"种子放错 seed_box_iou_gt < {th['seed_iou']:g}；IoU 不变 |Δ| ≤ {th['delta_eps']:g}。恰好落在阈值上算达到阈值"
          f"（比较留 {doc['boundary_tol']:g} 的余量）。",
          "- 分类先后：没有提交 → 修改把好的结果改坏 → 种子点放错 → 贴边算法本身的问题（每条只进一类）。",
          f"- 重放：{doc['n_rows']} 条，不一致 {doc['n_replay_mismatch']} 条（不进分类）；返回图的 sha256 不同的调用 "
          f"{doc['n_image_mismatch_calls']} 次（只计数）。进入分析 {doc['n_analyzed']} 条。", ""]
    if doc["replay_mismatch"]:
        L += ["不一致的条目（前 20 条）：", ""] + [f"- {m['id']}：{', '.join(m['reasons'])}" for m in doc["replay_mismatch"][:20]] + [""]
    ov = doc["overall"]
    L += ["## 总体", "",
          f"- 有效提交 {ov['n_valid_submission']}/{ov['n']}；失败码 {ov['failures']}",
          f"- cIoU：实际 {_f(ov['ciou_actual'])}；若交过程中最好的 mask {_f(ov['ciou_if_best'])}；若交最后写出的 mask "
          f"{_f(ov['ciou_if_last'])}；W1-F {_f(ov['ciou_w1f'])}",
          f"- 种子框的来源：{ov['seed_box_source']}", ""]
    for d in FAILURE_DEFS:
        b = doc["failure_defs"][d]
        L += [f"## {d}：{b['rule']}", "",
              f"失败 {b['n_failed']} 条（占进入分析的 {_pct(b['share_of_analyzed'])}；不计 {b['n_excluded']} 条）。", "",
              "| 类 | 条数 | 占比 | best_iou 中位 | seed_box_iou_gt 中位 | w1f_iou 中位 | final_iou 中位 | 没有种子 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in CLASSES:
            x = b["classes"][c]
            L.append(f"| {x['label']} | {x['n']} | {_pct(x['share'])} | {_f(x['median_best_iou'])} | {_f(x['median_seed_box_iou_gt'])} | "
                     f"{_f(x['median_w1f_iou'])} | {_f(x['median_final_iou'])} | {x['n_seed_missing']} |")
        fb = b["no_submission_fallback"]
        L += ["", f"\"没有提交\"这一类（{fb['n']} 条）：若交过程中最好的 mask，cIoU {_f(fb['ciou_if_best'])}；若交最后写出的 mask "
                  f"{_f(fb['ciou_if_last'])}；同一批句子上 W1-F {_f(fb['ciou_w1f'])}；其中 {fb['n_with_any_mask']} 条过程中有过 mask。", ""]
    ns = doc["no_submission_all"]
    L += [f"全部没有有效提交的 {ns['n']} 条：若交最好的 mask，cIoU {_f(ns['ciou_if_best'])}；若交最后写出的 mask {_f(ns['ciou_if_last'])}；"
          f"W1-F {_f(ns['ciou_w1f'])}。", ""]
    sb = doc["seed_box"]
    L += ["## 写进工具的框（seed_box）", "", "| 量 | n | 缺失 | 均值 | p10 | p25 | 中位 | p75 | p90 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, label in (("seed_box_iou_own", "种子框 vs 自己的框"), ("seed_box_iou_gt", "种子框 vs 真值框"), ("own_box_iou_gt", "自己的框 vs 真值框")):
        s = sb[k]
        q = s.get("quantiles") or {}
        L.append(f"| {label} | {s['n']} | {s['n_missing']} | {_f(s.get('mean'))} | {_f(q.get('p10'))} | {_f(q.get('p25'))} | "
                 f"{_f(q.get('p50'))} | {_f(q.get('p75'))} | {_f(q.get('p90'))} |")
    h = sb["seed_box_iou_own"].get("hist")
    if h:
        L += ["", "种子框 vs 自己的框，IoU 直方图（左闭右开，最后一格含 1.0）：", "",
              "| " + " | ".join(f"{h['edges'][i]:.1f}–{h['edges'][i + 1]:.1f}" for i in range(len(h["counts"]))) + " |",
              "|" + "---:|" * len(h["counts"]), "| " + " | ".join(str(c) for c in h["counts"]) + " |"]
    p = sb["paired"]
    L += ["", "（自己的框解析失败的，与真值框的 IoU 记 0，同 E1；没有自己的框时\"种子框 vs 自己的框\"记缺失。）", "",
          f"同一批 {p['n']} 条（种子框与自己的框都有）：与真值框的 IoU 中位，种子框 {_f(p['median_seed_box_iou_gt'])}，自己的框 "
          f"{_f(p['median_own_box_iou_gt'])}；逐条差（种子框 − 自己的框）中位 {_f(p['median_diff_seed_minus_own'])}，种子框更差的 "
          f"{p['n_seed_worse']} 条；Acc@0.5，种子框 {_pct(p['acc50_seed'])}，自己的框 {_pct(p['acc50_own'])}。", ""]
    L += ["## 每种工具调用之后 IoU 的变化", "",
          "Δ = 写出的 mask 的 IoU − 参照 mask 在调用前的 IoU；参照：工具的输入 mask（refine / morph / components / 用 mask 初始化的 grabcut）"
          " → 同名的旧 mask → 调用前最后写出的 mask；都没有的记\"无参照\"。", "",
          "| 工具 | 执行 | 成功 | 写出 mask | 升 | 降 | 不变 | 无参照 | Δ 中位 | Δ 均值 | 写出后 IoU 中位 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tool, x in doc["tool_iou_change"].items():
        L.append(f"| {tool} | {x['n_executed']} | {x['n_ok']} | {x['n_wrote']} | {x['up']} | {x['down']} | {x['same']} | {x['no_ref']} | "
                 f"{_f(x['median_delta'])} | {_f(x['mean_delta'])} | {_f(x['median_iou_after'])} |")
    L += ["", "各类的样本 id 见 error_analysis.json（failure_defs.<F1|F2>.classes.<类>.ids）。", ""]
    return "\n".join(L)


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "BOUNDARY_TOL",
    "CLASSES",
    "CLASS_LABELS",
    "FAILURE_DEFS",
    "INPUT_MASK_ARGS",
    "REPLAY_SCHEMA_VERSION",
    "ReplayError",
    "SNAP_TOOLS",
    "Thresholds",
    "abs_to_norm1000",
    "classify",
    "classify_row",
    "executed_calls",
    "is_failure",
    "load_factory",
    "mask_bbox_abs",
    "render_md",
    "replay_episode",
    "replay_job",
    "safe_box_iou",
    "seed_from_grabcut",
    "summarize",
    "tool_iou_changes",
    "workbench_kwargs",
]
