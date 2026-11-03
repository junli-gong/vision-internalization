"""X-M 的指标（DEV-0080；指令第四节；预注册 XM-3a）：LLM 不给置信度，用基于匹配的 F1。纯函数，供运行器（scripts/run_xm.py）与判定脚本
（DEV-0081）共用。

**匹配**（match；每个"图像 × 类别"对内，预测按模型的输出顺序，至多 MAX_DETS 个）：
- 缺省 MATCHING = "coco"（XM-3a，Rex-Omni 的 evaluation/metrics/coco_lvis_metric.py 与它移植的 COCOeval）：逐个预测按输出顺序（输出顺序
  就是分数顺序），在没被占用的非 crowd 真值里找 IoU ≥ t 且最大的一个（同值取后一个，COCOeval 的 `if iou < best: continue`）；一个
  真值至多配一次，所以重复的框记 FP。真值表里 crowd 区域排在非 crowd 之后，可以被重复配；预测配上 crowd 区域（COCOeval 的 iscrowd
  口径：IoA = 交 / 预测面积 ≥ t）就忽略，不记 FP；已经配上非 crowd 真值之后不再看 crowd（COCOeval 的 break）。逐行照 COCOeval
  evaluateImg 写（含阈值取 min(t, 1 − 1e−10)），tests/test_xm_metrics.py 用手算例子钉住。
- 可选（任务书原文的两种）："greedy_iou"：全部 IoU ≥ t 的（预测, 真值）按 IoU 从高到低一对一（同值按预测、真值编号小的先）；
  "hungarian"：先使配上的对数最多、再使 IoU 之和最大（scipy linear_sum_assignment）。两种都在一对一之后，把没配上、且与某个 crowd
  区域的 IoA ≥ t 的预测忽略。
- LVIS 的非穷尽类别（not_exhaustive = True 的对）：没配上的预测一律忽略（LVIS 官方惯例），只记 TP 与 FN。
- 检测用框 IoU（pycocotools mask.iou，连续坐标，与 COCOeval 相同；crowd 区域取它的 RLE 的外接框，同 COCO 的 crowd 标注），
  实例分割用 mask IoU（同一个函数，RLE）。
- 每个对在 IOU_THRESHOLDS（COCOeval 的 0.5:0.05:0.95，10 个）上各数一遍 TP、FP、FN、忽略数；HEADLINE_IOU = 0.5 那一档另记配上的 IoU。

**数据集读数**（readout）：
- 主读数 f1_rexomni（XM-3a）：每个类别 c 在全部对上合计 TP_c、FP_c、FN_c；R_c = TP_c / (TP_c + FN_c)，P_c = TP_c / (TP_c + FP_c)
  （分母为 0 时这一类不进相应的平均）；宏平均 P = mean_c P_c、R = mean_c R_c，F1 = 2PR / (P + R)（P + R = 0 记 0；没有任何预测时
  P 无定义、F1 记 0）。不照搬 Rex-Omni 的 safe_mean（丢掉 ≤ 0 的值）与它的 precision 不清零。
- f1_rexomni_miou：每个阈值各算宏平均 P(t)、R(t)，先对 10 个阈值平均 P 与 R，再取调和平均。
- f1_rexomni_exact、f1_rexomni_exact_miou（诊断，只为与 Rex-Omni 报的数可比；只给点估计，不算 CI，不进判定）：照 Rex-Omni 的
  evaluation/fastevaluate 的 cocoeval.cpp（COCOEval::accumulate）与 coco_lvis_metric.py 复现（rexomni_exact）：
  - 每个类别把各对的预测按（对的顺序, 输出顺序）连起来（Rex-Omni 的分数都是常数 0.2，这里假定排序稳定）；P_c 是 COCO 的插值精度
    （单调包络）在召回阈值 0:0.01:1 里最高的、精度 > 0 的那一档的值，**不是** TP_c / (TP_c + FP_c)；R_c = 最后的累计 TP / 真值数；
  - precision 不清零：某类在某一档没有 > 0 的精度时，沿用上一个类别（IoU 0.5 那一列）或上一档（mIoU 那一列；每类第一档沿用上一类最后
    一档）的值；第一个类别就没有时记 0（C++ 里是未初始化的值）；
  - safe_mean：P、R 各自只平均 > 0 的值，F1 = 2PR / (P + R)。
- 次读数 f1_micro = 2ΣTP / (2ΣTP + ΣFP + ΣFN)（各对求和，IoU 0.5），另报 precision_micro、recall_micro；matched_mean_iou（本项目
  自己的读数，不是 Rex-Omni 的）= IoU 0.5 下配上的对的 IoU 平均。

**置信区间与配对差**（bootstrap_readout、paired_bootstrap）：按**图像**重抽（同一张图的各个类别一起抽），percentile 法；重抽下标与
src/stats._resample_indices 相同（np.random.default_rng(seed).integers(0, n, (B, n))），每次重抽在被抽到的图上重新合计、重新算宏平均。
运行时的 metrics.json 用 src/constants.BOOTSTRAP_N_RESAMPLES（1,000）次；判定用 DECISION_N_RESAMPLES（10,000）次（指令第五节）。
结果与 src.stats.bootstrap_ci / paired_bootstrap_ci（statistic = 本模块的统计量）逐位相同（tests/test_xm_metrics.py 核对）：计数都是整数，
按图合计用矩阵乘法也是精确的，之后对每次重抽调用同一个标量函数。

**指代分割**（refseg_readout）：ReasonSeg 走 X-2 的运行器（src/explore/x2_runner），W2 格式的 predictions.jsonl 每行有 score.inter / union；
gIoU、cIoU 同 W2 的口径（src/metrics/seg）。

只 import numpy、scipy、pycocotools 与本仓库的统计函数（不含学习）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from src import stats as S
from src.constants import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, DETECTION_OPERATING_IOU
from src.metrics.seg import ciou, giou

METRICS_VERSION = "xm-metrics-v2"  # v2：XM-3a（Rex-Omni 的宏平均 F1 为主读数，COCOeval 式匹配）
MATCHING = "coco"  # 缺省（XM-3a）；可选见 MATCHINGS
MATCHINGS: tuple[str, ...] = ("coco", "greedy_iou", "hungarian")
MAX_DETS = 100  # COCOeval 的 maxDets（每个对）
HEADLINE_IOU = DETECTION_OPERATING_IOU  # 0.5
# COCOeval 的 iouThrs：np.linspace(.5, 0.95, round((0.95 − .5) / .05) + 1)（不取整，0.9 那一档是 0.8999999999999999）
IOU_THRESHOLDS: tuple[float, ...] = tuple(float(t) for t in np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True))
HEADLINE_INDEX = 0
assert IOU_THRESHOLDS[HEADLINE_INDEX] == HEADLINE_IOU
DECISION_N_RESAMPLES = 10_000  # 判定（指令第五节：配对 bootstrap，95%，10,000 次）
KINDS: tuple[str, ...] = ("det", "seg")  # det：框 IoU；seg：mask IoU
STATISTICS: tuple[str, ...] = ("f1_rexomni", "f1_rexomni_miou", "f1_micro", "matched_mean_iou")  # 有 CI、可配对
EXACT_STATISTICS: tuple[str, ...] = ("f1_rexomni_exact", "f1_rexomni_exact_miou")  # 诊断：只给点估计
REC_THRS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)  # COCOeval 的 recThrs（101 档）
PRIMARY = "f1_rexomni"
COUNT_KEYS: tuple[str, ...] = ("tp", "fp", "fn", "ignored")


def identity() -> dict[str, Any]:
    """进运行身份的指标口径。"""
    return {"version": METRICS_VERSION, "matching": MATCHING, "max_dets": MAX_DETS, "headline_iou": HEADLINE_IOU,
            "iou_thresholds": list(IOU_THRESHOLDS), "primary": PRIMARY, "bootstrap_unit": "image",
            "runtime_resamples": BOOTSTRAP_N_RESAMPLES, "decision_resamples": DECISION_N_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "level": BOOTSTRAP_CI_LEVEL}


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def coco_rle(rle: Mapping[str, Any]) -> dict[str, Any]:
    """COCO RLE（压缩：counts 为 str / bytes；未压缩：counts 为整数列表，如 COCO 的 crowd 标注）→ pycocotools 的压缩 RLE。"""
    h, w = (int(v) for v in rle["size"])
    c = rle["counts"]
    if isinstance(c, list):
        return mask_utils.frPyObjects({"size": [h, w], "counts": c}, h, w)
    return {"size": [h, w], "counts": c.encode("ascii") if isinstance(c, str) else c}


def coco_rle_json(rle: Mapping[str, Any]) -> dict[str, Any]:
    """任何 COCO RLE → src/executors/raster.rle_to_mask 能读的形状（未压缩的整数 counts 原样；压缩的 counts 转成 str）。"""
    c = rle["counts"]
    return {"size": [int(v) for v in rle["size"]], "counts": c.decode("ascii") if isinstance(c, bytes) else c}


def _xywh(boxes: Sequence[Sequence[float]]) -> list[list[float]]:
    out = []
    for b in boxes:
        x0, y0, x1, y1 = (float(v) for v in b)
        out.append([x0, y0, x1 - x0, y1 - y0])
    return out


def _iou(dt: list, gt: list, crowd: bool) -> np.ndarray:
    if not dt or not gt:
        return np.zeros((len(dt), len(gt)), np.float64)
    return np.asarray(mask_utils.iou(dt, gt, [int(crowd)] * len(gt)), np.float64).reshape(len(dt), len(gt))


def box_ious(pred: Sequence[Sequence[float]], gt: Sequence[Sequence[float]], crowd_rles: Sequence[Mapping[str, Any]] = ()
             ) -> tuple[np.ndarray, np.ndarray]:
    """框（原图像素 xyxy）→ (IoU (n_pred, n_gt), 与 crowd 区域的 IoA (n_pred, n_crowd))。crowd 区域取 RLE 的外接框。"""
    crowd_boxes = [list(map(float, mask_utils.toBbox(coco_rle(r)))) for r in crowd_rles]
    p = _xywh(pred)
    return _iou(p, _xywh(gt), False), _iou(p, crowd_boxes, True)


def mask_ious(pred: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]], crowd_rles: Sequence[Mapping[str, Any]] = ()
              ) -> tuple[np.ndarray, np.ndarray]:
    """mask（COCO RLE）→ (IoU, 与 crowd 区域的 IoA)。"""
    p = [coco_rle(r) for r in pred]
    return _iou(p, [coco_rle(r) for r in gt], False), _iou(p, [coco_rle(r) for r in crowd_rles], True)


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """一个阈值上的匹配：gt_of[i] = 预测 i 配上的真值编号（−1 = 没配上），ignored[i] = 预测 i 被忽略（crowd 或非穷尽类别）。"""

    gt_of: tuple[int, ...]
    ignored: tuple[bool, ...]

    def counts(self, n_gt: int) -> dict[str, int]:
        tp = sum(1 for g in self.gt_of if g >= 0)
        ign = sum(1 for g, x in zip(self.gt_of, self.ignored) if g < 0 and x)
        return {"tp": tp, "fp": len(self.gt_of) - tp - ign, "fn": n_gt - tp, "ignored": ign}


def _match_coco(ious: np.ndarray, crowd: np.ndarray, t: float) -> tuple[list[int], list[bool]]:
    """COCOeval evaluateImg 的匹配（真值表 = 非 crowd 在前、crowd 在后，gtIg 分别为 0、1；areaRng = all）。"""
    n_p, n_g = ious.shape
    full = np.concatenate([ious, crowd], axis=1) if crowd.size or n_g else np.zeros((n_p, 0))
    ig = [False] * n_g + [True] * crowd.shape[1]
    gtm = [-1] * len(ig)
    gt_of, ignored = [-1] * n_p, [False] * n_p
    for d in range(n_p):
        best = min(t, 1 - 1e-10)
        m = -1
        for g in range(len(ig)):
            if gtm[g] >= 0 and not ig[g]:  # 已配的非 crowd 真值跳过（crowd 可以重复配）
                continue
            if m > -1 and not ig[m] and ig[g]:  # 已经配上非 crowd，不再看 crowd
                break
            if full[d, g] < best:
                continue
            best, m = full[d, g], g
        if m == -1:
            continue
        if ig[m]:
            ignored[d] = True
        else:
            gt_of[d] = m
        gtm[m] = d
    return gt_of, ignored


def _crowd_ignore(gt_of: list[int], crowd: np.ndarray, t: float) -> list[bool]:
    if crowd.shape[1] == 0:
        return [False] * len(gt_of)
    hit = crowd.max(axis=1) >= min(t, 1 - 1e-10)
    return [bool(g < 0 and h) for g, h in zip(gt_of, hit)]


def _match_greedy(ious: np.ndarray, crowd: np.ndarray, t: float) -> tuple[list[int], list[bool]]:
    n_p, n_g = ious.shape
    thr = min(t, 1 - 1e-10)
    cand = sorted(((-ious[i, j], i, j) for i in range(n_p) for j in range(n_g) if ious[i, j] >= thr))
    gt_of, used = [-1] * n_p, set()
    for _, i, j in cand:
        if gt_of[i] < 0 and j not in used:
            gt_of[i] = j
            used.add(j)
    return gt_of, _crowd_ignore(gt_of, crowd, t)


def _match_hungarian(ious: np.ndarray, crowd: np.ndarray, t: float) -> tuple[list[int], list[bool]]:
    n_p, n_g = ious.shape
    gt_of = [-1] * n_p
    if n_p and n_g:
        adj = ious >= min(t, 1 - 1e-10)
        k = float(min(n_p, n_g) + 1)  # 每多配一对加 k（> 全部 IoU 之和），先保证对数最多
        w = np.where(adj, k + ious, 0.0)
        rows, cols = linear_sum_assignment(-w)
        for r, c in zip(rows, cols):
            if adj[r, c]:
                gt_of[r] = int(c)
    return gt_of, _crowd_ignore(gt_of, crowd, t)


_MATCHERS: Mapping[str, Callable[[np.ndarray, np.ndarray, float], tuple[list[int], list[bool]]]] = {
    "coco": _match_coco, "greedy_iou": _match_greedy, "hungarian": _match_hungarian}
assert set(_MATCHERS) == set(MATCHINGS)


def _as2d(a: Any, rows: int, name: str) -> np.ndarray:
    x = np.asarray(a, np.float64)
    if x.size == 0:
        return np.zeros((rows, x.shape[1] if x.ndim == 2 else 0), np.float64)
    if x.ndim != 2 or x.shape[0] != rows:
        raise ValueError(f"{name} 必须是 ({rows}, k) 的二维数组，收到形状 {x.shape}")
    return x


def match(ious: np.ndarray, crowd_ioa: np.ndarray, t: float, *, method: str = MATCHING, not_exhaustive: bool = False) -> Match:
    """一个对、一个阈值的匹配（模块说明）。ious：(n_pred, n_gt)，预测按模型的输出顺序；crowd_ioa：(n_pred, n_crowd)。"""
    if method not in _MATCHERS:
        raise ValueError(f"匹配算法必须是 {MATCHINGS} 之一，收到 {method!r}")
    n_p = int(np.asarray(ious).shape[0]) if np.asarray(ious).ndim else 0
    ious, crowd_ioa = _as2d(ious, n_p, "ious"), _as2d(crowd_ioa, n_p, "crowd_ioa")
    gt_of, ignored = _MATCHERS[method](ious, crowd_ioa, float(t))
    if not_exhaustive:
        ignored = [g < 0 for g in gt_of]
    return Match(tuple(int(g) for g in gt_of), tuple(bool(x) for x in ignored))


def eval_pair(ious: np.ndarray, crowd_ioa: np.ndarray, n_gt: int, *, method: str = MATCHING, not_exhaustive: bool = False
              ) -> dict[str, Any]:
    """一个对在全部阈值上的计数：{"tp", "fp", "fn", "ignored"（各 10 个，按 IOU_THRESHOLDS）, "flags"（每档每个预测：1 = TP、0 = FP、
    −1 = 忽略；f1_rexomni_exact 用）, "matched_iou"（IoU 0.5 下配上的对的 IoU，按预测的顺序）, "gt_of"（IoU 0.5 下每个预测配上的真值编号）}。ious：(n_pred, n_gt)；预测多于 MAX_DETS 个时只取前 MAX_DETS 个
    （输出顺序）。"""
    x = np.asarray(ious, np.float64)
    n_p = int(x.shape[0]) if x.ndim == 2 else 0
    ious = _as2d(x.reshape(n_p, int(n_gt)) if x.size else np.zeros((n_p, int(n_gt))), n_p, "ious")
    crowd_ioa = _as2d(crowd_ioa, n_p, "crowd_ioa")
    ious, crowd_ioa = ious[:MAX_DETS], crowd_ioa[:MAX_DETS]
    out: dict[str, Any] = {k: [] for k in COUNT_KEYS}
    out["flags"] = []
    for k, t in enumerate(IOU_THRESHOLDS):
        mt = match(ious, crowd_ioa, t, method=method, not_exhaustive=not_exhaustive)
        c = mt.counts(int(n_gt))
        for key in COUNT_KEYS:
            out[key].append(c[key])
        out["flags"].append([1 if g >= 0 else (-1 if x else 0) for g, x in zip(mt.gt_of, mt.ignored)])
        if k == HEADLINE_INDEX:
            out["gt_of"] = list(mt.gt_of)
            out["matched_iou"] = [float(ious[i, g]) for i, g in enumerate(mt.gt_of) if g >= 0]
    return out


# ---------------------------------------------------------------------------
# 数据集读数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """一个设置、一个划分的计数，按对存放（稀疏）：pair_ids（各对的 id，输入顺序；配对时核对）、images（图像键，首次出现的顺序）、
    categories（类别 id，升序）、pair_img / pair_cat（每个对的图像、类别下标）、pair_counts (n_pairs, T, 3) 的 (tp, fp, fn)、
    iou (n_img, 2) 的 (IoU 0.5 下配上的 IoU 之和, 配上的个数)。"""

    pair_ids: tuple[str, ...]
    images: tuple[str, ...]
    categories: tuple[Any, ...]
    pair_img: np.ndarray
    pair_cat: np.ndarray
    pair_counts: np.ndarray
    iou: np.ndarray

    def totals(self, weights: np.ndarray | None = None) -> np.ndarray:
        """(C, T, 3) 的各类合计；weights：(n_img,) 的每张图的重数（bootstrap；None = 每张一次）。计数是整数，结果精确。"""
        out = np.zeros((len(self.categories), len(IOU_THRESHOLDS), 3), np.float64)
        w = np.ones(len(self.pair_ids), np.float64) if weights is None else np.asarray(weights, np.float64)[self.pair_img]
        np.add.at(out, self.pair_cat, self.pair_counts * w[:, None, None])
        return out

    def dense(self) -> np.ndarray:
        """(n_img, C, T, 3) 的按图计数（src.stats 的参考实现按第 0 维重抽；只用于核对与小数据）。"""
        out = np.zeros((len(self.images), len(self.categories), len(IOU_THRESHOLDS), 3), np.float64)
        np.add.at(out, (self.pair_img, self.pair_cat), self.pair_counts.astype(np.float64))
        return out


def table(rows: Sequence[Mapping[str, Any]], kind: str = "det") -> Table:
    """rows：每个对一行，{"id", "image_key", "category_id", kind: eval_pair 的结果}（predictions.jsonl 的行就是这个形状）。"""
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一，收到 {kind!r}")
    images: dict[str, int] = {}
    cats = sorted({r["category_id"] for r in rows}, key=lambda c: (str(type(c)), c))
    ci = {c: k for k, c in enumerate(cats)}
    for r in rows:
        images.setdefault(str(r["image_key"]), len(images))
    n = len(rows)
    pc = np.zeros((n, len(IOU_THRESHOLDS), 3), np.int64)
    iou = np.zeros((len(images), 2), np.float64)
    for k, r in enumerate(rows):
        e = r[kind]
        pc[k, :, 0], pc[k, :, 1], pc[k, :, 2] = e["tp"], e["fp"], e["fn"]
        i = images[str(r["image_key"])]
        iou[i, 0] += float(sum(e["matched_iou"]))
        iou[i, 1] += len(e["matched_iou"])
    pimg = np.array([images[str(r["image_key"])] for r in rows], np.int64)
    pcat = np.array([ci[r["category_id"]] for r in rows], np.int64)
    return Table(tuple(str(r["id"]) for r in rows), tuple(images), tuple(cats), pimg, pcat, pc, iou)


def _macro(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[float | None, float | None]:
    """一个阈值上的宏平均 (P, R)：tp、fp、fn 是 (C,) 的各类合计。分母为 0 的类不进相应的平均（无定义）。"""
    tp, fp, fn = (np.asarray(v, np.float64) for v in (tp, fp, fn))
    dp, dr = tp + fp, tp + fn
    pc, rc = tp[dp > 0] / dp[dp > 0], tp[dr > 0] / dr[dr > 0]
    return (float(np.mean(pc)) if pc.size else None), (float(np.mean(rc)) if rc.size else None)


def _f1(p: float | None, r: float | None) -> float:
    if r is None:
        return float("nan")
    if p is None or p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def prf_from_totals(totals: np.ndarray, statistic: str) -> tuple[float | None, float | None, float]:
    """(C, T, 3) 的合计 → (P, R, F1)。statistic 见 STATISTICS（matched_mean_iou 不在这里）。R 无定义（没有真值）时 F1 为 nan。"""
    tp, fp, fn = totals[..., 0], totals[..., 1], totals[..., 2]
    if statistic == "f1_rexomni":
        p, r = _macro(tp[:, HEADLINE_INDEX], fp[:, HEADLINE_INDEX], fn[:, HEADLINE_INDEX])
        return p, r, _f1(p, r)
    if statistic == "f1_rexomni_miou":
        pr = [_macro(tp[:, k], fp[:, k], fn[:, k]) for k in range(len(IOU_THRESHOLDS))]
        ps, rs = [p for p, _ in pr if p is not None], [r for _, r in pr if r is not None]
        p = float(np.mean(ps)) if ps else None
        r = float(np.mean(rs)) if rs else None
        return p, r, _f1(p, r)
    if statistic == "f1_micro":
        a, b, c = int(tp[:, HEADLINE_INDEX].sum()), int(fp[:, HEADLINE_INDEX].sum()), int(fn[:, HEADLINE_INDEX].sum())
        p = a / (a + b) if a + b else None
        r = a / (a + c) if a + c else None
        return p, r, (2 * a / (2 * a + b + c) if 2 * a + b + c else float("nan"))
    raise ValueError(f"未知的统计量 {statistic!r}")


def _safe_mean(values: Sequence[float]) -> float:
    """Rex-Omni 的 safe_mean：只平均 > 0 的值，没有就记 0。"""
    v = [x for x in values if x > 0]
    return float(np.mean(v)) if v else 0.0


def _envelope_precision(flags: Sequence[int], n_gt: int) -> tuple[float | None, float]:
    """一个类别、一个阈值：按顺序的预测（1 = TP、0 = FP、−1 = 忽略）→ (插值精度在最高的、> 0 的召回档的值（没有为 None）, 召回)。
    同 cocoeval.cpp 的 accumulate：pr = tp / (tp + fp + eps) 取单调包络，q[r] = pr[第一个 rc ≥ recThrs[r] 的位置]。真值为 0 时召回记 −1。"""
    f = np.asarray([x for x in flags if x >= 0], np.float64)
    if n_gt == 0:
        return None, -1.0
    if f.size == 0:
        return None, 0.0
    tp, fp = np.cumsum(f), np.cumsum(1 - f)
    rc = tp / n_gt
    pr = tp / (fp + tp + np.spacing(1))
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    inds = np.searchsorted(rc, REC_THRS, side="left")
    q = np.array([pr[i] if i < len(pr) else 0.0 for i in inds])
    pos = np.flatnonzero(q > 0)
    return (float(q[pos[-1]]) if pos.size else None), float(rc[-1])


def rexomni_exact(rows: Sequence[Mapping[str, Any]], kind: str = "det") -> dict[str, dict[str, float]]:
    """f1_rexomni_exact 与 f1_rexomni_exact_miou（模块说明；诊断）。rows 同 table 的输入（要 kind 一节的 flags 与 fn）。"""
    cats = sorted({r["category_id"] for r in rows}, key=lambda c: (str(type(c)), c))
    n_t = len(IOU_THRESHOLDS)
    p50, r50, pm, rm = [], [], [], []
    last50 = 0.0  # 第一个类别就没有 > 0 的精度时：C++ 里是未初始化的值，这里记 0（safe_mean 丢掉）
    tmp = 0.0
    for c in cats:
        rs = [r for r in rows if r["category_id"] == c]
        acc_p, acc_r = 0.0, 0.0
        for k in range(n_t):
            flags = [x for r in rs for x in r[kind]["flags"][k]]
            n_gt = int(sum(r[kind]["tp"][k] + r[kind]["fn"][k] for r in rs))
            pv, rv = _envelope_precision(flags, n_gt)
            if k == HEADLINE_INDEX:
                if pv is not None:
                    last50 = pv
                p50.append(last50)
                r50.append(rv)
            if pv is not None:
                tmp = pv
            acc_p += tmp
            acc_r += rv
        pm.append(acc_p / n_t)
        rm.append(acc_r / n_t)
    out = {}
    for name, ps, rs_ in (("f1_rexomni_exact", p50, r50), ("f1_rexomni_exact_miou", pm, rm)):
        p, r = _safe_mean(ps), _safe_mean(rs_)
        out[name] = {"value": (2 * p * r / (p + r)) if p + r > 0 else 0.0, "precision": p, "recall": r}
    return out


def statistic_fn(statistic: str) -> Callable[[np.ndarray], float]:
    """src.stats 的约定下的统计量（输入是按图的行，第 0 维是图，bootstrap 沿它重抽）：F1 类的输入是 Table.dense() 的 (n, C, T, 3)，
    先按图合计再算；matched_mean_iou 的输入是 Table.iou 的 (n, 2)，ratio_of_sums。"""
    if statistic == "matched_mean_iou":
        return S.ratio_of_sums
    if statistic not in STATISTICS:
        raise ValueError(f"未知的统计量 {statistic!r}")
    return lambda x: prf_from_totals(np.asarray(x, np.float64).sum(axis=0), statistic)[2]


def point(tb: Table, statistic: str) -> float | None:
    if statistic == "matched_mean_iou":
        v = S.ratio_of_sums(tb.iou)
    else:
        v = prf_from_totals(tb.totals(), statistic)[2]
    return None if not np.isfinite(v) else float(v)


def _boot_values(tb: Table, statistic: str, idx: np.ndarray) -> np.ndarray:
    """每次重抽（idx 的一行 = 被抽到的图的下标）的统计量。F1 类：按图的重数加权合计（整数，精确），再调同一个标量函数；
    matched_mean_iou：取被抽到的行，ratio_of_sums。两者都与 src.stats 的逐次循环逐位相同。"""
    n = len(tb.images)
    out = np.empty(idx.shape[0], np.float64)
    for k, row in enumerate(idx):
        if statistic == "matched_mean_iou":
            out[k] = S.ratio_of_sums(tb.iou[row])
        else:
            out[k] = prf_from_totals(tb.totals(np.bincount(row, minlength=n)), statistic)[2]
    return out


def _ci(est: float, boot: np.ndarray, level: float, n_resamples: int) -> S.CI:
    S._check_finite("原样本上的统计量", np.asarray(est))
    S._check_boot(boot)
    return S._percentile_ci(est, boot, level, n_resamples)


def bootstrap_readout(tb: Table, statistic: str, *, n_resamples: int = BOOTSTRAP_N_RESAMPLES, level: float = BOOTSTRAP_CI_LEVEL,
                      seed: int = BOOTSTRAP_SEED) -> S.CI | None:
    """按图像重抽的 bootstrap CI（模块说明）。没有图像时返回 None。等于 src.stats.bootstrap_ci(数据, statistic=statistic_fn(…))。"""
    if not tb.images:
        return None
    idx = S._resample_indices(len(tb.images), n_resamples, seed)
    est = point(tb, statistic)
    return _ci(float("nan") if est is None else est, _boot_values(tb, statistic, idx), level, n_resamples)


def align(a: Table, b: Table) -> None:
    """配对的前提：同一批对（id 与顺序相同）、同一批图像（顺序相同）、同一份类别表。"""
    if a.pair_ids != b.pair_ids or a.images != b.images or a.categories != b.categories:
        raise ValueError("两个系统的对、图像或类别不一致，不能配对（必须是同一个划分的同一批对，顺序相同）")


def paired_bootstrap(a: Table, b: Table, statistic: str, *, n_resamples: int = DECISION_N_RESAMPLES, level: float = BOOTSTRAP_CI_LEVEL,
                     seed: int = BOOTSTRAP_SEED) -> S.CI | None:
    """配对 bootstrap（A − B）：同一组按图像重抽的下标同时作用于两个系统。缺省 10,000 次（判定）。
    等于 src.stats.paired_bootstrap_ci(数据A, 数据B, statistic=statistic_fn(…))。"""
    align(a, b)
    if not a.images:
        return None
    idx = S._resample_indices(len(a.images), n_resamples, seed)
    pa, pb = point(a, statistic), point(b, statistic)
    est = float("nan") if pa is None or pb is None else pa - pb
    return _ci(est, _boot_values(a, statistic, idx) - _boot_values(b, statistic, idx), level, n_resamples)


def _ci_dict(ci: S.CI | None) -> dict[str, Any] | None:
    return None if ci is None else {"low": ci.low, "high": ci.high, "level": ci.level, "n_resamples": ci.n_resamples}


def readout(rows: Sequence[Mapping[str, Any]], kind: str = "det", *, n_resamples: int | None = BOOTSTRAP_N_RESAMPLES) -> dict[str, Any]:
    """一个设置、一个划分的读数（metrics.json 的 det / seg 一节）。n_resamples = None 时不算 CI。"""
    tb = table(rows, kind)
    out: dict[str, Any] = {"kind": kind, "iou": "box" if kind == "det" else "mask", "n_pairs": len(rows), "n_images": len(tb.images),
                           "n_categories": len(tb.categories), "primary": PRIMARY}
    for st in STATISTICS:
        v = point(tb, st)
        e: dict[str, Any] = {"value": v}
        if st != "matched_mean_iou":
            p, r, _ = prf_from_totals(tb.totals(), st)
            e.update(precision=p, recall=r)
        if n_resamples and v is not None:
            try:
                e["ci"] = _ci_dict(bootstrap_readout(tb, st, n_resamples=n_resamples))
            except ValueError as err:  # 某些重抽上无定义（如 matched_mean_iou 抽到没有 TP 的图）：运行时的读数照写，CI 记 None
                e["ci"], e["ci_note"] = None, str(err)
        out[st] = e
    for st, e in rexomni_exact(rows, kind).items():
        out[st] = {**e, "note": "诊断（不进判定，没有 CI）：照 Rex-Omni 的 cocoeval.cpp 复现——插值精度、precision 不清零、safe_mean"}
    out["f1_rexomni"]["note"] = "主读数（XM-3a）：Rex-Omni 的类别宏平均 F1，IoU 0.5，COCOeval 式匹配（输出顺序），P_c = TP_c / (TP_c + FP_c)"
    out["matched_mean_iou"]["note"] = "本项目自己的读数（不是 Rex-Omni 的）：IoU 0.5 下配上的对的平均 IoU"
    tot = tb.totals().sum(axis=0)
    out["totals_iou50"] = {"tp": int(round(tot[HEADLINE_INDEX, 0])), "fp": int(round(tot[HEADLINE_INDEX, 1])),
                           "fn": int(round(tot[HEADLINE_INDEX, 2])),
                           "ignored": int(sum(r[kind]["ignored"][HEADLINE_INDEX] for r in rows))}
    return out


# ---------------------------------------------------------------------------
# 指代分割（ReasonSeg；X-2 的运行器写的 W2 格式）
# ---------------------------------------------------------------------------


def refseg_readout(recs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """W2 格式 predictions.jsonl 的行（score.inter / union；失败的行 failure 非空，按空 mask 计）→ {"n", "gIoU", "cIoU"}（同 W2 口径）。"""
    st = [(int(r["score"]["inter"]), int(r["score"]["union"]), r.get("answer") is None) for r in recs]
    return {"n": len(st), "gIoU": giou(st), "cIoU": ciou([(i, u) for i, u, _ in st])}


__all__ = [
    "COUNT_KEYS", "DECISION_N_RESAMPLES", "HEADLINE_IOU", "IOU_THRESHOLDS", "KINDS", "MATCHING", "MATCHINGS", "MAX_DETS", "METRICS_VERSION",
    "EXACT_STATISTICS", "Match", "PRIMARY", "REC_THRS", "STATISTICS", "rexomni_exact", "Table", "align", "bootstrap_readout", "box_ious", "coco_rle", "coco_rle_json", "eval_pair", "identity", "mask_ious",
    "match", "paired_bootstrap", "point", "prf_from_totals", "readout", "refseg_readout", "statistic_fn", "table",
]
