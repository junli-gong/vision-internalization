"""X-1 上限测量的格点、逐句计算与汇总（DEV-0068；scripts/explore_ceiling.py 调用）。

逐句计算（evaluate_job）在进程池的子进程里跑，所以本模块与它 import 的模块都不含 torch（tests/test_explore_guidance.py 钉住）。
一个 job = 一句话 × 一种框（gt / base / w2c）。对这个框按格点（configs/explore_ceiling.json 的 grids.<名>）算出每个格点的最终 mask，
只回传与真值的交、并（不回传 mask）与这个通道的输出文本（主进程用 tokenizer 数 token）。

格点名（cell_names 给出固定顺序）：
- box：框本身（box_to_mask）；
- gen.<名字>：生成器库 v2 的单个候选；genK.<K>：前 K 个名字（generators.order）里真值 IoU 最高的（并列取编号小的，整数精确比较，
  同 src/w2/loop.oracle_index）；1 号候选 gc_default 失败时按空 mask 用（同 W2 的裁定 A），其余失败的用框；
- ref.<起点>：除 gc_default 以外的起点 mask 本身（devcheck2 上 w2c 框的 W2-c 最终 mask）；
- pts.k<k>.<fg|fgbg>.<pr_bgd|pr_fgd>.<噪声档>：k 个前景点（fgbg：另加 k 个背景点）→ guidance.points_grabcut，框内初始化为可能背景
  （pr_bgd，任务书原文）或可能前景（pr_fgd，W1-F 式）；同一组点给两种初始化；
- spl.n<n>.<raster|gc_soft|gc_trimap>.<噪声档>：n 个控制点的 B 样条 → 直接光栅化（框约束）或作 GrabCut 的种子（两种模式）；
- cor.<起点>.m<m>.<lock|nolock>.<grabcut|direct>.r<轮>.<噪声档>：从起点 mask 开始，每轮至多 m 个修正框（oracle_corrections，
  约束框 = 本句的框），第 1..rounds 轮之后的 mask；每轮的提议按当前 mask 重算。
框解析失败（box = None）的句子：所有格点都按空 mask 计分（交 0、并 = |真值|，同 W2）。
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.executors.geometry import box_iou
from src.executors.raster import box_to_mask, rle_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as O
from src.metrics.seg import mask_inter_union

START_GC_DEFAULT = "gc_default"
CHANNELS = ("box", "gen", "genK", "ref", "pts", "spl", "cor")
SPLINE_USES = ("raster", "gc_soft", "gc_trimap")
# 进程池子进程的线程限制（与 src/w2/runner.THREAD_ENV_VARS 相同；测试核对。这里不 import 它：那个模块带 torch）
THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")


# ---------------------------------------------------------------------------
# 格点
# ---------------------------------------------------------------------------


def _lock_name(lock: bool) -> str:
    return "lock" if lock else "nolock"


def cell_names(grid: Mapping[str, Any], generator_order: Sequence[str], starts: Sequence[str]) -> list[str]:
    """一种框上全部格点的名字（固定顺序）。starts：这个 job 有的起点 mask（总有 gc_default）。"""
    names = ["box"]
    g = grid["generators"]
    names += [f"gen.{n}" for n in generator_order[: max(g["K"])]]
    names += [f"genK.{k}" for k in g["K"]]
    names += [f"ref.{s}" for s in starts if s != START_GC_DEFAULT]
    p = grid["points"]
    names += [f"pts.k{k}.{'fgbg' if bg else 'fg'}.{init}.{nz}" for nz in p["noise"] for k in p["k"] for bg in p["bg"]
              for init in p["box_init"]]
    s = grid["spline"]
    names += [f"spl.n{n}.{u}.{nz}" for nz in s["noise"] for n in s["n"] for u in s["use"]]
    c = grid["corrections"]
    for st in c["starts"]:
        if st not in starts:
            continue
        names += [f"cor.{st}.m{m}.{_lock_name(lk)}.{rule}.r{r}.{nz}" for nz in c["noise"] for m in c["m"] for lk in c["lock"]
                  for rule in c["rule"] for r in range(1, int(c["rounds"]) + 1)]
    return names


def check_grid(grid: Mapping[str, Any], generator_order: Sequence[str], noise_levels: Mapping[str, Any]) -> None:
    """格点的自洽：键齐全、取值合法、噪声档都有定义、K 不超过生成器个数、名字不重复。"""
    need = {"generators": {"K"}, "points": {"k", "bg", "box_init", "noise"}, "spline": {"n", "use", "noise"},
            "corrections": {"rounds", "m", "lock", "rule", "noise", "starts"}}
    for sec, keys in need.items():
        if sec not in grid or set(grid[sec]) != keys:
            raise ValueError(f"格点的 {sec} 一节必须恰有键 {sorted(keys)}，收到 {sorted(grid.get(sec, {}))}")
    if not grid["generators"]["K"] or max(grid["generators"]["K"]) > len(generator_order) or min(grid["generators"]["K"]) < 1:
        raise ValueError(f"K 必须在 1..{len(generator_order)} 内，收到 {grid['generators']['K']}")
    for sec in ("points", "spline", "corrections"):
        bad = [n for n in grid[sec]["noise"] if n not in noise_levels]
        if bad:
            raise ValueError(f"格点 {sec} 用到未定义的噪声档 {bad}")
    if not grid["points"]["box_init"] or any(b not in G.BOX_INITS for b in grid["points"]["box_init"]):
        raise ValueError(f"points.box_init 必须是 {G.BOX_INITS} 的非空子集")
    if any(u not in SPLINE_USES for u in grid["spline"]["use"]):
        raise ValueError(f"spline.use 必须是 {SPLINE_USES} 的子集")
    if any(r not in G.CORRECTION_RULES for r in grid["corrections"]["rule"]):
        raise ValueError(f"corrections.rule 必须是 {G.CORRECTION_RULES} 的子集")
    if START_GC_DEFAULT not in grid["corrections"]["starts"]:
        raise ValueError("corrections.starts 必须含 gc_default")
    if int(grid["corrections"]["rounds"]) < 1:
        raise ValueError("corrections.rounds 必须 ≥ 1")
    names = cell_names(grid, generator_order, list(grid["corrections"]["starts"]))
    if len(set(names)) != len(names):
        raise ValueError("格点名重复")


# ---------------------------------------------------------------------------
# 逐句计算（子进程）
# ---------------------------------------------------------------------------


def _iu(mask: np.ndarray | None, gt: np.ndarray) -> list[int]:
    s = mask_inter_union(mask, gt)
    return [int(s.inter), int(s.union)]


def _better(a: Sequence[int], b: Sequence[int]) -> bool:
    """a 的 IoU 严格高于 b（整数交叉相乘；并为 0 时 IoU 记 1，同 sample_iou）。"""
    ia, ua = a
    ib, ub = b
    va = (ia, ua) if ua else (1, 1)
    vb = (ib, ub) if ub else (1, 1)
    return va[0] * vb[1] > vb[0] * va[1]


def evaluate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """一句话 × 一种框 → {"key", "cells": {名: [交, 并]}, "texts": {名: [文本…]}, "cpu_s": {通道: 秒}, "info": {…}}。
    job：id、box_kind、image（原图路径）、gt_rle、gt_box、box（原图像素 xyxy 或 None）、starts（{名: RLE}，不含 gc_default）、plan。"""
    from PIL import Image

    t_job = time.process_time()
    plan = job["plan"]
    grid, tools, order = plan["grid"], plan["tools"], list(plan["generator_order"])
    noise = {k: O.NoiseLevel.from_dict(k, v) for k, v in plan["noise_levels"].items()}
    salt, sid, kind = plan["salt"], job["id"], job["box_kind"]
    gt = rle_to_mask(job["gt_rle"])
    h, w = gt.shape
    starts_avail = [START_GC_DEFAULT] + sorted(job.get("starts") or {})
    names = cell_names(grid, order, starts_avail)
    cells: dict[str, list[int]] = {}
    texts: dict[str, list[str]] = {}
    cpu: dict[str, float] = {}
    info: dict[str, Any] = {}
    box = job["box"]
    if box is None:
        fail = _iu(None, gt)
        return {"key": [sid, kind], "cells": {n: list(fail) for n in names}, "texts": {}, "cpu_s": {"total": time.process_time() - t_job},
                "info": {"box_failure": True}, "runtime": _runtime()}
    with Image.open(job["image"]) as im:
        img = np.asarray(im.convert("RGB"))
    if img.shape[:2] != (h, w):
        raise RuntimeError(f"{sid}：原图 {img.shape[:2]} 与真值 {(h, w)} 尺寸不符")
    bx = tuple(float(v) for v in box)
    box_mask = box_to_mask(bx, h, w)
    cells["box"] = _iu(box_mask, gt)

    # 生成器库 v2
    t0 = time.process_time()
    kmax = max(grid["generators"]["K"])
    gens = G.generate_v2(img, bx, order[:kmax])
    gen_masks = []
    for i, r in enumerate(gens):
        m = r.mask if (r.ok or i > 0) else np.zeros((h, w), bool)  # 1 号失败按空 mask（W2 的裁定 A）
        gen_masks.append(m)
        cells[f"gen.{r.name}"] = _iu(m, gt)
    info["gen_not_ok"] = [r.name for r in gens if not r.ok]
    for k in grid["generators"]["K"]:
        best = 0
        for i in range(1, k):
            if _better(cells[f"gen.{order[i]}"], cells[f"gen.{order[best]}"]):
                best = i
        cells[f"genK.{k}"] = list(cells[f"gen.{order[best]}"])
        texts[f"genK.{k}"] = [O.pick_text(best + 1)]
    cpu["gen"] = time.process_time() - t0
    starts: dict[str, np.ndarray] = {START_GC_DEFAULT: gen_masks[0]}
    for nm, rle in sorted((job.get("starts") or {}).items()):
        if rle is None:
            raise RuntimeError(f"{sid}/{kind}：有框却没有起点 {nm} 的 mask")
        m = rle_to_mask(rle)
        if m.shape != (h, w):
            raise RuntimeError(f"{sid}：起点 {nm} 的 mask 尺寸 {m.shape} 与真值 {(h, w)} 不符")
        starts[nm] = m
        cells[f"ref.{nm}"] = _iu(m, gt)

    # 点
    t0 = time.process_time()
    pt, n_fb = tools["points"], 0
    for nz in grid["points"]["noise"]:
        for k in grid["points"]["k"]:
            for bg in grid["points"]["bg"]:
                pts_name = f"pts.k{k}.{'fgbg' if bg else 'fg'}.{nz}"  # 同一组点给不同的框内初始化（配对）
                o = O.oracle_points(gt, bx, int(k), with_bg=bool(bg), noise=noise[nz], rng=O.rng_for(salt, sid, kind, pts_name),
                                    suppress_factor=pt["suppress_factor"], min_suppress_px=pt["min_suppress_px"])
                txt = O.points_text(o["fg_q"], o["bg_q"])
                for init in grid["points"]["box_init"]:
                    name = f"pts.k{k}.{'fgbg' if bg else 'fg'}.{init}.{nz}"
                    res = G.points_grabcut(img, bx, o["fg"], o["bg"], radius_pct=pt["radius_pct"], radius_min_px=pt["radius_min_px"],
                                           iterations=pt["iterations"], box_init=init, context_pct=pt["context_pct"])
                    n_fb += not res.ok
                    cells[name] = _iu(res.mask, gt)
                    texts[name] = [txt]
    info["pts_not_ok"] = n_fb
    cpu["pts"] = time.process_time() - t0

    # B 样条
    sp = tools["spline"]
    t_r = t_g = 0.0
    n_fb = 0
    for nz in grid["spline"]["noise"]:
        for n in grid["spline"]["n"]:
            t0 = time.process_time()
            o = O.oracle_spline(gt, int(n), noise=noise[nz], rng=O.rng_for(salt, sid, kind, f"spl.n{n}.{nz}"), parts=sp["parts"],
                                holes=sp["holes"])
            txt = O.spline_text(o["loops_q"])
            t_r += time.process_time() - t0
            for use in grid["spline"]["use"]:
                t0 = time.process_time()
                name = f"spl.n{n}.{use}.{nz}"
                if use == "raster":
                    m = G.rasterize_loops(o["loops"], (h, w), box=bx)
                    t_r += time.process_time() - t0
                else:
                    res = G.spline_grabcut(img, bx, o["loops"], mode=use.split("_", 1)[1], band_pct=sp["band_pct"],
                                           band_min_px=sp["band_min_px"], iterations=sp["iterations"], context_pct=sp["context_pct"])
                    n_fb += not res.ok
                    m = res.mask
                    t_g += time.process_time() - t0
                cells[name] = _iu(m, gt)
                texts[name] = [txt]
    info["spl_gc_not_ok"] = n_fb
    cpu["spl_raster"], cpu["spl_gc"] = t_r, t_g

    # 局部修正
    co = tools["corrections"]
    c = grid["corrections"]
    confine = bx if co["confine_to_box"] else None
    agg = {"proposed": 0, "gated": 0, "applied": 0, "fallback_direct": 0}
    for rule in c["rule"]:
        cpu.setdefault(f"cor_{rule}", 0.0)
    for st in c["starts"]:
        if st not in starts:
            continue
        for nz in c["noise"]:
            for m_ in c["m"]:
                for rule in c["rule"]:
                    # 同一起点、噪声、m、规则下，用不用软锁定的区域（至少第 1 轮）相同：只算一次（结果逐位不变；用完即弃，省内存）
                    region_cache: dict | None = {} if rule == "grabcut" else None
                    for lk in c["lock"]:
                        t0 = time.process_time()
                        rng = O.rng_for(salt, sid, kind, f"cor.{st}.m{m_}.{nz}")
                        cur, lock = starts[st].copy(), None
                        acc: list[str] = []
                        for r in range(1, int(c["rounds"]) + 1):
                            o = O.oracle_corrections(cur, gt, int(m_), confine_box=confine, noise=noise[nz], rng=rng)
                            cur, lock, inf = G.correction_round(
                                img, cur, lock, o["proposals"], rule=rule, soft_lock=bool(lk), alpha=co["alpha"], gate=co["gate"],
                                confine_box=confine, context_pct=co["context_pct"], iterations=co["iterations"],
                                gate_min_px=co["gate_min_px"], gate_min_frac=co["gate_min_frac"], region_cache=region_cache)
                            agg["proposed"] += inf["n_proposed"]
                            agg["gated"] += inf["n_gated"]
                            agg["applied"] += inf["n_applied"]
                            agg["fallback_direct"] += inf["n_fallback_direct"]
                            acc.append(O.corrections_text(o["proposals_q"]))
                            name = f"cor.{st}.m{m_}.{_lock_name(lk)}.{rule}.r{r}.{nz}"
                            cells[name] = _iu(cur, gt)
                            texts[name] = list(acc)
                        cpu[f"cor_{rule}"] += time.process_time() - t0
    info["cor"] = agg
    missing = [n for n in names if n not in cells]
    if missing or len(cells) != len(names):
        raise RuntimeError(f"{sid}/{kind}：格点不全（缺 {missing[:5]}），或多出 {sorted(set(cells) - set(names))[:5]}")
    cpu["total"] = time.process_time() - t_job
    return {"key": [sid, kind], "cells": {n: cells[n] for n in names}, "texts": texts, "cpu_s": cpu, "info": info,
            "runtime": _runtime()}


def _runtime() -> dict[str, Any]:
    """不进 predictions 的运行信息：进程号与这个进程里已经 import 的学习型模块（进程池子进程里应为空）。"""
    return {"pid": os.getpid(), "learned_modules": G.loaded_learned_modules()}


# ---------------------------------------------------------------------------
# 进程池（同 W2 的候选生成：spawn，子进程单线程）
# ---------------------------------------------------------------------------


def _set_cv2_threads(n: int | None) -> int | None:
    import cv2

    old = int(cv2.getNumThreads())
    if n is not None:
        cv2.setNumThreads(int(n))
    return old


def init_worker() -> None:
    """子进程初始化：OpenCV 1 个线程（BLAS 的线程数由建池时设成 1 的环境变量继承）。"""
    _set_cv2_threads(1)


class JobRunner:
    """workers = 1：本进程（OpenCV 临时 1 个线程）；workers > 1：spawn 的进程池，建池期间 BLAS 环境变量设为 1，结束后还原。
    map 按输入顺序返回，所以结果与进程数无关。"""

    def __init__(self, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"workers 必须是正整数，收到 {workers!r}")
        self.workers = workers
        self.pool = None
        self._env: dict[str, str | None] = {}
        self._cv2_old: int | None = None

    def __enter__(self) -> "JobRunner":
        if self.workers == 1:
            self._cv2_old = _set_cv2_threads(1)
            return self
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self._env = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
        os.environ.update({k: "1" for k in THREAD_ENV_VARS})
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=init_worker)
        return self

    def map(self, jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.pool is None:
            return [evaluate_job(j) for j in jobs]
        return list(self.pool.map(evaluate_job, jobs))

    def __exit__(self, *exc: Any) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            for k, v in self._env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        elif self._cv2_old is not None:
            _set_cv2_threads(self._cv2_old)


# ---------------------------------------------------------------------------
# 汇总（只由 predictions.jsonl 算）
# ---------------------------------------------------------------------------


def ciou_ci(pairs: np.ndarray, *, seed: int, n_resamples: int, level: float) -> dict[str, Any] | None:
    """(n, 2) 的 (交, 并) → cIoU 与 bootstrap CI（percentile）。与 src.stats.bootstrap_ci(statistic=ratio_of_sums) 同一组重采样
    下标、逐位相同（测试钉住），只是向量化。空集合或总并为 0 时 None。"""
    from src.stats import _percentile_ci, _resample_indices

    a = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    if not len(a) or a[:, 1].sum() <= 0:
        return None
    idx = _resample_indices(len(a), n_resamples, seed)
    num, den = a[idx, 0].sum(axis=1), a[idx, 1].sum(axis=1)
    if np.any(den <= 0):
        return {"ciou": float(a[:, 0].sum() / a[:, 1].sum()), "ci": None, "n": int(len(a))}
    ci = _percentile_ci(float(a[:, 0].sum() / a[:, 1].sum()), num / den, level, n_resamples)
    return {"ciou": ci.estimate, "ci": [ci.low, ci.high], "n": int(len(a))}


def _giou(pairs: np.ndarray) -> float | None:
    if not len(pairs):
        return None
    i, u = pairs[:, 0], pairs[:, 1]
    return float(np.mean(np.where(u > 0, i / np.maximum(u, 1), 1.0)))


def aggregate(lines: Sequence[Mapping[str, Any]], *, good_iou: float, seed: int, n_resamples: int, level: float) -> dict[str, Any]:
    """predictions.jsonl 的行 → 每种框、每个格点的 cIoU 与 CI（全部、框 IoU ≥ good_iou、< good_iou 各一次）、gIoU、token 均值
    （只在有框的句子上平均：框解析失败的句子没有这个通道的输出）。"""
    kinds: dict[str, list[Mapping[str, Any]]] = {}
    for ln in lines:
        kinds.setdefault(ln["box_kind"], []).append(ln)
    out: dict[str, Any] = {}
    for kind, rows in kinds.items():
        names = list(rows[0]["cells"])
        for r in rows:
            if list(r["cells"]) != names:
                raise ValueError(f"{kind}：{r['id']} 的格点与第一行不同")
        good = np.array([(r["box_iou"] or 0.0) >= good_iou for r in rows])
        arr = np.array([[r["cells"][n][:2] for n in names] for r in rows], dtype=np.float64)  # (句, 格点, 2)
        has_box = np.array([r["box"] is not None for r in rows])
        tok = np.array([[r["cells"][n][3] for n in names] for r in rows], dtype=np.float64)[has_box]  # 框解析失败的句子没有输出，不进均值
        cells = {}
        for j, n in enumerate(names):
            pr = arr[:, j, :]
            ent: dict[str, Any] = {}
            for bin_name, sel in (("all", np.ones(len(rows), bool)), ("good", good), ("bad", ~good)):
                ci = ciou_ci(pr[sel], seed=seed, n_resamples=n_resamples, level=level)
                ent[bin_name] = None if ci is None else {**ci, "giou": _giou(pr[sel])}
            ent["tokens_mean"] = float(tok[:, j].mean()) if len(tok) else None
            cells[n] = ent
        out[kind] = {"n": len(rows), "n_good_box": int(good.sum()), "n_bad_box": int((~good).sum()),
                     "n_box_failure": sum(1 for r in rows if r["box"] is None),
                     "box_acc50": float(np.mean([(r["box_iou"] or 0.0) >= 0.5 for r in rows])),
                     "cells": cells}
    return out


def box_iou_or_none(box: Sequence[float] | None, gt_box: Sequence[float]) -> float | None:
    return None if box is None else float(box_iou(box, gt_box))


__all__ = ["CHANNELS", "JobRunner", "START_GC_DEFAULT", "THREAD_ENV_VARS", "aggregate", "box_iou_or_none", "cell_names", "check_grid",
           "ciou_ci", "evaluate_job", "init_worker"]
