"""X-2 的工具一步（DEV-0069）：只用 src/explore/guidance（不含学习），纯函数，单进程与 spawn 进程池共用。放在单独的轻模块里，
进程池的子进程只 import 它（不带进 torch：tests/test_x2_prompts.py 在干净的子进程里核对）。

- spline_tool_job：模型（或椭圆消融）的网格控制点 → 直接光栅化（x2_prompts.spline_mask，采样点夹在提出的框内）与 trimap
  （guidance.spline_grabcut）；
- correction_tool_job：一轮带符号的局部修正（guidance.correction_round，约束框 = 提出的框，effect gate 开）；
- ellipse_ctrl_q：椭圆消融——提出的框的内切椭圆按参数角 ELLIPSE_SAMPLES 点采样，guidance.fit_closed_bspline 拟合 n 个控制点，
  夹到图内后取整到 0–1000 网格（同模型写的样条）；
- random_items：随机框消融——每个框保持网格宽高与符号，左上角在提出的框里均匀抽（框比提出的框大时贴左上、截到提出的框）；
- ToolPool：workers = 1 时在本进程（OpenCV 临时 1 个线程），> 1 时 spawn 的进程池（子进程 OpenCV 与 BLAS 都是 1 个线程，同 W2 的候选
  生成）；map 按输入顺序返回，结果与单进程逐位相同。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Sequence

import numpy as np

from src.executors.raster import mask_to_rle
from src.explore import guidance as G
from src.explore import oracle_sim as OS
from src.explore import x2_prompts as XP

ELLIPSE_SAMPLES = 256
THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                    "NUMEXPR_NUM_THREADS")


def load_rgb(path: str) -> np.ndarray:
    """原图 (H, W, 3) uint8：PIL 打开后 convert("RGB")（同 E1 的 load_images 与 W2 的候选生成）。"""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def ellipse_ctrl_q(box: Sequence[float], width: int, height: int, n: int) -> list[list[int]]:
    x0, y0, x1, y1 = (float(v) for v in box)
    t = np.arange(ELLIPSE_SAMPLES) * (2 * math.pi / ELLIPSE_SAMPLES)
    poly = np.stack([(x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(t), (y0 + y1) / 2 + (y1 - y0) / 2 * np.sin(t)], axis=1)
    ctrl = G.fit_closed_bspline(poly, n)
    ctrl = np.stack([np.clip(ctrl[:, 0], 0.0, float(width)), np.clip(ctrl[:, 1], 0.0, float(height))], axis=1)
    return OS.quantize_xy(ctrl, width, height)


def random_items(items: Sequence[Any], box_q: Sequence[int], rng: np.random.Generator) -> list[dict[str, Any]]:
    X0, Y0, X1, Y1 = (int(v) for v in box_q)
    out = []
    for it in items:
        a, b, c, d = (int(v) for v in it["box"])
        wq, hq = min(c - a, X1 - X0), min(d - b, Y1 - Y0)
        x = int(rng.integers(X0, max(X0, X1 - wq) + 1))
        y = int(rng.integers(Y0, max(Y0, Y1 - hq) + 1))
        out.append({"sign": it["sign"], "box": [x, y, x + wq, y + hq]})
    return out


def spline_tool_job(job: tuple[str, Sequence[float], Any, Any, str]) -> dict[str, Any]:
    """(图像路径, 提出的框, 样条的网格控制点或 None, 椭圆的网格控制点或 None, trimap 的模式) →
    {"spline": {raster_rle, trimap_rle, trimap_ok} | None, "ellipse": … | None, "wall_s"}。"""
    path, box, ctrl_q, ell_q, mode = job
    t0 = time.perf_counter()
    img = load_rgb(path)
    h, w = img.shape[:2]
    out: dict[str, Any] = {}
    for key, q in (("spline", ctrl_q), ("ellipse", ell_q)):
        if q is None:
            out[key] = None
            continue
        raster = XP.spline_mask(q, box, (h, w))
        tri = G.spline_grabcut(img, box, XP.spline_loops(q, w, h), mode=mode)
        out[key] = {"raster_rle": mask_to_rle(raster), "trimap_rle": mask_to_rle(tri.mask), "trimap_ok": bool(tri.ok)}
    out["wall_s"] = time.perf_counter() - t0
    return out


def correction_tool_job(job: tuple[str, np.ndarray, Any, Sequence[Any], Sequence[float], str, bool]) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """(图像路径, 当前 mask, 承诺场或 None, proposals（原图像素）, 约束框, 规则, 软锁定) → (新 mask, 新承诺场, info, 耗时)。"""
    path, mask, lock, props, box, rule, soft = job
    t0 = time.perf_counter()
    img = load_rgb(path)
    new, lk, info = G.correction_round(img, mask, lock, props, rule=rule, soft_lock=soft, gate=True, confine_box=box)
    return new, lk, info, time.perf_counter() - t0


def _init_worker() -> None:
    import cv2

    cv2.setNumThreads(1)


class ToolPool:
    """见模块说明。"""

    def __init__(self, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"workers 必须是正整数，收到 {workers!r}")
        self.workers, self.pool, self._env, self._cv2 = workers, None, {}, None

    def __enter__(self) -> "ToolPool":
        import cv2

        if self.workers == 1:
            self._cv2 = int(cv2.getNumThreads())
            cv2.setNumThreads(1)
            return self
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self._env = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
        os.environ.update({k: "1" for k in THREAD_ENV_VARS})
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker)
        return self

    def map(self, fn: Callable[[Any], Any], jobs: Sequence[Any], chunksize: int = 1) -> list[Any]:
        if self.pool is None:
            return [fn(j) for j in jobs]
        return list(self.pool.map(fn, jobs, chunksize=chunksize))

    def __exit__(self, *exc: Any) -> None:
        import cv2

        if self.pool is not None:
            self.pool.shutdown(wait=True)
            for k, v in self._env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        elif self._cv2 is not None:
            cv2.setNumThreads(self._cv2)


__all__ = ["ELLIPSE_SAMPLES", "ToolPool", "correction_tool_job", "ellipse_ctrl_q", "load_rgb", "random_items", "spline_tool_job"]
