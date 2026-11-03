"""X-M 的确定性后处理（DEV-0080；指令第三节）：不含学习，纯函数，单进程与 spawn 进程池共用（不 import torch；tests/test_xm_post.py 在干净
的子进程里核对）。

1. 框的 NMS（nms）：没有分数，按模型的输出顺序（阅读顺序）贪心——依次看每个框，与已保留的任一框的 IoU > NMS_IOU 就去掉，否则保留。
   IoU 是原图像素上的连续坐标（同 src/executors/geometry.box_iou；两个框面积都为 0 时记 0）。NMS_IOU 缺省 0.7，管理员在预注册里定。
   调用 1 的框先做 NMS，调用 2 只拿保留下来的框（检测的读数也用它们）。
2. 每个实例独立贴边（pair_tool_job，一个对一张图只读一次；实例之间允许重叠），工具与参数全部照 X-2（src/explore/x2_tools.spline_tool_job、
   configs/x2_run.json 的 x2_spline_trimap）：
   - 样条：x2_prompts.spline_mask（直接光栅化；采样点夹在各自的框内，即"控制点限制在框内"的 X-2 做法）与 guidance.spline_grabcut
     （mode = TRIMAP_MODE = "trimap"；带宽、裁剪、迭代取 guidance 的缺省，同 X-2；GrabCut 跑不起来时用光栅化本身，ok = False）；
   - 椭圆：x2_tools.ellipse_ctrl_q（框的内切椭圆拟合成 n 个网格控制点），之后同样光栅化与 trimap；
   - 框 → GrabCut：guidance.generate_v2(…, ["gc_default"])，即 W1-F 的缺省 GrabCut（= W2 的 gc_default 候选，逐位相同；失败时 mask 用
     框本身、ok = False，同 W2）。
3. 样条失败的退回（instance_mask）：SPLINE_FAILURE_FALLBACK = "box_grabcut"（D168：同 X-2 的 x2_spline_trimap 实际的退回——1 号候选 =
   W1-F 的缺省 GrabCut；失败的实例与对照 xm_box_grabcut 相同，判定第 1 条在它们上中性）；可选 "ellipse"（框内椭圆：trimap 设置用椭圆的
   trimap，光栅化设置用椭圆的光栅化）。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from src.executors.raster import mask_to_rle
from src.explore import guidance as G
from src.explore import x2_prompts as XP
from src.explore.x2_tools import ToolPool, ellipse_ctrl_q, load_rgb

POST_VERSION = "xm-post-v1"
NMS_IOU = 0.7
TRIMAP_MODE = "trimap"  # = configs/x2_run.json 的 spline.trimap.mode（tests/test_xm_post.py 钉住）
BOX_GRABCUT = "gc_default"  # guidance.LIBRARY_V2 的名字（W1-F 的缺省 GrabCut）
FALLBACKS: tuple[str, ...] = ("ellipse", "box_grabcut")
SPLINE_FAILURE_FALLBACK = "box_grabcut"  # D168
SYSTEMS: tuple[str, ...] = ("xm", "base")  # xm：样条、椭圆、框 GrabCut 都算；base：只有框 GrabCut


def identity() -> dict[str, Any]:
    return {"version": POST_VERSION, "nms_iou": NMS_IOU, "nms_rule": "输出顺序贪心，IoU > nms_iou 去掉后者", "trimap_mode": TRIMAP_MODE,
            "band_pct": G.BAND_PCT_DEFAULT, "band_min_px": G.BAND_MIN_PX, "context_pct": G.CONTEXT_PCT_DEFAULT,
            "grabcut_iterations": G.GRABCUT_ITERATIONS, "box_grabcut": BOX_GRABCUT, "spline_failure_fallback": SPLINE_FAILURE_FALLBACK}


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def box_iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    """(len(a), len(b)) 的 IoU（原图像素 xyxy，连续坐标，不加 1；并集为 0 时记 0）。"""
    A = np.asarray(a, np.float64).reshape(-1, 4)
    B = np.asarray(b, np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def nms(boxes: Sequence[Sequence[float]], iou: float = NMS_IOU) -> list[int]:
    """保留的框的下标（升序 = 输出顺序）。规则见模块说明第 1 条。"""
    if not 0.0 <= float(iou) <= 1.0:
        raise ValueError(f"NMS 的 IoU 阈值必须在 [0, 1] 内，收到 {iou!r}")
    if not len(boxes):
        return []
    m = box_iou_matrix(boxes, boxes)
    kept: list[int] = []
    for i in range(len(boxes)):
        if all(m[i, j] <= iou for j in kept):
            kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# 逐实例的工具
# ---------------------------------------------------------------------------


def _spline_out(img: np.ndarray, box: Sequence[float], q: Sequence[Sequence[int]]) -> dict[str, Any]:
    """一条网格样条 → 光栅化与 trimap（与 x2_tools.spline_tool_job 同一套调用）。"""
    h, w = img.shape[:2]
    raster = XP.spline_mask(q, box, (h, w))
    tri = G.spline_grabcut(img, box, XP.spline_loops(q, w, h), mode=TRIMAP_MODE)
    return {"raster_rle": mask_to_rle(raster), "trimap_rle": mask_to_rle(tri.mask), "trimap_ok": bool(tri.ok)}


def pair_tool_job(job: tuple[str, str, Sequence[Sequence[float]], Sequence[Any] | None, int]) -> dict[str, Any]:
    """(图像路径, system, 框（原图像素，NMS 之后）, 每个框的网格样条（None = 失败；base 时整个为 None）, n) →
    {"instances": [{"spline": … | None, "ellipse_q", "ellipse": …, "box_gc": {"rle", "ok", "error"}}], "wall_s"}。base 只有 box_gc。"""
    path, system, boxes, splines, n = job
    if system not in SYSTEMS:
        raise ValueError(f"system 必须是 {SYSTEMS} 之一，收到 {system!r}")
    t0 = time.perf_counter()
    out: list[dict[str, Any]] = []
    if boxes:
        img = load_rgb(path)
        h, w = img.shape[:2]
        for k, box in enumerate(boxes):
            b = [float(v) for v in box]
            gc = G.generate_v2(img, b, [BOX_GRABCUT])[0]
            rec: dict[str, Any] = {"box_gc": {"rle": mask_to_rle(gc.mask), "ok": bool(gc.ok), "error": gc.info.get("error")}}
            if system == "xm":
                q = None if splines is None else splines[k]
                rec["spline"] = None if q is None else _spline_out(img, b, q)
                eq = ellipse_ctrl_q(b, w, h, int(n))
                rec["ellipse_q"] = eq
                rec["ellipse"] = _spline_out(img, b, eq)
            out.append(rec)
    return {"instances": out, "wall_s": time.perf_counter() - t0}


MASK_KINDS: tuple[str, ...] = ("spline_trimap", "spline_raster", "ellipse_trimap", "box_grabcut")


def instance_mask(inst: dict[str, Any], kind: str, fallback: str | None = None) -> tuple[dict[str, Any], str | None]:
    """一个实例在一种设置下交的 mask 的 RLE 与退回说明（None = 没退回）。kind 见 MASK_KINDS；样条失败时按 fallback 退回
    （None = 调用时的 SPLINE_FAILURE_FALLBACK）。"""
    fallback = SPLINE_FAILURE_FALLBACK if fallback is None else fallback
    if kind == "box_grabcut":
        return inst["box_gc"]["rle"], None
    if kind == "ellipse_trimap":
        return inst["ellipse"]["trimap_rle"], None
    if kind not in ("spline_trimap", "spline_raster"):
        raise ValueError(f"kind 必须是 {MASK_KINDS} 之一，收到 {kind!r}")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback 必须是 {FALLBACKS} 之一，收到 {fallback!r}")
    key = "trimap_rle" if kind == "spline_trimap" else "raster_rle"
    if inst.get("spline") is not None:
        return inst["spline"][key], None
    if fallback == "ellipse":
        return inst["ellipse"][key], "ellipse"
    return inst["box_gc"]["rle"], "box_grabcut"


__all__ = ["BOX_GRABCUT", "FALLBACKS", "MASK_KINDS", "NMS_IOU", "POST_VERSION", "SPLINE_FAILURE_FALLBACK", "SYSTEMS", "TRIMAP_MODE",
           "ToolPool", "box_iou_matrix", "identity", "instance_mask", "nms", "pair_tool_job"]
