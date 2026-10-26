"""引导工具库（DEV-0068；探索 X-1；预注册 X）：LLM 给出形状信息（点、B 样条、带符号的局部修正框），确定性工具把它变成 mask。

之后给真正由 LLM 驱动的方法复用，所以每个工具都是纯函数：输入原图（(H, W, 3) uint8 RGB）、原图绝对像素的框与几何，输出
(H, W) bool 的 mask。坐标约定同 src/w2/candidates.py：框 xyxy、点 (x, y)，都是原图绝对像素；框的光栅化一律是
src/executors/raster.box_to_mask（像素中心落在 (lo, hi] 内）。

**只用不含学习的工具**（D113 §二.2、预注册 X §1）：numpy、scipy、scikit-image、OpenCV 的非 DNN 部分（cv2.grabCut、cvtColor、
形态学），以及 pycocotools 的 RLE（src/executors/raster）。不 import torch、transformers 等任何学习型模块，也不用 OpenCV 的 dnn、
ximgproc、saliency（tests/test_explore_guidance.py 在干净的子进程里钉住）。为此本模块**不 import src/w1/workbench.py 与
src/w2/candidates.py**（它们经 src/adapters/chat → batching 会带进 torch），而是镜像其中几个纯函数与常量：
grabcut_mask、point_disks、color_gradient、morph_mask、components_by_area、apply_post 与 GRABCUT_* / SLIC_* / MORPH_* 常量。
测试逐个核对它们与原版逐位相同、数值相等。

工具与规则：

1. **点 → GrabCut**（points_grabcut）：框外为确定背景，框内为可能背景（box_init = "pr_bgd"，任务书原文；"pr_fgd" 是 W1-F 式的
   可选项）；每个前景点处半径 r 的圆盘为确定前景，背景点同理（背景最后写，重叠处算背景，同 workbench.grabcut_mask）；
   r = max(radius_min_px, radius_pct% × 框短边)。在框外扩 context_pct% 的裁剪区域上跑 GrabCut（None = 整张图）。
2. **B 样条**（LLMSeg 附录 B.2 的参数化）：
   - 拟合（fit_mask_bspline）：mask 每个连通域（按面积从大到小取前 parts 个）填洞后的外轮廓（skimage.measure.find_contours 在
     0.5 等值线上，8 连通），按弧长均匀重采样到 max(16n, 256) 个点，以均匀参数最小二乘拟合 n 个控制点的闭合周期均匀三次 B 样条；
     holes = True 时每个洞同法拟合，光栅化时减去。
   - 求值（eval_closed_bspline）：t ∈ [0, n) 上 m 个等距参数（不含端点）；与 LLMSeg 的 scipy BSpline 构造是同一条曲线（测试钉住）。
   - 光栅化（rasterize_loops）：m = min(2000, max(64, round(2(w + h))))，w、h 是约束框（没有框时是这条样条控制点的外接框）的宽高；
     采样点逐坐标夹到框内（再夹到图内），偶奇规则填充（像素中心、(lo, hi] 规则，与 box_to_mask 一致）；外环取并，洞减去。
   - 样条 → GrabCut（spline_grabcut）：带宽 b = max(band_min_px, band_pct% × 框短边)，带 = 到轮廓（样条区域的边界）距离 ≤ b 的
     像素。mode = "soft"（任务书原文）：样条内为可能前景，样条外为可能背景，带为未知；mode = "trimap"：样条内、带外为确定前景，
     样条外、带外为确定背景，带为未知。两种模式里"未知"都按 GrabCut 的约定取值：未知区 T_U 初始化为前景（Rother 等 2004；
     OpenCV 的 GC_INIT_WITH_RECT 同样把未知区记 GC_PR_FGD），所以带记可能前景。框外一律确定背景。
3. **带符号的局部修正**（correction_round；LLMSeg §3.3、公式 4、附录 B.2）：
   - 修正框先夹到约束框（实例框）里，空了就丢掉。
   - effect gate：框里"还能改动"的像素数 ≥ max(EFFECT_MIN_PX, ceil(EFFECT_MIN_FRAC × 框面积)) 才保留；"还能改动"= "−" 时
     框内的前景且承诺 ≥ 0，"+" 时框内的背景且承诺 ≤ 0（LLMSeg achievable_px；不用软锁定时承诺场恒为 0）。
   - 补、删哪些像素由工具决定，不看真值（rule）：
     - "direct"：整个修正框；
     - "grabcut"（局部 GrabCut）：在修正框外扩 context_pct% 的裁剪区域上，修正框外的像素按当前 mask 记确定前景 / 确定背景；
       框内 "+" 时当前前景记确定前景、其余记可能前景，"−" 时当前背景记确定背景、其余记可能背景；GrabCut 之后，"+" 的区域 =
       框内的前景，"−" 的区域 = 框内的背景。GrabCut 跑不起来（某一类没有像素、OpenCV 报错）时退回 "direct"，info 记次数。
   - 同一轮里所有区域都按本轮开始时的 mask 求出，然后先删后补（LLMSeg：negatives before positives）。
   - 软锁定（apply_signed，公式 4）：带符号的承诺场 Λ（初值 0）。区域里的像素，Λ = 0 时直接写成目标并令 Λ = s；与 Λ 同号时写成
     目标并 Λ += s；反号时 |Λ| −= α，减到 ≤ 0 才翻转（写成目标、Λ = s）。α 的缺省值 LOCK_ALPHA = 1（论文公式 4；LLMSeg 代码
     geometry.REFINE_LOCK_ALPHA 是 0.5，见完成总结）。不用软锁定时区域直接写成目标。
4. **生成器库 v2**（LIBRARY_V2，按名字组成候选池；generate_v2）：GrabCut 变体（迭代、框的内缩、初始化、颜色空间、裁剪）、
   GMM 颜色模型（分量数、颜色空间；见 GENERATOR_TYPES_V2["gmm"] 的说明：OpenCV 的 GrabCut 固定 5 个分量，不能改）、超像素
   （SLIC、Felzenszwalb）合并后按面积 / 与框外的颜色连通 / 颜色最近邻取前景、分水岭变体、形态学后处理组合。前四个名字就是
   W2 冻结的 K = 4 候选（configs/w2.json），逐位等于 src/w2/candidates.generate_candidates（测试钉住）。失败（生成器跑不起来）
   时 mask 用框本身、ok = False（同 W2）。

确定性：同样的输入逐位得到同样的结果（GrabCut 每次调用前把 OpenCV 的随机数种子设成 GRABCUT_RNG_SEED；GMM 用确定的初始化）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.filters import sobel
from skimage.measure import find_contours
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import watershed as sk_watershed

from src.executors.raster import box_to_mask

# ---------------------------------------------------------------------------
# 镜像的常量（与 src/w1/workbench.py、src/w2/candidates.py 相等；tests/test_explore_guidance.py 核对）
# ---------------------------------------------------------------------------

GRABCUT_ITERATIONS = 5
GRABCUT_MAX_ITERATIONS = 20
GRABCUT_RNG_SEED = 0
GRABCUT_MASK_MARGIN = 0.2
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 1.0
MORPH_RADIUS_DEFAULT = 2
MORPH_RADIUS_MAX = 50
SEED_RADIUS_MIN_PX = 2.0
SEED_RADIUS_PCT_DEFAULT = 4.0
POST_OPS = ("largest_cc", "fill_holes", "open", "close", "dilate", "erode")

# LLMSeg（arXiv 2610.29409 附录 B.2；代码 geometry.py @ f2191e66）
SPLINE_SAMPLES_MIN = 64
SPLINE_SAMPLES_MAX = 2000
SPLINE_SAMPLES_PER_SIDE = 2.0  # m = round(2 (w + h))
LOCK_ALPHA = 1.0  # 公式 4 的 α（论文）；代码里是 0.5
EFFECT_MIN_PX = 16
EFFECT_MIN_FRAC = 0.02

# 本模块的缺省参数
FIT_RESAMPLE_PER_CTRL = 16
FIT_RESAMPLE_MIN = 256
CONTEXT_PCT_DEFAULT = 50.0
CONTEXT_MIN_PX = 8
BAND_PCT_DEFAULT = 4.0
BAND_MIN_PX = 2.0
COLOR_SPACES = ("rgb", "lab", "hsv")
SIGN_ADD = "+"
SIGN_REMOVE = "-"
SIGNS = (SIGN_ADD, SIGN_REMOVE)
CORRECTION_RULES = ("grabcut", "direct")
SPLINE_GC_MODES = ("soft", "trimap")
BOX_INITS = ("pr_bgd", "pr_fgd")

GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = int(cv2.GC_BGD), int(cv2.GC_FGD), int(cv2.GC_PR_BGD), int(cv2.GC_PR_FGD)

# 学习型模块（或带学习型模块的框架）的顶层包名：本模块与 src/explore/oracle_sim.py、ceiling.py 都不得把它们带进进程
# （tests/test_explore_guidance.py 在干净的子进程里核对；scripts/explore_ceiling.py 在进程池子进程里再审计一次，记进 compute.json）。
BANNED_MODULE_ROOTS: tuple[str, ...] = (
    "torch", "torchvision", "transformers", "timm", "tensorflow", "keras", "jax", "flax", "onnxruntime", "openvino", "paddle",
    "mxnet", "sklearn", "segment_anything", "sam2", "open_clip", "clip", "ultralytics", "detectron2", "mmdet", "mmseg", "mmcv",
    "kornia", "lightgbm", "xgboost", "rembg", "mediapipe",
)
BANNED_SUBMODULES: tuple[str, ...] = ("skimage.future",)  # 可训练的分割（随机森林）


def loaded_learned_modules() -> list[str]:
    """当前进程里已经 import 的学习型模块（BANNED_MODULE_ROOTS 的顶层包与 BANNED_SUBMODULES），排好序。"""
    import sys

    roots = {m.split(".")[0] for m in sys.modules}
    out = sorted(r for r in roots if r in BANNED_MODULE_ROOTS)
    out += sorted(s for s in BANNED_SUBMODULES if s in sys.modules)
    return out


@dataclass(frozen=True, eq=False)
class ToolResult:
    """一个工具的输出：mask（(H, W) bool）、是否正常产出、说明（参数、失败原因、计数）。"""

    mask: np.ndarray
    ok: bool
    info: dict
    name: str = ""


# ---------------------------------------------------------------------------
# 基础（镜像 src/w1/workbench.py 的纯函数）
# ---------------------------------------------------------------------------


def _check_image(image_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(image_rgb)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image_rgb 必须是 (H, W, 3) uint8，收到 dtype {img.dtype}、形状 {img.shape}")
    return img


def _check_mask(m: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    a = np.asarray(m)
    if a.dtype != bool or a.shape != tuple(shape):
        raise ValueError(f"{name} 必须是形状 {tuple(shape)} 的 bool 数组，收到 dtype {a.dtype}、形状 {a.shape}")
    return a


def check_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError(f"框坐标非有限值: {box}")
    if x1 < x0 or y1 < y0:
        raise ValueError(f"框坐标 x1<x0 或 y1<y0: {box}")
    return x0, y0, x1, y1


def box_short_side(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = check_box(box)
    return min(x1 - x0, y1 - y0)


def adjust_box(box: Sequence[float], pct: float) -> tuple[float, float, float, float]:
    """框每边外扩 pct%（左右按框宽、上下按框高；负数内缩）。同 src/w2/candidates._Ctx.adjusted。"""
    x0, y0, x1, y1 = check_box(box)
    dx, dy = pct / 100.0 * (x1 - x0), pct / 100.0 * (y1 - y0)
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def disk_footprint(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return ((xx * xx + yy * yy) <= r * r).astype(np.uint8)


def color_gradient(image_rgb: np.ndarray) -> np.ndarray:
    """彩色梯度幅值：逐通道 Sobel（图像先归一化到 [0, 1]），三通道平方和开方（同 workbench.color_gradient）。"""
    img = np.asarray(image_rgb).astype(np.float64) / 255.0
    return np.sqrt(sum(sobel(img[..., c]) ** 2 for c in range(3)))


def lab_gradient(image_rgb: np.ndarray) -> np.ndarray:
    """CIELAB 上的梯度幅值：L/100、(a+128)/255、(b+128)/255 三个通道各自 Sobel，平方和开方。"""
    lab = rgb2lab(np.asarray(image_rgb))
    chans = (lab[..., 0] / 100.0, (lab[..., 1] + 128.0) / 255.0, (lab[..., 2] + 128.0) / 255.0)
    return np.sqrt(sum(sobel(c) ** 2 for c in chans))


def point_disks(shape: tuple[int, int], pts_abs: Any, radius: float) -> np.ndarray:
    """以各点为圆心、半径 radius（px）的圆盘之并，外加点所在的像素（同 workbench.point_disks）。"""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), bool)
    r = float(radius)
    for x, y in np.asarray(pts_abs, dtype=np.float64).reshape(-1, 2):
        c, rr = min(int(math.floor(x)), w - 1), min(int(math.floor(y)), h - 1)
        out[rr, c] = True
        c0, c1 = max(int(math.floor(x - r)) - 1, 0), min(int(math.ceil(x + r)) + 1, w)
        r0, r1 = max(int(math.floor(y - r)) - 1, 0), min(int(math.ceil(y + r)) + 1, h)
        cy = np.arange(r0, r1)[:, None] + 0.5
        cx = np.arange(c0, c1)[None, :] + 0.5
        out[r0:r1, c0:c1] |= (cx - x) ** 2 + (cy - y) ** 2 <= r * r
    return out


def morph_mask(src: np.ndarray, op: str, radius: int = MORPH_RADIUS_DEFAULT) -> np.ndarray:
    """二值形态学（同 workbench.morph_mask）：fill_holes 或 dilate / erode / open / close（半径 radius 的圆盘）。"""
    if op == "fill_holes":
        return ndimage.binary_fill_holes(src)
    k = disk_footprint(radius)
    m8 = np.asarray(src).astype(np.uint8)
    fn = {
        "dilate": lambda: cv2.dilate(m8, k),
        "erode": lambda: cv2.erode(m8, k),
        "open": lambda: cv2.morphologyEx(m8, cv2.MORPH_OPEN, k),
        "close": lambda: cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k),
    }[op]
    return fn() > 0


def components_by_area(mask: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    """8 连通域（scipy.ndimage.label，行优先扫描）；按面积从大到小排好的标号，同面积按扫描顺序（同 workbench）。"""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    area = np.bincount(lab.ravel(), minlength=n + 1)
    order = sorted(range(1, n + 1), key=lambda i: (-int(area[i]), i))
    return lab, int(n), order


def parse_post(ops: Any) -> tuple[tuple[str, int | None], ...]:
    """后处理的写法同 src/w2/candidates.parse_post：["largest_cc", "fill_holes", "open:3", …]。"""
    if not isinstance(ops, (list, tuple)):
        raise ValueError(f"post 必须是列表，收到 {ops!r}")
    out = []
    for op in ops:
        if not isinstance(op, str):
            raise ValueError(f"post 的每项必须是字符串，收到 {op!r}")
        name, _, arg = op.partition(":")
        if name in ("largest_cc", "fill_holes"):
            if arg:
                raise ValueError(f"后处理 {name} 不带参数，收到 {op!r}")
            out.append((name, None))
        elif name in POST_OPS:
            try:
                r = int(arg)
            except ValueError:
                raise ValueError(f"后处理 {name} 要写成 '{name}:半径'，收到 {op!r}") from None
            if not 1 <= r <= MORPH_RADIUS_MAX or str(r) != arg:
                raise ValueError(f"后处理 {op!r} 的半径必须是 1..{MORPH_RADIUS_MAX} 的整数")
            out.append((name, r))
        else:
            raise ValueError(f"未知的后处理 {op!r}；可选 {POST_OPS}")
    return tuple(out)


def apply_post(mask: np.ndarray, post: Sequence[str]) -> np.ndarray:
    """按顺序执行后处理（同 src/w2/candidates.apply_post）。总是返回新数组。"""
    out = np.array(mask, dtype=bool, copy=True)
    for op, rad in parse_post(list(post)):
        if op == "largest_cc":
            lab, n, order = components_by_area(out)
            if n:
                out = lab == order[0]
        else:
            out = np.asarray(morph_mask(out, op, rad if rad is not None else MORPH_RADIUS_DEFAULT), dtype=bool)
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    return apply_post(mask, ["largest_cc"])


# ---------------------------------------------------------------------------
# GrabCut
# ---------------------------------------------------------------------------


def _gc_image(img: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return np.ascontiguousarray(img[..., ::-1])  # OpenCV 的 BGR（同 workbench.grabcut_mask）
    if color_space == "lab":
        return np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2LAB))
    if color_space == "hsv":
        return np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV))
    raise ValueError(f"color_space 必须是 {COLOR_SPACES} 之一，收到 {color_space!r}")


def context_crop(box: Sequence[float], pct: float | None, h: int, w: int) -> tuple[int, int, int, int]:
    """裁剪区域 (r0, r1, c0, c1)：框每边外扩 pct%（至少 CONTEXT_MIN_PX 像素），取整到像素、夹到图内。pct = None：整张图。"""
    if pct is None:
        return 0, h, 0, w
    x0, y0, x1, y1 = check_box(box)
    dx = max(pct / 100.0 * (x1 - x0), CONTEXT_MIN_PX)
    dy = max(pct / 100.0 * (y1 - y0), CONTEXT_MIN_PX)
    c0, c1 = max(int(math.floor(x0 - dx)), 0), min(int(math.ceil(x1 + dx)), w)
    r0, r1 = max(int(math.floor(y0 - dy)), 0), min(int(math.ceil(y1 + dy)), h)
    if c1 - c0 < 2 or r1 - r0 < 2:
        raise ValueError(f"裁剪区域 {c1 - c0}×{r1 - r0} px 太小")
    return r0, r1, c0, c1


def run_grabcut(image_rgb: np.ndarray, labels: np.ndarray, *, iterations: int = GRABCUT_ITERATIONS, color_space: str = "rgb",
                crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """在 labels（(H, W) 的 GC_BGD / GC_FGD / GC_PR_BGD / GC_PR_FGD）上跑 cv2.grabCut（GC_INIT_WITH_MASK），返回 (H, W) bool 的前景。
    crop = (r0, r1, c0, c1)：只在这块上跑（裁剪区外记背景，也不进 GMM）。某一类没有像素时抛 ValueError；OpenCV 内部失败抛 cv2.error。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    lab = np.asarray(labels)
    if lab.shape != (h, w):
        raise ValueError(f"labels 形状 {lab.shape} 与原图 {(h, w)} 不符")
    if isinstance(iterations, bool) or int(iterations) != iterations or not 1 <= iterations <= GRABCUT_MAX_ITERATIONS:
        raise ValueError(f"iterations 必须是 1..{GRABCUT_MAX_ITERATIONS} 的整数，收到 {iterations}")
    r0, r1, c0, c1 = crop if crop is not None else (0, h, 0, w)
    gc = np.ascontiguousarray(lab[r0:r1, c0:c1].astype(np.uint8))
    is_fg = (gc == GC_FGD) | (gc == GC_PR_FGD)
    if not is_fg.any():
        raise ValueError("GrabCut 的初始化里没有前景像素")
    if is_fg.all():
        raise ValueError("GrabCut 的初始化里没有背景像素")
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(GRABCUT_RNG_SEED)
    cv2.grabCut(_gc_image(np.ascontiguousarray(img[r0:r1, c0:c1]), color_space), gc, None, bgd, fgd, int(iterations),
                cv2.GC_INIT_WITH_MASK)
    out = np.zeros((h, w), bool)
    out[r0:r1, c0:c1] = (gc == GC_FGD) | (gc == GC_PR_FGD)
    return out


def grabcut_mask(image_rgb: np.ndarray, *, box_abs: Sequence[float] | None = None, init_mask: np.ndarray | None = None,
                 fg: np.ndarray | None = None, bg: np.ndarray | None = None, iterations: int = GRABCUT_ITERATIONS,
                 color_space: str = "rgb", context_pct: float | None = None) -> np.ndarray:
    """镜像 src/w1/workbench.grabcut_mask（color_space = "rgb"、context_pct = None 时逐位相同，测试钉住）：
    box_abs：框内可能前景、框外确定背景；init_mask：mask 内可能前景，外接框每边外扩 GRABCUT_MASK_MARGIN 以内可能背景，更远确定背景；
    fg / bg：确定前景 / 背景的种子（背景最后写）。context_pct：只在框（或 init_mask 的外接框）外扩 context_pct% 的裁剪区域上跑。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    if (box_abs is None) == (init_mask is None):
        raise ValueError("box_abs 与 init_mask 必须恰好给一个")
    if box_abs is not None:
        inside = box_to_mask(box_abs, h, w)
        gc = np.where(inside, GC_PR_FGD, GC_BGD).astype(np.uint8)
        ref_box = check_box(box_abs)
    else:
        inside = _check_mask(init_mask, (h, w), "init_mask")
        gc = np.full((h, w), GC_BGD, np.uint8)
        ref_box = (0.0, 0.0, float(w), float(h))
        if inside.any():
            rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
            r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
            mr = int(math.ceil(GRABCUT_MASK_MARGIN * (r1 - r0)))
            mc = int(math.ceil(GRABCUT_MASK_MARGIN * (c1 - c0)))
            gc[max(r0 - mr, 0) : min(r1 + mr, h), max(c0 - mc, 0) : min(c1 + mc, w)] = GC_PR_BGD
            ref_box = (float(c0), float(r0), float(c1), float(r1))
        gc[inside] = GC_PR_FGD
    if fg is not None:
        gc[_check_mask(fg, (h, w), "fg")] = GC_FGD
    if bg is not None:
        gc[_check_mask(bg, (h, w), "bg")] = GC_BGD
    crop = None if context_pct is None else context_crop(ref_box, context_pct, h, w)
    return run_grabcut(img, gc, iterations=iterations, color_space=color_space, crop=crop)


def _fallback(mask: np.ndarray, err: Exception, **info: Any) -> ToolResult:
    return ToolResult(mask=mask, ok=False, info={**info, "error": f"{type(err).__name__}: {str(err).strip()[:300]}"})


# ---------------------------------------------------------------------------
# 1. 点 → GrabCut
# ---------------------------------------------------------------------------


def clip_points(pts: Any, h: int, w: int) -> np.ndarray:
    """(n, 2) 的 (x, y) 夹到 [0, W] × [0, H] 内；空输入给 (0, 2)。"""
    a = np.asarray(pts if pts is not None else [], dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(a)):
        raise ValueError("点含非有限值")
    return np.stack([np.clip(a[:, 0], 0.0, float(w)), np.clip(a[:, 1], 0.0, float(h))], axis=1) if len(a) else a


def points_grabcut(image_rgb: np.ndarray, box: Sequence[float], fg_points: Any, bg_points: Any = None, *,
                   radius_pct: float = SEED_RADIUS_PCT_DEFAULT, radius_min_px: float = SEED_RADIUS_MIN_PX,
                   iterations: int = GRABCUT_ITERATIONS, box_init: str = "pr_bgd", context_pct: float | None = CONTEXT_PCT_DEFAULT,
                   color_space: str = "rgb") -> ToolResult:
    """框外确定背景、框内可能背景（box_init）；前景点、背景点处半径 r 的圆盘为确定前景、确定背景；跑 GrabCut。失败时 mask 用框本身。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    bx = check_box(box)
    if box_init not in BOX_INITS:
        raise ValueError(f"box_init 必须是 {BOX_INITS} 之一，收到 {box_init!r}")
    inside = box_to_mask(bx, h, w)
    fg, bgp = clip_points(fg_points, h, w), clip_points(bg_points, h, w)
    r = max(float(radius_min_px), radius_pct / 100.0 * box_short_side(bx))
    info = {"radius_px": round(r, 4), "n_fg": int(len(fg)), "n_bg": int(len(bgp)), "box_init": box_init}
    gc = np.where(inside, GC_PR_BGD if box_init == "pr_bgd" else GC_PR_FGD, GC_BGD).astype(np.uint8)
    if len(fg):
        gc[point_disks((h, w), fg, r)] = GC_FGD
    if len(bgp):
        gc[point_disks((h, w), bgp, r)] = GC_BGD
    try:
        crop = context_crop(bx, context_pct, h, w)
        return ToolResult(run_grabcut(img, gc, iterations=iterations, color_space=color_space, crop=crop), True, info)
    except (ValueError, cv2.error) as e:
        return _fallback(inside.copy(), e, **info)


# ---------------------------------------------------------------------------
# 2. B 样条
# ---------------------------------------------------------------------------


def bspline_samples(box: Sequence[float]) -> int:
    """LLMSeg 附录 B.2：m = min(2000, max(64, round(2(w + h))))。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    m = int(round(SPLINE_SAMPLES_PER_SIDE * (abs(x1 - x0) + abs(y1 - y0))))
    return int(min(SPLINE_SAMPLES_MAX, max(SPLINE_SAMPLES_MIN, m)))


def _basis_weights(u: np.ndarray) -> np.ndarray:
    """均匀三次 B 样条一段内的四个权重（对应控制点 i−1, i, i+1, i+2），u ∈ [0, 1)。"""
    u2, u3 = u * u, u * u * u
    return np.stack([(1 - u) ** 3, 3 * u3 - 6 * u2 + 4, -3 * u3 + 3 * u2 + 3 * u + 1, u3], axis=1) / 6.0


def bspline_basis(t: np.ndarray, n: int) -> np.ndarray:
    """闭合周期均匀三次 B 样条在参数 t（∈ [0, n)）处的基函数矩阵 (len(t), n)。"""
    t = np.mod(np.asarray(t, dtype=np.float64), n)
    i = np.floor(t).astype(int)
    wts = _basis_weights(t - i)
    out = np.zeros((len(t), n))
    rows = np.arange(len(t))
    for j, off in enumerate((-1, 0, 1, 2)):
        np.add.at(out, (rows, (i + off) % n), wts[:, j])
    return out


def eval_closed_bspline(ctrl: Any, m: int) -> np.ndarray:
    """n 个控制点的闭合周期均匀三次 B 样条在 t = k·n/m（k = 0..m−1）处的 m 个点 (m, 2)。"""
    p = np.asarray(ctrl, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 4:
        raise ValueError(f"三次闭合样条至少要 4 个 (x, y) 控制点，收到形状 {p.shape}")
    n = p.shape[0]
    return bspline_basis(np.arange(int(m)) * (n / float(m)), n) @ p


def resample_closed(poly: Any, m: int) -> np.ndarray:
    """闭合折线按弧长均匀重采样到 m 个点（从第 0 个顶点起）。"""
    p = np.asarray(poly, dtype=np.float64)
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(p[:1], m, axis=0)
    s = np.arange(m) * (total / m)
    return np.stack([np.interp(s, cum, q[:, 0]), np.interp(s, cum, q[:, 1])], axis=1)


def fit_closed_bspline(contour: Any, n: int) -> np.ndarray:
    """闭合轮廓（(N, 2) 的 (x, y)，首尾不重复）→ n 个控制点（最小二乘；按弧长均匀重采样到 max(16n, 256) 个点、均匀参数）。"""
    if isinstance(n, bool) or int(n) != n or n < 4:
        raise ValueError(f"控制点数 n 必须是 ≥ 4 的整数，收到 {n!r}")
    m = max(FIT_RESAMPLE_PER_CTRL * int(n), FIT_RESAMPLE_MIN)
    pts = resample_closed(contour, m)
    basis = bspline_basis(np.arange(m) * (n / float(m)), int(n))
    ctrl, *_ = np.linalg.lstsq(basis, pts, rcond=None)
    return ctrl


def _outer_contour(region: np.ndarray) -> np.ndarray | None:
    """一个区域（不含洞）的外轮廓：0.5 等值线（8 连通），取围成面积最大的那条；像素中心坐标 (x, y) = (列 + 0.5, 行 + 0.5)。"""
    rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
    if not len(rows):
        return None
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    pad = np.pad(region[r0:r1, c0:c1].astype(np.float64), 1)
    best, best_a = None, -1.0
    for cont in find_contours(pad, 0.5, fully_connected="high"):
        if len(cont) < 3:
            continue
        y, x = cont[:, 0], cont[:, 1]
        a = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if a > best_a:
            best, best_a = cont, a
    if best is None:
        return None
    if np.allclose(best[0], best[-1]):
        best = best[:-1]
    return np.stack([best[:, 1] - 1 + c0 + 0.5, best[:, 0] - 1 + r0 + 0.5], axis=1)


def fit_mask_bspline(mask: np.ndarray, n: int, *, parts: int = 1, holes: bool = False) -> list[dict]:
    """mask → 样条环的列表 [{"role": "outer" | "hole", "ctrl": (n, 2)}]：面积最大的 parts 个 8 连通域各一条外环（填洞后的外轮廓）；
    holes = True 时再给每个域里的洞各一条洞环。空 mask 给 []。"""
    m = np.asarray(mask, dtype=bool)
    lab, k, order = components_by_area(m)
    loops: list[dict] = []
    for idx in order[: max(int(parts), 0)]:
        comp = lab == idx
        filled = ndimage.binary_fill_holes(comp)
        c = _outer_contour(filled)
        if c is None:
            continue
        loops.append({"role": "outer", "ctrl": fit_closed_bspline(c, n)})
        if holes:
            hl, nh, horder = components_by_area(filled & ~comp)
            for hi in horder:
                hc = _outer_contour(hl == hi)
                if hc is not None:
                    loops.append({"role": "hole", "ctrl": fit_closed_bspline(hc, n)})
    return loops


def fill_polygon_evenodd(poly: Any, h: int, w: int) -> np.ndarray:
    """多边形（(m, 2) 的 (x, y)，自动闭合）→ (H, W) bool：偶奇规则，像素中心 (c + 0.5, r + 0.5)，边界按 (lo, hi]
    （与 box_to_mask 一致：轴对齐的矩形多边形与 box_to_mask 逐位相同）。自交的部分按偶奇规则成洞（同 LLMSeg 的 PIL 填充）。"""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    out = np.zeros((h, w), bool)
    if len(p) < 3:
        return out
    xa, ya = p[:, 0], p[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    ylo, yhi = np.minimum(ya, yb), np.maximum(ya, yb)
    rs = np.maximum(np.floor(ylo - 0.5).astype(np.int64) + 1, 0)  # 行 r 被边跨过：ylo < r + 0.5 ≤ yhi
    re = np.minimum(np.floor(yhi - 0.5).astype(np.int64), h - 1)
    cnt = np.maximum(re - rs + 1, 0)
    cnt[ya == yb] = 0
    total = int(cnt.sum())
    if total == 0:
        return out
    e = np.repeat(np.arange(len(p)), cnt)
    starts = np.repeat(np.cumsum(cnt) - cnt, cnt)
    r = rs[e] + (np.arange(total) - starts)
    yc = r + 0.5
    xc = xa[e] + (yc - ya[e]) * (xb[e] - xa[e]) / (yb[e] - ya[e])
    col = np.clip(np.floor(xc - 0.5).astype(np.int64) + 1, 0, w)  # 第一个像素中心 > xc 的列
    tog = np.zeros((h, w + 1), np.int64)
    np.add.at(tog, (r, col), 1)
    return (np.cumsum(tog, axis=1)[:, :w] & 1).astype(bool)


def _ctrl_box(ctrl: np.ndarray) -> tuple[float, float, float, float]:
    return float(ctrl[:, 0].min()), float(ctrl[:, 1].min()), float(ctrl[:, 0].max()), float(ctrl[:, 1].max())


def spline_polygon(ctrl: Any, *, box: Sequence[float] | None = None, shape: tuple[int, int] | None = None) -> np.ndarray:
    """一条样条环的采样多边形：m 由约束框（没有时由控制点的外接框）定；给了框就逐坐标夹到框内；给了 shape 再夹到图内。"""
    c = np.asarray(ctrl, dtype=np.float64)
    ref = check_box(box) if box is not None else _ctrl_box(c)
    pts = eval_closed_bspline(c, bspline_samples(ref))
    if box is not None:
        x0, y0, x1, y1 = ref
        pts = np.stack([np.clip(pts[:, 0], x0, x1), np.clip(pts[:, 1], y0, y1)], axis=1)
    if shape is not None:
        pts = np.stack([np.clip(pts[:, 0], 0.0, float(shape[1])), np.clip(pts[:, 1], 0.0, float(shape[0]))], axis=1)
    return pts


def rasterize_loops(loops: Sequence[Mapping[str, Any]], shape: tuple[int, int], *, box: Sequence[float] | None = None) -> np.ndarray:
    """样条环 → (H, W) bool：外环取并，洞环减去；每条环按 spline_polygon 采样（box 给了就做框约束），偶奇填充。"""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), bool)
    for want in ("outer", "hole"):
        for lp in loops:
            if lp.get("role", "outer") != want:
                continue
            reg = fill_polygon_evenodd(spline_polygon(lp["ctrl"], box=box, shape=(h, w)), h, w)
            out = (out | reg) if want == "outer" else (out & ~reg)
    return out


def contour_band(region: np.ndarray, band_px: float) -> np.ndarray:
    """到 region 边界距离 ≤ band_px 的像素（两侧）：region 内到最近背景像素、region 外到最近前景像素的欧氏距离 ≤ band_px。"""
    r = np.asarray(region, dtype=bool)
    if not r.any() or r.all():
        return np.zeros_like(r)
    d_in = ndimage.distance_transform_edt(r)
    d_out = ndimage.distance_transform_edt(~r)
    return (r & (d_in <= band_px)) | (~r & (d_out <= band_px))


def spline_grabcut(image_rgb: np.ndarray, box: Sequence[float], loops: Sequence[Mapping[str, Any]], *, mode: str = "soft",
                   band_pct: float = BAND_PCT_DEFAULT, band_min_px: float = BAND_MIN_PX, iterations: int = GRABCUT_ITERATIONS,
                   context_pct: float | None = CONTEXT_PCT_DEFAULT, color_space: str = "rgb") -> ToolResult:
    """样条（框约束后光栅化）作 GrabCut 的种子（规则见模块说明第 2 条）。失败时 mask 用样条的光栅化本身。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    bx = check_box(box)
    if mode not in SPLINE_GC_MODES:
        raise ValueError(f"mode 必须是 {SPLINE_GC_MODES} 之一，收到 {mode!r}")
    s = rasterize_loops(loops, (h, w), box=bx)
    inside = box_to_mask(bx, h, w)
    b = max(float(band_min_px), band_pct / 100.0 * box_short_side(bx))
    band = contour_band(s, b) & inside
    info = {"mode": mode, "band_px": round(b, 4), "spline_area": int(s.sum())}
    gc = np.full((h, w), GC_BGD, np.uint8)
    if mode == "soft":
        gc[inside & ~s] = GC_PR_BGD
        gc[inside & s] = GC_PR_FGD
    else:
        gc[inside & ~s] = GC_BGD
        gc[inside & s] = GC_FGD
    gc[band] = GC_PR_FGD  # 未知区按 GrabCut 的约定初始化为前景
    try:
        if not s.any():
            raise ValueError("样条的光栅化为空")
        crop = context_crop(bx, context_pct, h, w)
        return ToolResult(run_grabcut(img, gc, iterations=iterations, color_space=color_space, crop=crop), True, info)
    except (ValueError, cv2.error) as e:
        return _fallback(s, e, **info)


# ---------------------------------------------------------------------------
# 3. 带符号的局部修正（LLMSeg §3.3、公式 4、附录 B.2）
# ---------------------------------------------------------------------------


def sign_value(sign: str) -> float:
    if sign == SIGN_ADD:
        return 1.0
    if sign in (SIGN_REMOVE, "−"):
        return -1.0
    raise ValueError(f"符号必须是 '+' 或 '-'，收到 {sign!r}")


def intersect_box(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float] | None:
    """两个框的交；空（宽或高 ≤ 0）时 None。"""
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def achievable_px(sign: str, box: Sequence[float], mask: np.ndarray, lock: np.ndarray) -> tuple[int, int]:
    """(这个修正框在软锁定下还能改动的像素数, 框的像素面积)：同 LLMSeg geometry.achievable_px（框用 box_to_mask 光栅化）。"""
    h, w = mask.shape
    roi = box_to_mask(box, h, w)
    area = int(roi.sum())
    if area == 0:
        return 0, 0
    target = (mask & (lock >= 0.0)) if sign_value(sign) < 0 else (~mask & (lock <= 0.0))
    return int((roi & target).sum()), area


def effect_gate(sign: str, box: Sequence[float], mask: np.ndarray, lock: np.ndarray, *, min_px: int = EFFECT_MIN_PX,
                min_frac: float = EFFECT_MIN_FRAC) -> bool:
    """LLMSeg 的 effect gate：还能改动的像素数 ≥ max(min_px, ceil(min_frac × 框面积)) 才保留。"""
    n, area = achievable_px(sign, box, mask, lock)
    return n >= max(int(min_px), int(math.ceil(min_frac * max(area, 1))))


def apply_signed(mask: np.ndarray, lock: np.ndarray, region: np.ndarray, sign: str, *, alpha: float = LOCK_ALPHA) -> None:
    """软锁定的写入（公式 4；就地修改 mask、lock）：Λ = 0 直接写、Λ = s；同号写并 Λ += s；反号 |Λ| −= α，≤ 0 时翻转并 Λ = s。"""
    s = sign_value(sign)
    target = s > 0
    free = region & (lock == 0.0)
    agree = region & (np.sign(lock) == s)
    oppose = region & (np.sign(lock) == -s)
    mask[free] = target
    lock[free] = s
    mask[agree] = target
    lock[agree] += s
    cand = np.abs(lock) - float(alpha)
    survive = oppose & (cand > 0.0)
    flip = oppose & (cand <= 0.0)
    lock[survive] = np.sign(lock[survive]) * cand[survive]
    mask[flip] = target
    lock[flip] = s


def correction_region(image_rgb: np.ndarray, mask: np.ndarray, box: Sequence[float], sign: str, *, rule: str = "grabcut",
                      context_pct: float | None = CONTEXT_PCT_DEFAULT, iterations: int = GRABCUT_ITERATIONS,
                      color_space: str = "rgb") -> tuple[np.ndarray, dict]:
    """一个修正框要补（"+"）或删（"−"）的像素（规则见模块说明第 3 条；不看真值）。返回 (区域, info)。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    m = _check_mask(mask, (h, w), "mask")
    s = sign_value(sign)
    roi = box_to_mask(box, h, w)
    if rule == "direct":
        return roi, {"rule": "direct"}
    if rule != "grabcut":
        raise ValueError(f"rule 必须是 {CORRECTION_RULES} 之一，收到 {rule!r}")
    gc = np.where(m, GC_FGD, GC_BGD).astype(np.uint8)
    if s > 0:
        gc[roi & ~m] = GC_PR_FGD
    else:
        gc[roi & m] = GC_PR_BGD
    try:
        crop = context_crop(box, context_pct, h, w)
        g = run_grabcut(img, gc, iterations=iterations, color_space=color_space, crop=crop)
    except (ValueError, cv2.error) as e:
        return roi, {"rule": "direct", "fallback": f"{type(e).__name__}: {str(e).strip()[:200]}"}
    return (roi & g) if s > 0 else (roi & ~g), {"rule": "grabcut"}


def correction_round(image_rgb: np.ndarray, mask: np.ndarray, lock: np.ndarray | None, proposals: Sequence[tuple[Sequence[float], str]],
                     *, rule: str = "grabcut", soft_lock: bool = True, alpha: float = LOCK_ALPHA, gate: bool = True,
                     confine_box: Sequence[float] | None = None, context_pct: float | None = CONTEXT_PCT_DEFAULT,
                     iterations: int = GRABCUT_ITERATIONS, color_space: str = "rgb",
                     gate_min_px: int = EFFECT_MIN_PX, gate_min_frac: float = EFFECT_MIN_FRAC,
                     region_cache: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """一轮带符号的局部修正（规则见模块说明第 3 条）。proposals：[(框, "+" | "-"), …]。返回 (新 mask, 新承诺场, info)；
    输入不改动。lock 为 None 时从全 0 开始；soft_lock = False 时承诺场不用、原样返回。
    region_cache（可选，同一张图上共用的 dict）：区域只取决于 (图, 本轮开始时的 mask, 框, 符号, 规则与参数)，算过的直接取
    （例如同一起点的第 1 轮，用不用软锁定结果相同）；结果与不用缓存逐位相同，info["n_region_cached"] 记命中数。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    cur = _check_mask(mask, (h, w), "mask").copy()
    lk = np.zeros((h, w), np.float64) if lock is None else np.asarray(lock, dtype=np.float64).copy()
    gate_lock = lk if soft_lock else np.zeros((h, w), np.float64)
    kept: list[tuple[tuple[float, ...], str]] = []
    n_empty = n_gated = 0
    for bx, sg in proposals:
        sign_value(sg)
        b = check_box(bx)
        if confine_box is not None:
            b = intersect_box(b, confine_box)
            if b is None:
                n_empty += 1
                continue
        if gate and not effect_gate(sg, b, cur, gate_lock, min_px=gate_min_px, min_frac=gate_min_frac):
            n_gated += 1
            continue
        kept.append((b, sg))
    regions = []
    n_fallback = n_cached = 0
    mkey = hashlib.sha1(np.packbits(cur).tobytes()).hexdigest() if (region_cache is not None and kept) else None
    for b, sg in kept:  # 都按本轮开始时的 mask 求区域
        key = None if mkey is None else (mkey, tuple(b), sg, rule, context_pct, iterations, color_space)
        if key is not None and key in region_cache:
            reg, inf = region_cache[key]
            n_cached += 1
        else:
            reg, inf = correction_region(img, cur, b, sg, rule=rule, context_pct=context_pct, iterations=iterations,
                                         color_space=color_space)
            if key is not None:
                region_cache[key] = (reg, inf)
        n_fallback += "fallback" in inf
        regions.append((reg, sg))
    out = cur.copy()
    for want_negative in (True, False):  # 先删后补
        for reg, sg in regions:
            if (sign_value(sg) < 0) != want_negative or not reg.any():
                continue
            if soft_lock:
                apply_signed(out, lk, reg, sg, alpha=alpha)
            else:
                out[reg] = sign_value(sg) > 0
    info = {"n_proposed": len(proposals), "n_outside_confine": n_empty, "n_gated": n_gated, "n_applied": len(kept),
            "n_fallback_direct": n_fallback, "n_region_cached": n_cached, "changed_px": int(np.count_nonzero(out ^ cur))}
    return out, lk, info


# ---------------------------------------------------------------------------
# 4. 生成器库 v2
# ---------------------------------------------------------------------------


def _p(kind: str, default: Any, doc: str, **kw: Any) -> dict:
    return {"type": kind, "default": default, "doc": doc, **kw}


_ITER = _p("integer", GRABCUT_ITERATIONS, "GrabCut iterations.", minimum=1, maximum=GRABCUT_MAX_ITERATIONS)
_BOX_PCT = _p("number", 0.0, "Each side of the box moves outward by this percent of the box width / height; negative moves inward.",
              minimum=-45.0, maximum=100.0)
_CS = _p("string", "rgb", "Color space of the image given to the color model.", enum=list(COLOR_SPACES))
_CTX = _p("number_or_null", None, "Run on the box expanded by this percent per side (at least CONTEXT_MIN_PX); null = whole image.",
          minimum=0.0, maximum=400.0)

GENERATOR_TYPES_V2: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "grabcut_default": {"doc": "W1-F: GrabCut initialized with the box, GRABCUT_ITERATIONS iterations, whole image (= W2's gc_default).",
                        "params": {}},
    "grabcut": {"doc": "GrabCut: box adjusted by box_pct; init rect (box = probable foreground, outside = definite background) or mask "
                       "(as W2's soft init); chosen iterations, color space and crop.",
                "params": {"iterations": _ITER, "box_pct": _BOX_PCT,
                           "init": _p("string", "rect", "rect or mask (see W2 grabcut).", enum=["rect", "mask"]),
                           "color_space": _CS, "context_pct": _CTX}},
    "gmm": {"doc": "Color-model segmentation without a graph cut (OpenCV's GrabCut fixes its GMMs at 5 components, and no max-flow "
                   "solver is available without new dependencies): k-component full-covariance GMMs (EM, deterministic quantile "
                   "initialization) for the foreground (box shrunk by fg_shrink_pct) and the background (a ring of ring_pct around "
                   "the box); per-pixel log-likelihood ratio, Gaussian-smoothed (sigma = smooth_pct of the box's shorter side), "
                   "> 0 inside the box is foreground; re-estimated from the current segmentation for `iterations` rounds.",
            "params": {"n_components": _p("integer", 5, "GMM components per model.", minimum=1, maximum=16), "color_space": _CS,
                       "iterations": _p("integer", 3, "Re-estimation rounds.", minimum=1, maximum=10),
                       "fg_shrink_pct": _p("number", 10.0, "Initial foreground samples: the box shrunk by this percent per side.",
                                           minimum=0.0, maximum=45.0),
                       "ring_pct": _p("number", 25.0, "Background samples: the box expanded by this percent per side, minus the box.",
                                      minimum=1.0, maximum=200.0),
                       "smooth_pct": _p("number", 1.0, "Smoothing sigma of the log-likelihood ratio, percent of the box's shorter side.",
                                        minimum=0.0, maximum=20.0)}},
    "superpixel": {"doc": "Superpixels (SLIC or Felzenszwalb) on the box expanded by context_pct; foreground = the union of the "
                          "superpixels selected by `select` (area: at least threshold_pct inside the box, as W2's slic; flood: superpixels "
                          "mostly inside the box that are not reached by a flood from the superpixels mostly outside the box through "
                          "neighbours with mean CIELAB distance <= tau; color: superpixels mostly inside the box whose mean CIELAB color "
                          "is nearer to the center seeds (box shrunk 30%) than to the outside seeds), intersected with the box.",
                   "params": {"method": _p("string", "slic", "slic or felzenszwalb.", enum=["slic", "felzenszwalb"]),
                              "n_segments": _p("integer", 50, "SLIC n_segments.", minimum=4, maximum=2000),
                              "compactness": _p("number", SLIC_COMPACTNESS, "SLIC compactness.", minimum=0.01, maximum=100.0),
                              "scale": _p("number", 100.0, "Felzenszwalb scale.", minimum=1.0, maximum=10000.0),
                              "fz_sigma": _p("number", 0.8, "Felzenszwalb sigma.", minimum=0.0, maximum=10.0),
                              "min_size": _p("integer", 20, "Felzenszwalb min_size.", minimum=1, maximum=10000),
                              "select": _p("string", "flood", "area, flood or color.", enum=["area", "flood", "color"]),
                              "threshold_pct": _p("number", 50.0, "area: kept if this percent lies inside the box.", minimum=1.0,
                                                  maximum=100.0),
                              "tau": _p("number", 10.0, "flood: CIELAB distance threshold.", minimum=0.0, maximum=200.0),
                              "context_pct": _p("number", 25.0, "Crop: the box expanded by this percent per side.", minimum=1.0,
                                                maximum=200.0)}},
    "watershed": {"doc": "Marker watershed on the color gradient of the whole image: foreground marker = the box shrunk by fg_shrink_pct, "
                         "background marker = outside the box expanded by bg_margin_pct (= W2's ws_pts / ws_center in box mode for "
                         "gradient rgb).",
                  "params": {"fg_shrink_pct": _p("number", 30.0, "Foreground marker shrink.", minimum=0.0, maximum=49.0),
                             "bg_margin_pct": _p("number", 0.0, "Background marker margin.", minimum=0.0, maximum=100.0),
                             "gradient": _p("string", "rgb", "rgb (W2's color_gradient) or lab.", enum=["rgb", "lab"])}},
    "box": {"doc": "The box itself (box_to_mask).", "params": {}},
})

# 名字 → 生成器（类型、参数、后处理）。前四个 = W2 冻结的 K = 4（configs/w2.json；名字同 src/w2/candidates 的自动命名）。
LIBRARY_V2: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "gc_default": {"type": "grabcut_default"},
    "gc_soft_box_minus10": {"type": "grabcut", "iterations": 5, "box_pct": -10.0, "init": "mask"},
    "gc_default+lcc+fill": {"type": "grabcut_default", "post": ["largest_cc", "fill_holes"]},
    "ws_pts": {"type": "watershed", "fg_shrink_pct": 30.0, "bg_margin_pct": 0.0, "gradient": "rgb"},
    "gc_iter10": {"type": "grabcut", "iterations": 10},
    "gc_lab": {"type": "grabcut", "color_space": "lab"},
    "gc_ctx50": {"type": "grabcut", "context_pct": 50.0},
    "slic_s50_flood": {"type": "superpixel", "method": "slic", "n_segments": 50, "select": "flood"},
    "felz_k100_flood": {"type": "superpixel", "method": "felzenszwalb", "scale": 100.0, "select": "flood"},
    "gmm_k3_lab+lcc+fill": {"type": "gmm", "n_components": 3, "color_space": "lab", "post": ["largest_cc", "fill_holes"]},
    "ws_lab_shrink20_margin5": {"type": "watershed", "fg_shrink_pct": 20.0, "bg_margin_pct": 5.0, "gradient": "lab"},
    "gc_default+open3+lcc+fill": {"type": "grabcut_default", "post": ["open:3", "largest_cc", "fill_holes"]},
    "gc_hsv": {"type": "grabcut", "color_space": "hsv"},
    "slic_s100_color": {"type": "superpixel", "method": "slic", "n_segments": 100, "select": "color"},
    "gmm_k8_rgb+lcc+fill": {"type": "gmm", "n_components": 8, "color_space": "rgb", "post": ["largest_cc", "fill_holes"]},
    "gc_box_minus5+close5+fill": {"type": "grabcut", "box_pct": -5.0, "post": ["close:5", "fill_holes"]},
})
LIBRARY_V2_ORDER: tuple[str, ...] = tuple(LIBRARY_V2)
W2_FROZEN_NAMES: tuple[str, ...] = LIBRARY_V2_ORDER[:4]


def _check_value(where: str, pname: str, schema: Mapping[str, Any], v: Any) -> Any:
    kind = schema["type"]
    if kind == "string":
        if v not in schema["enum"]:
            raise ValueError(f"{where}：参数 {pname} 必须是 {schema['enum']} 之一，收到 {v!r}")
        return v
    if kind == "number_or_null" and v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}：参数 {pname} 必须是有限的数，收到 {v!r}")
    if kind == "integer":
        if int(v) != v:
            raise ValueError(f"{where}：参数 {pname} 必须是整数，收到 {v!r}")
        v = int(v)
    else:
        v = float(v)
    if not schema["minimum"] <= v <= schema["maximum"]:
        raise ValueError(f"{where}：参数 {pname} = {v:g} 不在 [{schema['minimum']:g}, {schema['maximum']:g}] 内")
    return v


def resolve_v2(name: str, entry: Mapping[str, Any] | None = None) -> dict:
    """库里的一项（或给定的 entry）校验并补全缺省参数：{"type", 参数…, "post": [...], "name"}。"""
    e = dict(LIBRARY_V2[name] if entry is None else entry)
    t = e.get("type")
    if t not in GENERATOR_TYPES_V2:
        raise ValueError(f"{name}：未知的类型 {t!r}；可选 {sorted(GENERATOR_TYPES_V2)}")
    params = GENERATOR_TYPES_V2[t]["params"]
    extra = set(e) - set(params) - {"type", "post"}
    if extra:
        raise ValueError(f"{name}（{t}）：未知的参数 {sorted(extra)}")
    r: dict[str, Any] = {"type": t}
    for pname, schema in params.items():
        r[pname] = _check_value(f"{name}（{t}）", pname, schema, e.get(pname, schema["default"]))
    r["post"] = [op if rad is None else f"{op}:{rad}" for op, rad in parse_post(e.get("post", []))]
    r["name"] = name
    return r


def _gen_grabcut_default(img: np.ndarray, box: tuple, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    return grabcut_mask(img, box_abs=[float(v) for v in box], iterations=GRABCUT_ITERATIONS), {}


def _gen_grabcut(img: np.ndarray, box: tuple, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    adj = adjust_box(box, r["box_pct"])
    kw: dict[str, Any] = {"iterations": r["iterations"], "color_space": r["color_space"]}
    if r["init"] == "rect":
        kw["box_abs"] = list(adj)
    else:
        kw["init_mask"] = box_to_mask(adj, h, w)
    m = grabcut_mask(img, **kw, context_pct=r["context_pct"])
    return m, {"box_used": [round(v, 3) for v in adj]}


def _color_features(img: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return img.astype(np.float64)
    return _gc_image(img, color_space).astype(np.float64)


def _subsample(x: np.ndarray, n_max: int = 4000) -> np.ndarray:
    if len(x) <= n_max:
        return x
    return x[:: int(math.ceil(len(x) / n_max))]


def fit_gmm(x: np.ndarray, k: int, *, em_iters: int = 10, reg: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k 个分量的全协方差 GMM（EM）。初始化确定：按三通道之和排序后等分成 k 段，各段的均值与协方差。返回 (权重, 均值, 协方差)。"""
    x = np.asarray(x, dtype=np.float64)
    k = int(max(1, min(k, len(x))))
    order = np.argsort(x.sum(axis=1), kind="stable")
    chunks = np.array_split(order, k)
    eye = np.eye(x.shape[1]) * reg
    mu = np.stack([x[c].mean(axis=0) for c in chunks])
    cov = np.stack([(np.cov(x[c].T) if len(c) > 1 else np.zeros((3, 3))) + eye for c in chunks])
    wt = np.array([len(c) for c in chunks], np.float64) / len(x)
    for _ in range(em_iters):
        lp = _component_logpdf(x, wt, mu, cov)
        mx = lp.max(axis=1, keepdims=True)
        resp = np.exp(lp - mx)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-9
        wt = nk / nk.sum()
        mu = (resp.T @ x) / nk[:, None]
        for j in range(k):
            d = x - mu[j]
            cov[j] = (resp[:, j, None] * d).T @ d / nk[j] + eye
    return wt, mu, cov


def _component_logpdf(x: np.ndarray, wt: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    out = np.empty((len(x), len(wt)))
    for j in range(len(wt)):
        ch = np.linalg.cholesky(cov[j])
        z = np.linalg.solve(ch, (x - mu[j]).T)
        out[:, j] = (np.log(max(wt[j], 1e-12)) - 0.5 * (z * z).sum(axis=0) - np.log(np.diag(ch)).sum()
                     - 0.5 * x.shape[1] * math.log(2 * math.pi))
    return out


def gmm_loglik(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    lp = _component_logpdf(np.asarray(x, dtype=np.float64), *model)
    mx = lp.max(axis=1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True)))[:, 0]


def _gen_gmm(img: np.ndarray, box: tuple, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    inside = box_to_mask(box, h, w)
    if not inside.any():
        raise ValueError("框里没有像素")
    ring = box_to_mask(adjust_box(box, r["ring_pct"]), h, w) & ~inside
    if not ring.any():
        raise ValueError("框外没有背景样本（框盖满了整张图）")
    fg0 = box_to_mask(adjust_box(box, -r["fg_shrink_pct"]), h, w)
    if not fg0.any():
        fg0 = inside
    feat = _color_features(img, r["color_space"])
    r0, r1, c0, c1 = context_crop(adjust_box(box, r["ring_pct"]), 0.0, h, w)
    sub = feat[r0:r1, c0:c1].reshape(-1, 3)
    in_c = inside[r0:r1, c0:c1]
    sigma = r["smooth_pct"] / 100.0 * box_short_side(box)
    fg = fg0
    for _ in range(r["iterations"]):
        mf = fit_gmm(_subsample(feat[fg]), r["n_components"])
        bg_s = ring | (inside & ~fg)
        mb = fit_gmm(_subsample(feat[bg_s]), r["n_components"])
        llr = (gmm_loglik(sub, mf) - gmm_loglik(sub, mb)).reshape(r1 - r0, c1 - c0)
        if sigma > 0:
            llr = ndimage.gaussian_filter(llr, sigma=sigma, mode="nearest")
        cur = np.zeros((h, w), bool)
        cur[r0:r1, c0:c1] = (llr > 0) & in_c
        if not cur.any():
            break
        fg = cur
    return (fg & inside), {"fg_px": int((fg & inside).sum())}


def _superpixel_labels(crop: np.ndarray, r: Mapping[str, Any]) -> np.ndarray:
    if r["method"] == "slic":
        return slic(np.ascontiguousarray(crop), n_segments=r["n_segments"], compactness=r["compactness"], sigma=SLIC_SIGMA,
                    start_label=1, channel_axis=-1)
    return felzenszwalb(np.ascontiguousarray(crop), scale=r["scale"], sigma=r["fz_sigma"], min_size=r["min_size"],
                        channel_axis=-1) + 1


def _gen_superpixel(img: np.ndarray, box: tuple, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    inside_full = box_to_mask(box, h, w)
    if not inside_full.any():
        raise ValueError("框里没有像素")
    r0, r1, c0, c1 = context_crop(box, r["context_pct"], h, w)
    crop = img[r0:r1, c0:c1]
    lab = _superpixel_labels(crop, r)
    n = int(lab.max())
    inside = inside_full[r0:r1, c0:c1]
    total = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    n_in = np.bincount(lab[inside], minlength=n + 1).astype(np.float64)
    frac_in = np.divide(n_in, total, out=np.zeros_like(n_in), where=total > 0)
    mostly_in = frac_in >= 0.5
    mostly_in[0] = False
    if r["select"] == "area":
        keep = n_in * 100.0 >= r["threshold_pct"] * total
        keep[0] = False
    else:
        lab_img = rgb2lab(crop).reshape(-1, 3)
        cnt = np.maximum(total, 1)[:, None]
        mean = np.stack([np.bincount(lab.ravel(), weights=lab_img[:, j], minlength=n + 1) for j in range(3)], axis=1) / cnt
        outside_seed = (~mostly_in) & (total > 0)
        outside_seed[0] = False
        if not outside_seed.any():
            raise ValueError("框外没有超像素（框盖满了裁剪区域）")
        if r["select"] == "flood":
            pairs = np.concatenate([np.stack([lab[:, :-1].ravel(), lab[:, 1:].ravel()], 1),
                                    np.stack([lab[:-1, :].ravel(), lab[1:, :].ravel()], 1)])
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            pairs = np.unique(np.sort(pairs, axis=1), axis=0)
            nbrs: dict[int, list[int]] = {}
            for a, b in pairs.tolist():
                nbrs.setdefault(a, []).append(b)
                nbrs.setdefault(b, []).append(a)
            reached = outside_seed.copy()
            stack = sorted(np.flatnonzero(outside_seed).tolist())
            while stack:
                a = stack.pop()
                for b in nbrs.get(a, ()):
                    if not reached[b] and float(np.linalg.norm(mean[a] - mean[b])) <= r["tau"]:
                        reached[b] = True
                        stack.append(b)
            keep = mostly_in & ~reached
        else:
            center = box_to_mask(adjust_box(box, -30.0), h, w)[r0:r1, c0:c1]
            fg_seed = np.zeros(n + 1, bool)
            fg_seed[np.unique(lab[center])] = True
            fg_seed[0] = False
            fg_seed &= mostly_in
            if not fg_seed.any():
                raise ValueError("框中心没有超像素")
            df = np.linalg.norm(mean[:, None, :] - mean[None, fg_seed, :], axis=2).min(axis=1)
            db = np.linalg.norm(mean[:, None, :] - mean[None, outside_seed, :], axis=2).min(axis=1)
            keep = mostly_in & (df < db)
    out = np.zeros((h, w), bool)
    out[r0:r1, c0:c1] = keep[lab] & inside
    return out, {"crop": [c0, r0, c1, r1], "n_superpixels": n, "n_kept": int(keep.sum())}


def _gen_watershed(img: np.ndarray, box: tuple, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    fg = box_to_mask(adjust_box(box, -r["fg_shrink_pct"]), h, w)
    bg = ~box_to_mask(adjust_box(box, r["bg_margin_pct"]), h, w)
    if not fg.any():
        raise ValueError("前景标记为空（框太小）")
    markers = np.zeros((h, w), np.int32)
    markers[bg] = 1
    markers[fg] = 2
    if not (markers == 1).any():
        raise ValueError("背景标记为空（框盖满了整张图）")
    grad = color_gradient(img) if r["gradient"] == "rgb" else lab_gradient(img)
    return sk_watershed(grad, markers) == 2, {"fg_marker_px": int(fg.sum())}


def _gen_box(img: np.ndarray, box: tuple, r: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    return box_to_mask(box, *img.shape[:2]), {}


_GEN_FNS: Mapping[str, Callable[[np.ndarray, tuple, Mapping[str, Any]], tuple[np.ndarray, dict]]] = MappingProxyType({
    "grabcut_default": _gen_grabcut_default, "grabcut": _gen_grabcut, "gmm": _gen_gmm, "superpixel": _gen_superpixel,
    "watershed": _gen_watershed, "box": _gen_box,
})
assert set(_GEN_FNS) == set(GENERATOR_TYPES_V2)


def generate_v2(image_rgb: np.ndarray, box: Sequence[float], names: Sequence[str],
                library: Mapping[str, Mapping[str, Any]] | None = None) -> list[ToolResult]:
    """按名字列表生成候选（顺序同 names）。失败的生成器 mask 用框本身、ok = False；同类型同参数、只差后处理的共用一次基础计算。"""
    img = _check_image(image_rgb)
    h, w = img.shape[:2]
    bx = check_box(box)
    lib = LIBRARY_V2 if library is None else library
    cache: dict[str, tuple[np.ndarray | None, dict, str | None]] = {}
    out = []
    for nm in names:
        r = resolve_v2(nm, lib[nm])
        params = {k: v for k, v in r.items() if k not in ("type", "post", "name")}
        key = json.dumps([r["type"], params], sort_keys=True)
        if key not in cache:
            try:
                m, extra = _GEN_FNS[r["type"]](img, bx, r)
                cache[key] = (np.asarray(m, dtype=bool), extra, None)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as e:
                cache[key] = (None, {}, f"{type(e).__name__}: {str(e).strip()[:300]}")
        base, extra, err = cache[key]
        mask = apply_post(base, r["post"]) if base is not None else box_to_mask(bx, h, w)
        out.append(ToolResult(mask=mask, ok=base is not None, info={"type": r["type"], "params": params, "post": r["post"],
                                                                   "error": err, **extra}, name=nm))
    return out


__all__ = [
    "BAND_MIN_PX", "BAND_PCT_DEFAULT", "BOX_INITS", "COLOR_SPACES", "CONTEXT_PCT_DEFAULT", "CORRECTION_RULES", "EFFECT_MIN_FRAC",
    "EFFECT_MIN_PX", "GENERATOR_TYPES_V2", "GRABCUT_ITERATIONS", "LIBRARY_V2", "LIBRARY_V2_ORDER", "LOCK_ALPHA", "SIGNS",
    "SIGN_ADD", "SIGN_REMOVE", "SPLINE_GC_MODES", "ToolResult", "W2_FROZEN_NAMES", "achievable_px", "apply_post", "apply_signed",
    "bspline_basis", "bspline_samples", "contour_band", "correction_region", "correction_round", "effect_gate",
    "eval_closed_bspline", "fill_polygon_evenodd", "fit_closed_bspline", "fit_mask_bspline", "generate_v2", "grabcut_mask",
    "points_grabcut", "rasterize_loops", "resolve_v2", "run_grabcut", "spline_grabcut", "spline_polygon",
]
