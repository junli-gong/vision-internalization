"""W1 工作台：给冻结的 LLM 用的不含学习的通用工具（DEV-0050；D103 §二.2；接口约定见 docs/w1_plan.md §2）。

原则（D103）：语义判断由 LLM 做，底层拟合交给不含学习的算法。工作台与任务无关：所有任务用同一套工具，
工具说明里不写任何任务的解法。工具清单、参数、算法与固定参数见 docs/w1_tools.md（论文附录用）。

约定：
- 坐标：所有工具的坐标都是整张原图的 0–1000 相对坐标，x 在前（与 Qwen 的原生框格式一致）；
  换算成绝对像素一律走 src/executors/geometry.norm1000_to_abs。长度（半径）以原图像素计。
- 光栅化：框用 src/executors/raster.box_to_mask（像素中心落在 (lo, hi] 内），多边形与样条用 polygons_to_mask
  （pycocotools，与 COCO/RefCOCO 的 GT 同一套规则）。
- 状态：命名 mask（原图分辨率的 bool 数组）；图像注册表（id → PIL 图，"original" 是原图）；最近一次 SLIC 的标号图；
  submit_* 写入的最终答案。
- 出错（参数不对、越界、未知的 mask 名、算法失败）返回 ToolResult.error，不抛异常。
- 确定性：同样的调用序列逐位得到同样的结果。GrabCut 每次调用前把 OpenCV 的随机数种子设成固定值（它内部用
  k-means 初始化 GMM）；连通域用 scipy.ndimage.label（行优先扫描，标号顺序确定）。
- 被关掉的组（disabled_groups，消融用）不出现在 specs() 里；call() 调到时与未知工具返回同样的错误。
- 图像（管理员裁定，DEV-0050 审查）：把 mask 叠回原图属于"核对"，只有 check 组的 show 做。hand 组与 snap 组的
  其余工具只返回文字（mask 的面积、占比、0–1000 外接框）；返回图的只有 eye 组（原图的视图）与 snap 组里
  两个"看图本身"的视图：edges（边缘图）与 superpixels（带编号的分割图，选编号要用）。这样 A2（去掉 check）是干净的。

禁止项（D103）：任何学习型视觉模型或预训练权重，OpenCV 的 DNN 模块。这里只用 numpy、scipy、OpenCV 与
scikit-image 里不含学习的函数，且只用 OpenCV 4.10 与 5.x 都有的 API。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import sobel
from skimage.segmentation import find_boundaries, slic
from skimage.segmentation import watershed as sk_watershed

from src.constants import GUIDED_FILTER_EPS, GUIDED_FILTER_RADIUS, NORM1000_SCALE, OVERLAY_ALPHA_DEN, OVERLAY_ALPHA_NUM, OVERLAY_CONTOUR_WIDTH
from src.executors.geometry import norm1000_to_abs
from src.executors.overlay import render_overlay
from src.executors.raster import box_to_mask, polygons_to_mask
from src.executors.upsample import guide_to_unit, guided_filter
from src.w1.sandbox import STDOUT_LIMIT_CHARS, SandboxConfig, run_code

from src.adapters.chat import ToolSpec  # DEV-0049 的定义（docs/w1_plan.md §1；占位已在合入时删除，D106）


# ---------------------------------------------------------------------------
# 组与固定参数（docs/w1_tools.md 引用这里的名字；改动要同步文档，测试钉住）
# ---------------------------------------------------------------------------

GROUPS = ("eye", "hand", "snap", "check", "code", "submit")  # 消融：A1 去掉 eye；A2 去掉 check；A3 去掉 snap 与 code

ZOOM_LONG_SIDE = 640  # zoom 与 show(region) 的视图长边（px）。640×480 在 PixelBudget(300) 下恰为 300 个视觉 token
MAX_POINTS = 64  # 一次调用最多的点数（read_pixels、draw_points、draw_polygon、draw_spline、select_superpixels）
MAX_MASKS = 32  # 同时存在的命名 mask 上限
MASK_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_\-]{0,31}$"
MASK_MODES = ("new", "union", "subtract", "intersect")
READ_PIXELS_MAX_RADIUS = 20
TILE_MAX = 8
POINT_RADIUS_DEFAULT = 3.0  # draw_points 的圆盘半径（px）
POINT_RADIUS_MAX = 200.0
SPLINE_SAMPLES_PER_SEGMENT = 16  # 闭合 Catmull–Rom 样条每段的采样点数
SPLINE_ALPHA = 0.5  # 0.5 = 向心参数化（不会在控制点附近打结或自交）
CANNY_LOW = 50.0
CANNY_HIGH = 3 * CANNY_LOW  # 高阈值 = 3 × 低阈值（Canny 1986 建议 2:1 到 3:1）
CANNY_THRESHOLD_MAX = 2000.0
CANNY_BLUR_KSIZE = 5  # Canny 之前的高斯模糊核边长与 sigma（Canny 1986 的做法；OpenCV 的 Canny 本身不模糊）
CANNY_BLUR_SIGMA = 1.4
GRABCUT_ITERATIONS = 5
GRABCUT_MAX_ITERATIONS = 20
GRABCUT_RNG_SEED = 0
# mask 初始化时"可能背景"的范围：mask 外接框每边按框宽、框高外扩这个比例；更远处记"确定背景"（同框初始化）。
# 不设这个范围时，GrabCut 可以把前景长到整张图（开发机彩排里一个粗多边形长成了全图的 18%，耗时 2.3 s）。暂定，在开发-调参集上定。
GRABCUT_MASK_MARGIN = 0.2
SLIC_N_SEGMENTS = 100
SLIC_MAX_SEGMENTS = 600
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 1.0
REFINE_RADIUS = GUIDED_FILTER_RADIUS  # 引导滤波的默认参数沿用 constants（E3 的引导上采样同一组值）
REFINE_EPS = GUIDED_FILTER_EPS
REFINE_THRESHOLD = 0.5  # 滤波后 >= 0.5 记前景
REFINE_MAX_RADIUS = 32
MORPH_OPS = ("dilate", "erode", "open", "close", "fill_holes")
MORPH_RADIUS_DEFAULT = 2
MORPH_RADIUS_MAX = 50
COMPONENTS_LIST_MAX = 20
COUNT_MAX = 10000
LUMA_WEIGHTS = (0.299, 0.587, 0.114)  # 亮度系数：ITU-R 标清电视的 luma 定义，与 OpenCV 的 COLOR_RGB2GRAY 相同
CODE_RESULT_DEFAULT = "code_result"
# 绘制：网格线、编号、超像素边界的颜色（RGB）与 Hershey 字体（OpenCV 自带的矢量字体，不依赖 FreeType，跨机器一致）
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_YELLOW = (255, 225, 25)

_MASK_NAME_RE = re.compile(MASK_NAME_PATTERN)


# ---------------------------------------------------------------------------
# 公共数据类型（docs/w1_plan.md §2）
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    text: str  # 给模型看的文字结果（坐标、读数、报错）
    images: list = field(default_factory=list)  # 给模型看的图（PIL.Image，RGB）
    error: str | None = None  # 参数错误、越界、沙箱报错；error 时 text 写明原因


@dataclass(frozen=True, eq=False)
class Submission:
    """submit_* 写入的最终答案。kind = "mask" | "box" | "count"，对应字段非空。"""

    kind: str
    mask: np.ndarray | None = None  # (H, W) bool，原图分辨率
    box: tuple[float, float, float, float] | None = None  # 绝对像素 xyxy（浮点）
    count: int | None = None
    source: str | None = None  # kind="mask" 时是提交的 mask 名


class ToolError(Exception):
    """工具的预期错误（参数、越界、状态）：call() 把它转成 ToolResult.error。"""


# ---------------------------------------------------------------------------
# 与工作台状态无关的算法函数（W1-F 等固定流程直接调用同一个函数）
# ---------------------------------------------------------------------------


def grabcut_mask(
    image_rgb: np.ndarray,
    *,
    box_abs: Sequence[float] | None = None,
    init_mask: np.ndarray | None = None,
    fg: np.ndarray | None = None,
    bg: np.ndarray | None = None,
    iterations: int = GRABCUT_ITERATIONS,
) -> np.ndarray:
    """GrabCut（Rother, Kolmogorov, Blake 2004；cv2.grabCut，GC_INIT_WITH_MASK）。返回 (H, W) bool。

    初始化二选一：
    - box_abs（绝对像素 xyxy）：框内（box_to_mask 的像素中心规则）记"可能前景"，框外记"确定背景"，
      与 OpenCV 的 GC_INIT_WITH_RECT 同义，只是框的光栅化与工作台其余部分一致；
    - init_mask：mask 内记"可能前景"；mask 外、但在它的外接框每边外扩 GRABCUT_MASK_MARGIN（按框宽、框高）
      以内的记"可能背景"；更远处记"确定背景"。
    fg / bg（可选）：确定前景 / 确定背景的种子，覆盖初始化。每次调用前 cv2.setRNGSeed(GRABCUT_RNG_SEED)，结果确定。
    参数不合法或某一类没有像素时抛 ValueError；OpenCV 内部失败抛 cv2.error。
    """
    img = np.asarray(image_rgb)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image_rgb 必须是 (H, W, 3) uint8，收到 dtype {img.dtype}、形状 {img.shape}")
    h, w = img.shape[:2]
    if (box_abs is None) == (init_mask is None):
        raise ValueError("box_abs 与 init_mask 必须恰好给一个")
    if isinstance(iterations, bool) or int(iterations) != iterations or not 1 <= iterations <= GRABCUT_MAX_ITERATIONS:
        raise ValueError(f"iterations 必须是 1..{GRABCUT_MAX_ITERATIONS} 的整数，收到 {iterations}")
    if box_abs is not None:
        inside = box_to_mask(box_abs, h, w)
        gc = np.where(inside, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    else:
        inside = _check_mask(init_mask, (h, w), "init_mask")
        gc = np.full((h, w), cv2.GC_BGD, np.uint8)
        if inside.any():
            rows, cols = np.flatnonzero(inside.any(axis=1)), np.flatnonzero(inside.any(axis=0))
            r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
            mr = int(math.ceil(GRABCUT_MASK_MARGIN * (r1 - r0)))
            mc = int(math.ceil(GRABCUT_MASK_MARGIN * (c1 - c0)))
            gc[max(r0 - mr, 0) : min(r1 + mr, h), max(c0 - mc, 0) : min(c1 + mc, w)] = cv2.GC_PR_BGD
        gc[inside] = cv2.GC_PR_FGD
    if fg is not None:
        gc[_check_mask(fg, (h, w), "fg")] = cv2.GC_FGD
    if bg is not None:
        gc[_check_mask(bg, (h, w), "bg")] = cv2.GC_BGD
    is_fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    if not is_fg.any():
        raise ValueError("GrabCut 的初始化里没有前景像素")
    if is_fg.all():
        raise ValueError("GrabCut 的初始化里没有背景像素")
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(GRABCUT_RNG_SEED)
    cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, bgd, fgd, int(iterations), cv2.GC_INIT_WITH_MASK)
    return (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)


def spline_polygon(points_abs: np.ndarray, samples_per_segment: int = SPLINE_SAMPLES_PER_SEGMENT) -> np.ndarray:
    """闭合的向心 Catmull–Rom 样条（Barry–Goldman 金字塔公式，alpha = SPLINE_ALPHA），经过每个控制点。

    points_abs：(n, 2) 绝对像素，n ≥ 3，相邻点（含首尾）不得重合。返回 (n·samples_per_segment, 2) 的多边形顶点，
    第 i 段从控制点 i 出发、到 i+1 之前为止。
    """
    p = np.asarray(points_abs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError(f"样条至少要 3 个 (x, y) 控制点，收到形状 {p.shape}")
    n = p.shape[0]
    seg = np.linalg.norm(p[(np.arange(n) + 1) % n] - p, axis=1)
    if np.any(seg == 0):
        raise ValueError("样条的相邻控制点（含首尾）不得重合")
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + float(np.linalg.norm(p1 - p0)) ** SPLINE_ALPHA
        t2 = t1 + float(np.linalg.norm(p2 - p1)) ** SPLINE_ALPHA
        t3 = t2 + float(np.linalg.norm(p3 - p2)) ** SPLINE_ALPHA
        t = np.linspace(t1, t2, samples_per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.concatenate(out, axis=0)


def disk_footprint(radius: int) -> np.ndarray:
    """半径 radius 的圆盘结构元（uint8，边长 2r+1）：(i, j) 到中心的距离 ≤ r 即为 1。"""
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return ((xx * xx + yy * yy) <= r * r).astype(np.uint8)


def color_gradient(image_rgb: np.ndarray) -> np.ndarray:
    """彩色梯度幅值：逐通道 Sobel（skimage.filters.sobel，图像先归一化到 [0, 1]），三通道平方和开方。"""
    img = np.asarray(image_rgb).astype(np.float64) / 255.0
    return np.sqrt(sum(sobel(img[..., c]) ** 2 for c in range(3)))


def _check_mask(m, shape, name) -> np.ndarray:
    a = np.asarray(m)
    if a.dtype != bool or a.shape != tuple(shape):
        raise ValueError(f"{name} 必须是形状 {tuple(shape)} 的 bool 数组，收到 dtype {a.dtype}、形状 {a.shape}")
    return a


def _resize_long_side(arr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    s = long_side / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(arr), (nw, nh), interpolation=interp)


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, thickness: int) -> None:
    """白字黑边（先画粗的黑字再画白字），org 是文字左下角。"""
    cv2.putText(img, text, org, _FONT, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, _WHITE, thickness, cv2.LINE_AA)


def _fmt(v: float) -> str:
    """0–1000 坐标的显示：一位小数，去掉多余的 .0。"""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return repr(v)


# ---------------------------------------------------------------------------
# 参数的 JSON Schema 与解析（模板里 <parameter=…> 的值都是字符串，这里按 schema 解析）
# ---------------------------------------------------------------------------


def _p_coord(desc: str) -> dict:
    return {"type": "number", "minimum": 0, "maximum": 1000, "description": desc}


def _p_name(desc: str) -> dict:
    # "x-mask-name"：内部标记（按 MASK_NAME_PATTERN 校验），spec() 导出时去掉，省提示 token
    return {"type": "string", "description": desc, "x-mask-name": True}


def _p_points(desc: str, min_items: int = 1) -> dict:
    return {
        "type": "array",
        "items": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2},
        "minItems": min_items,
        "maxItems": MAX_POINTS,
        "description": desc,
    }


def _p_mode() -> dict:
    return {
        "type": "string",
        "enum": list(MASK_MODES),
        "default": "new",
        "description": "new: create or overwrite the mask; union / subtract / intersect: combine with the existing mask of that name.",
    }


_BOX_PARAMS = {
    "x1": _p_coord("Left edge, 0-1000 (full image)."),
    "y1": _p_coord("Top edge, 0-1000 (full image)."),
    "x2": _p_coord("Right edge, 0-1000 (full image)."),
    "y2": _p_coord("Bottom edge, 0-1000 (full image)."),
}
_REGION = {
    "type": "array",
    "items": {"type": "number", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
    "description": "Optional region [x1, y1, x2, y2] in 0-1000 full-image coordinates.",
}


def _parse_value(value: Any, schema: dict, pname: str, raw: bool = False) -> Any:
    t = schema["type"]
    if t == "string":
        if not isinstance(value, str):
            raise ToolError(f"parameter '{pname}' must be a string")
        if raw:
            return value
        s = value.strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            try:
                s = json.loads(s)
            except json.JSONDecodeError:
                pass
        if "enum" in schema and s not in schema["enum"]:
            raise ToolError(f"parameter '{pname}' must be one of {schema['enum']}; got {s!r}")
        if schema.get("x-mask-name") and not _MASK_NAME_RE.match(s):
            raise ToolError(f"parameter '{pname}' = {s!r} is not a valid name (letters, digits, '_' or '-', starting with a letter, at most 32 characters)")
        return s
    if isinstance(value, str):
        s = value.strip()
        if t == "boolean":
            s = s.lower()
        try:
            value = json.loads(s)
        except json.JSONDecodeError:
            raise ToolError(f"parameter '{pname}' must be {_type_phrase(schema)}; got {s[:80]!r}") from None
        if isinstance(value, str) and t in ("number", "integer"):
            try:
                value = float(value)
            except ValueError:
                raise ToolError(f"parameter '{pname}' must be {_type_phrase(schema)}; got {value[:80]!r}") from None
    if t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ToolError(f"parameter '{pname}' must be {_type_phrase(schema)}; got {value!r}")
        if t == "integer":
            if float(value) != int(value):
                raise ToolError(f"parameter '{pname}' must be an integer; got {value!r}")
            value = int(value)
        else:
            value = float(value)
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            rng = f"[{'-inf' if lo is None else f'{lo:g}'}, {'inf' if hi is None else f'{hi:g}'}]"
            raise ToolError(f"parameter '{pname}' = {value:g} is out of range {rng}")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolError(f"parameter '{pname}' must be one of {schema['enum']}; got {value:g}")
        return value
    if t == "boolean":
        if not isinstance(value, bool):
            raise ToolError(f"parameter '{pname}' must be true or false; got {value!r}")
        return value
    if t == "array":
        items = schema["items"]
        if not isinstance(value, list) and items["type"] in ("number", "integer") and schema.get("minItems", 1) <= 1:
            value = [value]  # 只要一个数的数组，接受裸数
        if not isinstance(value, list):
            raise ToolError(f"parameter '{pname}' must be {_type_phrase(schema)}; got {value!r}")
        n_lo, n_hi = schema.get("minItems", 0), schema.get("maxItems")
        if len(value) < n_lo or (n_hi is not None and len(value) > n_hi):
            rng = f"exactly {n_lo}" if n_lo == n_hi else f"{n_lo}..{n_hi}"
            raise ToolError(f"parameter '{pname}' must have {rng} items; got {len(value)}")
        return [_parse_value(v, items, f"{pname}[{i}]") for i, v in enumerate(value)]
    raise AssertionError(f"未知的 schema 类型 {t}")


def _type_phrase(schema: dict) -> str:
    t = schema["type"]
    if t == "array":
        return "a JSON array" + (" of [x, y] pairs" if schema["items"]["type"] == "array" else "")
    return {"number": "a number", "integer": "an integer", "boolean": "true or false"}.get(t, t)


@dataclass(frozen=True)
class _Tool:
    name: str
    group: str
    description: str
    properties: dict
    required: tuple[str, ...]
    handler: Callable[..., "ToolResult"]
    raw_params: frozenset[str] = frozenset()

    def spec(self) -> ToolSpec:
        params = {"type": "object", "properties": self.properties, "required": list(self.required), "additionalProperties": False}
        return ToolSpec(name=self.name, description=self.description, parameters=_strip_internal(json.loads(json.dumps(params))))


def _strip_internal(v):
    """去掉 schema 里以 "x-" 开头的内部键（只给解析器用，不给模型看）。"""
    if isinstance(v, dict):
        return {k: _strip_internal(x) for k, x in v.items() if not k.startswith("x-")}
    if isinstance(v, list):
        return [_strip_internal(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# 工作台
# ---------------------------------------------------------------------------


class Workbench:
    """一条样本一个实例。image 是原图（PIL，任何模式都先转 RGB）。"""

    def __init__(self, image: Image.Image, *, disabled_groups: frozenset[str] = frozenset(), sandbox: SandboxConfig | None = None):
        bad = set(disabled_groups) - set(GROUPS)
        if bad:
            raise ValueError(f"未知的工具组 {sorted(bad)}；可选 {GROUPS}")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        rgb.setflags(write=False)
        self._rgb = rgb
        self._h, self._w = int(rgb.shape[0]), int(rgb.shape[1])
        self._disabled = frozenset(disabled_groups)
        self._sandbox = sandbox or SandboxConfig()
        self._masks: dict[str, np.ndarray] = {}
        self._images: dict[str, Image.Image] = {"original": Image.fromarray(rgb.copy())}
        self._superpixels: np.ndarray | None = None
        self._submission: Submission | None = None
        self._log: list[dict] = []
        self._cur: dict = self._fresh_call_record()
        self._guide: np.ndarray | None = None
        self._gradient: np.ndarray | None = None
        self._tools = {t.name: t for t in self._build_tools()}

    # ---- 公共接口 ----------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        """当前可用的工具（已去掉 disabled 组），顺序固定：按 GROUPS，组内按定义顺序。"""
        return [t.spec() for t in self._tools.values() if t.group not in self._disabled]

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        self._cur = self._fresh_call_record()
        tool = self._tools.get(name) if isinstance(name, str) else None
        entry: dict = {
            "index": len(self._log),
            "tool": name if isinstance(name, str) else repr(name),
            "group": tool.group if tool is not None else None,
            "arguments": _jsonable(arguments) if isinstance(arguments, Mapping) else repr(arguments),
        }
        internal = False
        try:
            if tool is None or tool.group in self._disabled:
                raise ToolError(f"unknown tool '{name}'")
            if not isinstance(arguments, Mapping):
                raise ToolError("arguments must be an object of named parameters")
            if tool.name == "run_python" and isinstance(arguments.get("code"), str):
                self._cur["code"] = arguments["code"]
            result = tool.handler(**self._parse_args(tool, arguments))
        except ToolError as e:
            result = ToolResult(text=f"Error: {e}", images=[], error=str(e))
        except Exception as e:  # noqa: BLE001 —— 工具内部的意外错误也不抛出，日志里标 internal_error
            internal = True
            msg = f"internal error in tool '{name}': {type(e).__name__}: {e}"
            result = ToolResult(text=f"Error: {msg}", images=[], error=msg)
        entry.update(
            wall_time_s=time.perf_counter() - t0,
            ok=result.error is None,
            error=result.error,
            internal_error=internal,
            masks_written=dict(self._cur["masks_written"]),
            images=list(self._cur["images"]),
            code=self._cur["code"],
            sandbox=self._cur["sandbox"],
            submission=self._submission_summary() if tool is not None and tool.group == "submit" and result.error is None else None,
        )
        self._log.append(entry)
        return result

    @property
    def submission(self) -> Submission | None:
        return self._submission

    def log(self) -> list[dict]:
        """每次调用一条：名字、组、参数原文、耗时、是否出错、写了哪些 mask（面积）、返回的图（尺寸与 sha256）、
        沙箱代码原文与运行摘要、提交的答案。"""
        return json.loads(json.dumps(self._log))

    @property
    def masks(self) -> Mapping[str, np.ndarray]:
        """当前的命名 mask（只读视图；数组本身也只读）。"""
        return MappingProxyType(self._masks)

    @property
    def images(self) -> Mapping[str, Image.Image]:
        """图像注册表：id → PIL 图。"original" 是原图，之后每张返回给模型的图依次是 view1、view2……"""
        return MappingProxyType(self._images)

    @property
    def size(self) -> tuple[int, int]:
        """原图 (W, H)。"""
        return self._w, self._h

    # ---- 内部：参数、状态、显示 --------------------------------------------

    @staticmethod
    def _fresh_call_record() -> dict:
        return {"masks_written": {}, "images": [], "code": None, "sandbox": None}

    def _parse_args(self, tool: _Tool, arguments: Mapping[str, Any]) -> dict:
        unknown = sorted(set(arguments) - set(tool.properties))
        if unknown:
            raise ToolError(f"unexpected parameter(s) {unknown} for tool '{tool.name}'; accepted: {sorted(tool.properties)}")
        missing = [p for p in tool.required if p not in arguments]
        if missing:
            raise ToolError(f"missing required parameter(s) {missing} for tool '{tool.name}'")
        return {k: _parse_value(v, tool.properties[k], k, raw=k in tool.raw_params) for k, v in arguments.items()}

    def _norm_to_abs(self, coords) -> np.ndarray:
        return norm1000_to_abs(np.asarray(coords, dtype=np.float64), width=self._w, height=self._h)

    def _check_box(self, x1, y1, x2, y2) -> None:
        if not (x1 < x2 and y1 < y2):
            raise ToolError(f"box must satisfy x1 < x2 and y1 < y2; got ({x1:g}, {y1:g}, {x2:g}, {y2:g})")

    def _region_px(self, x1, y1, x2, y2) -> tuple[int, int, int, int]:
        """0–1000 区域 → 整数像素窗口 (c0, r0, c1, r1)，半开，向外取整，至少 1 px。"""
        self._check_box(x1, y1, x2, y2)
        ax1, ay1, ax2, ay2 = (round(float(v), 6) for v in self._norm_to_abs([x1, y1, x2, y2]))
        c0 = min(max(int(math.floor(ax1)), 0), self._w - 1)
        r0 = min(max(int(math.floor(ay1)), 0), self._h - 1)
        c1 = min(max(int(math.ceil(ax2)), c0 + 1), self._w)
        r1 = min(max(int(math.ceil(ay2)), r0 + 1), self._h)
        return c0, r0, c1, r1

    def _px_to_norm(self, c: float, r: float) -> tuple[float, float]:
        return c / self._w * NORM1000_SCALE, r / self._h * NORM1000_SCALE

    def _mask(self, name: str) -> np.ndarray:
        if name not in self._masks:
            have = ", ".join(sorted(self._masks)) or "none"
            raise ToolError(f"unknown mask '{name}' (existing masks: {have})")
        return self._masks[name]

    def _set_mask(self, name: str, m: np.ndarray) -> None:
        if name not in self._masks and len(self._masks) >= MAX_MASKS:
            raise ToolError(f"too many masks (limit {MAX_MASKS}); delete one with delete_mask first")
        a = np.array(m, dtype=bool)
        a.setflags(write=False)
        self._masks[name] = a
        self._cur["masks_written"][name] = int(a.sum())

    def _emit(self, arr: np.ndarray) -> Image.Image:
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        img = Image.fromarray(a)
        iid = f"view{len(self._images)}"
        self._images[iid] = img
        self._cur["images"].append(
            {"id": iid, "width": img.size[0], "height": img.size[1], "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        )
        return img

    def _mask_stats(self, name: str) -> str:
        m = self._masks[name]
        area = int(m.sum())
        pct = 100.0 * area / (self._h * self._w)
        head = f"Mask '{name}': {area} px ({pct:.2f}% of the {self._w}x{self._h} image)"
        if area == 0:
            return head + ", empty."
        rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        x0, y0 = self._px_to_norm(cols[0], rows[0])
        x1, y1 = self._px_to_norm(cols[-1] + 1, rows[-1] + 1)
        _, n = ndimage.label(m, structure=np.ones((3, 3), bool))
        return head + f", box [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}] (0-1000), {n} connected component{'s' if n != 1 else ''}."

    def _apply(self, name: str, shape: np.ndarray, mode: str, erase: np.ndarray | None = None) -> str:
        """把 shape 按 mode 写进 mask name（erase 最后从结果里擦掉）。返回说明文字。"""
        if mode == "new":
            res = shape.copy()
            verb = "Replaced" if name in self._masks else "Created"
        elif name not in self._masks:
            if mode != "union":
                raise ToolError(f"mask '{name}' does not exist; mode '{mode}' needs an existing mask")
            res, verb = shape.copy(), "Created"
        else:
            old = self._masks[name]
            res = {"union": old | shape, "subtract": old & ~shape, "intersect": old & shape}[mode]
            verb = {"union": "Added to", "subtract": "Subtracted from", "intersect": "Intersected"}[mode]
        if erase is not None:
            res &= ~erase
        self._set_mask(name, res)
        return f"{verb} mask '{name}'. " + self._mask_stats(name)

    def _crop_view(self, arr: np.ndarray, region) -> tuple[np.ndarray, str]:
        """按 0–1000 区域裁剪并缩放到 ZOOM_LONG_SIDE，返回 (视图, 覆盖范围说明)。"""
        c0, r0, c1, r1 = self._region_px(*region)
        view = _resize_long_side(arr[r0:r1, c0:c1], ZOOM_LONG_SIDE)
        vx0, vy0 = self._px_to_norm(c0, r0)
        vx1, vy1 = self._px_to_norm(c1, r1)
        vh, vw = view.shape[:2]
        text = (
            f"The view covers x = {_fmt(vx0)}..{_fmt(vx1)}, y = {_fmt(vy0)}..{_fmt(vy1)} (0-1000 full-image coordinates; "
            f"original pixel columns {c0}..{c1 - 1}, rows {r0}..{r1 - 1} of the {self._w}x{self._h} image), "
            f"shown at {vw}x{vh} px ({vw / (c1 - c0):.2f}x). A point at fraction (u, v) of the view (0-1 each, from the "
            f"top-left) is at x = {_fmt(vx0)} + u*{_fmt(vx1 - vx0)}, y = {_fmt(vy0)} + v*{_fmt(vy1 - vy0)} in full-image coordinates."
        )
        return view, text

    def _submission_summary(self) -> dict | None:
        s = self._submission
        if s is None:
            return None
        return {"kind": s.kind, "source": s.source, "area": int(s.mask.sum()) if s.mask is not None else None,
                "box": list(s.box) if s.box is not None else None, "count": s.count}

    def _guide_unit(self) -> np.ndarray:
        if self._guide is None:
            self._guide = guide_to_unit(self._rgb)
        return self._guide

    def _color_gradient(self) -> np.ndarray:
        if self._gradient is None:
            self._gradient = color_gradient(self._rgb)
        return self._gradient

    def _disks(self, pts_abs: np.ndarray, radius: float) -> np.ndarray:
        """以各点为圆心、半径 radius（px）的圆盘之并：像素中心到圆心的距离 ≤ radius，外加点所在的那个像素。"""
        out = np.zeros((self._h, self._w), bool)
        r = float(radius)
        for x, y in pts_abs:
            c, rr = min(int(math.floor(x)), self._w - 1), min(int(math.floor(y)), self._h - 1)
            out[rr, c] = True
            c0, c1 = max(int(math.floor(x - r)) - 1, 0), min(int(math.ceil(x + r)) + 1, self._w)
            r0, r1 = max(int(math.floor(y - r)) - 1, 0), min(int(math.ceil(y + r)) + 1, self._h)
            cy = np.arange(r0, r1)[:, None] + 0.5
            cx = np.arange(c0, c1)[None, :] + 0.5
            out[r0:r1, c0:c1] |= (cx - x) ** 2 + (cy - y) ** 2 <= r * r
        return out

    # ---- eye ----------------------------------------------------------------

    def _t_zoom(self, x1, y1, x2, y2) -> ToolResult:
        view, cover = self._crop_view(self._rgb, (x1, y1, x2, y2))
        return ToolResult(text="Zoomed view of the original image. " + cover, images=[self._emit(view)])

    def _t_read_pixels(self, points, radius=0) -> ToolResult:
        pts = self._norm_to_abs(np.asarray(points, dtype=np.float64).reshape(-1, 2).ravel()).reshape(-1, 2)
        lw = np.asarray(LUMA_WEIGHTS)
        lines = []
        for (xn, yn), (xa, ya) in zip(points, pts):
            c, r = min(int(math.floor(xa)), self._w - 1), min(int(math.floor(ya)), self._h - 1)
            if radius == 0:
                v = self._rgb[r, c].astype(np.int64)
                lines.append(f"({xn:g}, {yn:g}) -> pixel (col {c}, row {r}): RGB ({v[0]}, {v[1]}, {v[2]}), luminance {float(v @ lw):.1f}")
            else:
                win = self._rgb[max(r - radius, 0) : r + radius + 1, max(c - radius, 0) : c + radius + 1].reshape(-1, 3).astype(np.float64)
                mu, sd = win.mean(axis=0), win.std(axis=0)
                lines.append(
                    f"({xn:g}, {yn:g}) -> pixel (col {c}, row {r}), mean over {win.shape[0]} px: RGB ({mu[0]:.1f}, {mu[1]:.1f}, "
                    f"{mu[2]:.1f}), luminance {float(mu @ lw):.1f}, std RGB ({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f})"
                )
        return ToolResult(text="\n".join(lines))

    def _t_tile_view(self, rows, cols) -> ToolResult:
        img = np.array(self._rgb)
        h, w = self._h, self._w
        ry = [k * h // rows for k in range(rows + 1)]
        cx = [k * w // cols for k in range(cols + 1)]
        th = max(1, int(round(max(h, w) / 500)))
        for y in ry[1:-1]:
            cv2.line(img, (0, y), (w - 1, y), _BLACK, th + 2)
            cv2.line(img, (0, y), (w - 1, y), _WHITE, th)
        for x in cx[1:-1]:
            cv2.line(img, (x, 0), (x, h - 1), _BLACK, th + 2)
            cv2.line(img, (x, 0), (x, h - 1), _WHITE, th)
        scale = max(0.5, max(h, w) / 800)
        lines = [f"The {w}x{h} image is divided into {rows} x {cols} cells, numbered 1..{rows * cols} row by row from the top-left:"]
        for i in range(rows):
            for j in range(cols):
                k = i * cols + j + 1
                (tw, tht), _ = cv2.getTextSize(str(k), _FONT, scale, th)
                _put_label(img, str(k), (cx[j] + 3 * th + 1, ry[i] + tht + 3 * th + 1), scale, th)
                x0, y0 = self._px_to_norm(cx[j], ry[i])
                x1, y1 = self._px_to_norm(cx[j + 1], ry[i + 1])
                lines.append(f"cell {k} (row {i + 1}, col {j + 1}): x = {_fmt(x0)}..{_fmt(x1)}, y = {_fmt(y0)}..{_fmt(y1)}")
        return ToolResult(text="\n".join(lines), images=[self._emit(img)])

    # ---- hand ---------------------------------------------------------------

    def _t_draw_box(self, name, x1, y1, x2, y2, mode="new") -> ToolResult:
        self._check_box(x1, y1, x2, y2)
        m = box_to_mask(self._norm_to_abs([x1, y1, x2, y2]), self._h, self._w)
        return ToolResult(text=self._apply(name, m, mode))

    def _t_draw_points(self, name, points, labels=None, radius=POINT_RADIUS_DEFAULT, mode="new") -> ToolResult:
        lab = [1] * len(points) if labels is None else labels
        if len(lab) != len(points):
            raise ToolError(f"labels must have one entry per point ({len(points)}); got {len(lab)}")
        pts = self._norm_to_abs(np.asarray(points, dtype=np.float64).ravel()).reshape(-1, 2)
        pos = np.asarray(lab) == 1
        add = self._disks(pts[pos], radius) if pos.any() else np.zeros((self._h, self._w), bool)
        erase = self._disks(pts[~pos], radius) if (~pos).any() else None
        text = self._apply(name, add, mode, erase=erase)
        return ToolResult(text=f"{int(pos.sum())} positive, {int((~pos).sum())} negative point(s), radius {radius:g} px. " + text)

    def _poly_abs(self, points, min_points=3) -> np.ndarray:
        if len(points) < min_points:
            raise ToolError(f"need at least {min_points} points; got {len(points)}")
        return self._norm_to_abs(np.asarray(points, dtype=np.float64).ravel()).reshape(-1, 2)

    def _t_draw_polygon(self, name, points, mode="new") -> ToolResult:
        pts = self._poly_abs(points)
        m = polygons_to_mask([pts.ravel()], self._h, self._w)
        return ToolResult(text=self._apply(name, m, mode))

    def _t_draw_spline(self, name, points, mode="new") -> ToolResult:
        pts = self._poly_abs(points)
        keep = np.ones(len(pts), bool)  # 去掉与前一点重合的控制点（含首尾）
        keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
        pts = pts[keep]
        if len(pts) > 1 and np.all(pts[-1] == pts[0]):
            pts = pts[:-1]
        if len(pts) < 3:
            raise ToolError("need at least 3 distinct consecutive points for a closed spline")
        m = polygons_to_mask([spline_polygon(pts).ravel()], self._h, self._w)
        return ToolResult(text=self._apply(name, m, mode))

    def _t_combine_masks(self, name, other, mode="union") -> ToolResult:
        src = self._mask(other)
        return ToolResult(text=self._apply(name, src, mode))

    def _t_delete_mask(self, name) -> ToolResult:
        self._mask(name)
        del self._masks[name]
        rest = ", ".join(sorted(self._masks)) or "none"
        return ToolResult(text=f"Deleted mask '{name}'. Remaining masks: {rest}.")

    # ---- snap ---------------------------------------------------------------

    def _t_edges(self, low=CANNY_LOW, high=CANNY_HIGH, out=None) -> ToolResult:
        if not low < high:
            raise ToolError(f"low must be < high; got low={low:g}, high={high:g}")
        rgb = np.ascontiguousarray(self._rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(rgb, (CANNY_BLUR_KSIZE, CANNY_BLUR_KSIZE), CANNY_BLUR_SIGMA)
        e = cv2.Canny(blur, float(low), float(high), L2gradient=True) > 0  # 三通道输入：每个像素取梯度最大的通道
        vis = np.repeat(((gray.astype(np.uint16) * 2) // 5).astype(np.uint8)[..., None], 3, axis=2)
        vis[e] = _WHITE
        n = int(e.sum())
        text = (
            f"Canny edges on the color image (low={low:g}, high={high:g}, L2 gradient, after a {CANNY_BLUR_KSIZE}x{CANNY_BLUR_KSIZE} "
            f"Gaussian blur with sigma {CANNY_BLUR_SIGMA:g}), drawn in white over a darkened grayscale copy: {n} edge px "
            f"({100.0 * n / (self._h * self._w):.2f}% of the image)."
        )
        if out is not None:
            self._set_mask(out, e)
            text += f" Saved as mask '{out}'."
        return ToolResult(text=text, images=[self._emit(vis)])

    def _t_grabcut(self, out, box=None, mask=None, fg=None, bg=None, iterations=GRABCUT_ITERATIONS) -> ToolResult:
        if (box is None) == (mask is None):
            raise ToolError("give exactly one of 'box' or 'mask' to initialize GrabCut")
        kw: dict = {"iterations": iterations}
        if box is not None:
            self._check_box(*box)
            kw["box_abs"] = self._norm_to_abs(box)
            init = f"box [{', '.join(f'{v:g}' for v in box)}]"
        else:
            kw["init_mask"] = self._mask(mask)
            init = f"mask '{mask}'"
        if fg is not None:
            kw["fg"] = self._mask(fg)
        if bg is not None:
            kw["bg"] = self._mask(bg)
        try:
            res = grabcut_mask(self._rgb, **kw)
        except ValueError as e:
            raise ToolError(f"GrabCut cannot start: {e}") from None
        except cv2.error as e:
            raise ToolError(f"GrabCut failed (the foreground or background region may be too small): {str(e).strip()[:200]}") from None
        self._set_mask(out, res)
        seeds = "".join(f", {k} seeds from mask '{v}'" for k, v in (("foreground", fg), ("background", bg)) if v is not None)
        return ToolResult(text=f"GrabCut ({iterations} iterations) initialized from {init}{seeds}. " + self._mask_stats(out))

    def _t_watershed(self, out, fg, bg=None) -> ToolResult:
        f = self._mask(fg)
        if not f.any():
            raise ToolError(f"foreground seed mask '{fg}' is empty")
        if bg is not None:
            b = self._mask(bg)
            if (f & b).any():
                raise ToolError(f"seed masks '{fg}' and '{bg}' overlap")
            bdesc = f"mask '{bg}'"
        else:
            b = np.zeros_like(f)
            b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
            b &= ~f
            bdesc = "the image border"
        if not b.any():
            raise ToolError("background seeds are empty")
        markers = np.zeros((self._h, self._w), np.int32)
        markers[b] = 1
        markers[f] = 2
        res = sk_watershed(self._color_gradient(), markers) == 2
        self._set_mask(out, res)
        return ToolResult(
            text=f"Watershed on the color gradient, foreground seeds from mask '{fg}', background seeds from {bdesc}. " + self._mask_stats(out)
        )

    def _t_superpixels(self, n_segments=SLIC_N_SEGMENTS, compactness=SLIC_COMPACTNESS) -> ToolResult:
        lab = slic(
            np.asarray(self._rgb), n_segments=n_segments, compactness=compactness, sigma=SLIC_SIGMA, start_label=1, channel_axis=-1
        )
        _, inv = np.unique(lab, return_inverse=True)  # 编号重排为连续的 1..K，保持原来的大小顺序
        lab = (inv.reshape(lab.shape) + 1).astype(np.int32)
        self._superpixels = lab
        k = int(lab.max())
        bnd = find_boundaries(lab, mode="inner")
        vis = np.array(self._rgb)
        vis[bnd] = _YELLOW
        dist = ndimage.distance_transform_edt(np.pad(~bnd, 1, constant_values=False))[1:-1, 1:-1]
        anchors = ndimage.maximum_position(dist, lab, index=np.arange(1, k + 1))
        scale = max(0.3, max(self._h, self._w) / 1800)
        for i, (r, c) in enumerate(anchors, start=1):
            (tw, th), _ = cv2.getTextSize(str(i), _FONT, scale, 1)
            _put_label(vis, str(i), (int(c) - tw // 2, int(r) + th // 2), scale, 1)
        return ToolResult(
            text=(
                f"SLIC superpixels (n_segments={n_segments}, compactness={compactness:g}, sigma {SLIC_SIGMA:g}, CIELAB color): "
                f"{k} superpixels with ids 1..{k}, boundaries in yellow and ids drawn at their centers. "
                "select_superpixels refers to these ids until superpixels is called again."
            ),
            images=[self._emit(vis)],
        )

    def _t_select_superpixels(self, name, ids=None, points=None, mode="new") -> ToolResult:
        if self._superpixels is None:
            raise ToolError("no superpixels yet; call superpixels first")
        if not ids and not points:
            raise ToolError("give 'ids' and/or 'points'")
        k = int(self._superpixels.max())
        sel = set()
        for i in ids or []:
            if not 1 <= i <= k:
                raise ToolError(f"superpixel id {i} is out of range 1..{k}")
            sel.add(int(i))
        if points:
            pts = self._norm_to_abs(np.asarray(points, dtype=np.float64).ravel()).reshape(-1, 2)
            for x, y in pts:
                sel.add(int(self._superpixels[min(int(math.floor(y)), self._h - 1), min(int(math.floor(x)), self._w - 1)]))
        chosen = sorted(sel)
        m = np.isin(self._superpixels, chosen)
        return ToolResult(text=f"Selected superpixels {chosen}. " + self._apply(name, m, mode))

    def _t_refine(self, name, out=None, radius=REFINE_RADIUS, eps=REFINE_EPS) -> ToolResult:
        src = self._mask(name)
        q = guided_filter(self._guide_unit(), src.astype(np.float64), radius, eps)
        res = q >= REFINE_THRESHOLD
        dst = out or name
        changed = int((res != src).sum())
        self._set_mask(dst, res)
        return ToolResult(
            text=(
                f"Refined mask '{name}' with a guided filter (radius {radius} px, eps {eps:g}, guided by the original RGB image), "
                f"thresholded at {REFINE_THRESHOLD:g}; {changed} px changed. " + self._mask_stats(dst)
            )
        )

    def _t_morph(self, name, op, radius=MORPH_RADIUS_DEFAULT, out=None) -> ToolResult:
        src = self._mask(name)
        if op == "fill_holes":
            res = ndimage.binary_fill_holes(src)
            how = "fill_holes"
        else:
            k = disk_footprint(radius)
            m8 = src.astype(np.uint8)
            fn = {
                "dilate": lambda: cv2.dilate(m8, k),
                "erode": lambda: cv2.erode(m8, k),
                "open": lambda: cv2.morphologyEx(m8, cv2.MORPH_OPEN, k),
                "close": lambda: cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k),
            }[op]
            res = fn() > 0
            how = f"{op} with a disc of radius {radius} px"
        dst = out or name
        self._set_mask(dst, res)
        return ToolResult(text=f"Applied {how} to mask '{name}'. " + self._mask_stats(dst))

    def _t_components(self, name, keep=None, out=None) -> ToolResult:
        src = self._mask(name)
        lab, n = ndimage.label(src, structure=np.ones((3, 3), bool))
        if n == 0:
            if keep:
                raise ToolError(f"mask '{name}' is empty; nothing to keep")
            return ToolResult(text=f"Mask '{name}' is empty: 0 components.")
        area = np.bincount(lab.ravel(), minlength=n + 1)
        order = sorted(range(1, n + 1), key=lambda i: (-int(area[i]), i))  # 面积从大到小，同面积按扫描顺序
        objs = ndimage.find_objects(lab)
        lines = [f"Mask '{name}' has {n} 8-connected component{'s' if n != 1 else ''} (largest first):"]
        for rank, i in enumerate(order[:COMPONENTS_LIST_MAX], start=1):
            sr, sc = objs[i - 1]
            x0, y0 = self._px_to_norm(sc.start, sr.start)
            x1, y1 = self._px_to_norm(sc.stop, sr.stop)
            cy, cx = ndimage.center_of_mass(lab == i)
            mx, my = self._px_to_norm(cx + 0.5, cy + 0.5)
            lines.append(f"#{rank}: {int(area[i])} px, box [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}], center ({mx:.0f}, {my:.0f})")
        if n > COMPONENTS_LIST_MAX:
            lines.append(f"... and {n - COMPONENTS_LIST_MAX} smaller component(s).")
        if keep:
            bad = [r for r in keep if not 1 <= r <= n]
            if bad:
                raise ToolError(f"keep ranks {bad} are out of range 1..{n}")
            ranks = sorted(set(keep))
            dst = out or name
            self._set_mask(dst, np.isin(lab, [order[r - 1] for r in ranks]))
            lines.append(f"Kept component(s) {ranks} -> " + self._mask_stats(dst))
        return ToolResult(text="\n".join(lines))

    # ---- check --------------------------------------------------------------

    def _t_show(self, name, region=None) -> ToolResult:
        ov = render_overlay(np.asarray(self._rgb), [self._mask(name)])
        alpha = OVERLAY_ALPHA_NUM / OVERLAY_ALPHA_DEN
        text = f"Mask '{name}' overlaid on the original image (fill alpha {alpha:g}, outline {OVERLAY_CONTOUR_WIDTH} px). "
        if region is not None:
            ov, cover = self._crop_view(ov, region)
            text += cover + " "
        return ToolResult(text=text + self._mask_stats(name), images=[self._emit(ov)])

    # ---- code ---------------------------------------------------------------

    def _t_run_python(self, code, out=CODE_RESULT_DEFAULT) -> ToolResult:
        res = run_code(code, image=self._rgb, masks=dict(self._masks), config=self._sandbox)
        self._cur["sandbox"] = res.summary()
        so = res.stdout + ("\n[output truncated]" if res.stdout_truncated else "")
        if not res.ok:
            msg = f"{res.status}: {res.error}"
            raise ToolError(msg + (f"\nstdout before the error:\n{so}" if res.stdout else ""))
        text = "stdout:\n" + (so if res.stdout else "(no output)")
        if res.result_mask is not None:
            self._set_mask(out, res.result_mask)
            text += f"\nresult_mask saved. " + self._mask_stats(out)
        return ToolResult(text=text)

    # ---- submit -------------------------------------------------------------

    def _t_submit_mask(self, name) -> ToolResult:
        m = self._mask(name)
        self._submission = Submission(kind="mask", mask=np.array(m, dtype=bool), source=name)
        area = int(m.sum())
        return ToolResult(text=f"Submitted mask '{name}' ({area} px{', empty' if area == 0 else ''}) as the final answer.")

    def _t_submit_box(self, x1, y1, x2, y2) -> ToolResult:
        self._check_box(x1, y1, x2, y2)
        box = tuple(float(v) for v in self._norm_to_abs([x1, y1, x2, y2]))
        self._submission = Submission(kind="box", box=box)  # type: ignore[arg-type]
        return ToolResult(text=f"Submitted box [{x1:g}, {y1:g}, {x2:g}, {y2:g}] as the final answer.")

    def _t_submit_count(self, n) -> ToolResult:
        self._submission = Submission(kind="count", count=int(n))
        return ToolResult(text=f"Submitted count {n} as the final answer.")

    # ---- 工具表 -------------------------------------------------------------

    def _build_tools(self) -> list[_Tool]:
        name_new = _p_name("Name of the mask to write (created if it does not exist).")
        name_in = _p_name("Name of an existing mask.")
        out_opt = _p_name("Name of the mask to write the result to (default: overwrite the input mask).")
        T = _Tool
        tools = [
            # eye
            T("zoom", "eye", f"Return an enlarged view of a rectangular region of the original image (long side {ZOOM_LONG_SIDE} px); the text gives the exact region covered.",
              dict(_BOX_PARAMS), ("x1", "y1", "x2", "y2"), self._t_zoom),
            T("read_pixels", "eye", "Read the RGB values and luminance of the original image at given points, optionally averaged over a square window.",
              {"points": _p_points("Points [[x, y], ...] in 0-1000 full-image coordinates."),
               "radius": {"type": "integer", "minimum": 0, "maximum": READ_PIXELS_MAX_RADIUS, "default": 0,
                          "description": "Half-size of the square window in pixels (0 = the single pixel)."}},
              ("points",), self._t_read_pixels),
            T("tile_view", "eye", "Return the original image divided into a grid of numbered cells; the text lists each cell's coordinate range.",
              {"rows": {"type": "integer", "minimum": 1, "maximum": TILE_MAX, "description": "Number of rows."},
               "cols": {"type": "integer", "minimum": 1, "maximum": TILE_MAX, "description": "Number of columns."}},
              ("rows", "cols"), self._t_tile_view),
            # hand
            T("draw_box", "hand", "Rasterize an axis-aligned box into a named mask.",
              {"name": name_new, **_BOX_PARAMS, "mode": _p_mode()}, ("name", "x1", "y1", "x2", "y2"), self._t_draw_box),
            T("draw_points", "hand", "Paint discs at points into a named mask: positive points (label 1) add discs, negative points (label 0) erase discs.",
              {"name": name_new, "points": _p_points("Points [[x, y], ...] in 0-1000 full-image coordinates."),
               "labels": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}, "maxItems": MAX_POINTS,
                          "description": "One label per point: 1 = positive, 0 = negative (default: all 1)."},
               "radius": {"type": "number", "minimum": 0, "maximum": POINT_RADIUS_MAX, "default": POINT_RADIUS_DEFAULT,
                          "description": "Disc radius in pixels of the original image."},
               "mode": _p_mode()},
              ("name", "points"), self._t_draw_points),
            T("draw_polygon", "hand", "Rasterize a polygon (vertices in order) into a named mask.",
              {"name": name_new, "points": _p_points("Vertices [[x, y], ...] in 0-1000 full-image coordinates, at least 3.", 3),
               "mode": _p_mode()},
              ("name", "points"), self._t_draw_polygon),
            T("draw_spline", "hand", "Rasterize a closed smooth curve through the given points (centripetal Catmull-Rom spline) into a named mask.",
              {"name": name_new, "points": _p_points("Points on the curve [[x, y], ...] in order, 0-1000 full-image coordinates, at least 3.", 3),
               "mode": _p_mode()},
              ("name", "points"), self._t_draw_spline),
            T("combine_masks", "hand", "Combine mask 'other' into mask 'name' (new = copy, union, subtract or intersect).",
              {"name": name_new, "other": name_in, **{"mode": {**_p_mode(), "default": "union"}}},
              ("name", "other"), self._t_combine_masks),
            T("delete_mask", "hand", "Delete a named mask.", {"name": name_in}, ("name",), self._t_delete_mask),
            # snap
            T("edges", "snap", "Detect edges in the original image (Canny); returns the edge map over a darkened copy of the image, optionally saved as a mask.",
              {"low": {"type": "number", "minimum": 0, "maximum": CANNY_THRESHOLD_MAX, "default": CANNY_LOW, "description": "Lower hysteresis threshold."},
               "high": {"type": "number", "minimum": 0, "maximum": CANNY_THRESHOLD_MAX, "default": CANNY_HIGH, "description": "Upper hysteresis threshold."},
               "out": _p_name("Optional name of a mask to save the edge pixels to.")},
              (), self._t_edges),
            T("grabcut", "snap",
              "Segment with GrabCut (color models plus graph cuts), initialized from a box (outside = background) or from an existing mask "
              f"(inside = probable foreground; outside it but within its bounding box enlarged by {GRABCUT_MASK_MARGIN:.0%} per side = probable "
              "background; elsewhere = background); optional seed masks fix pixels as foreground or background.",
              {"out": _p_name("Name of the mask to write."),
               "box": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4,
                       "description": "Initial box [x1, y1, x2, y2] in 0-1000 full-image coordinates."},
               "mask": _p_name("Name of an existing mask to initialize from (instead of a box)."),
               "fg": _p_name("Optional mask of pixels that must be foreground."),
               "bg": _p_name("Optional mask of pixels that must be background."),
               "iterations": {"type": "integer", "minimum": 1, "maximum": GRABCUT_MAX_ITERATIONS, "default": GRABCUT_ITERATIONS,
                              "description": "Number of iterations."}},
              ("out",), self._t_grabcut),
            T("watershed", "snap", "Marker-based watershed on the color gradient of the original image: grows foreground seeds against background seeds and writes the foreground region.",
              {"out": _p_name("Name of the mask to write."),
               "fg": _p_name("Mask of foreground seed pixels."),
               "bg": _p_name("Optional mask of background seed pixels (default: the image border).")},
              ("out", "fg"), self._t_watershed),
            T("superpixels", "snap", "Partition the original image into superpixels (SLIC); returns the image with superpixel boundaries and numeric ids.",
              {"n_segments": {"type": "integer", "minimum": 2, "maximum": SLIC_MAX_SEGMENTS, "default": SLIC_N_SEGMENTS,
                              "description": "Approximate number of superpixels."},
               "compactness": {"type": "number", "minimum": 0.01, "maximum": 100, "default": SLIC_COMPACTNESS,
                               "description": "Higher = more regular shapes, lower = follows color more closely."}},
              (), self._t_superpixels),
            T("select_superpixels", "snap", "Write the union of chosen superpixels (from the latest superpixels call) into a named mask, chosen by id and/or by points inside them.",
              {"name": name_new,
               "ids": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": SLIC_MAX_SEGMENTS, "description": "Superpixel ids."},
               "points": _p_points("Points [[x, y], ...] in 0-1000 full-image coordinates; the superpixel containing each point is selected."),
               "mode": _p_mode()},
              ("name",), self._t_select_superpixels),
            T("refine", "snap", "Snap a mask's boundary to image edges: guided filter of the mask with the original color image as guide, then threshold.",
              {"name": name_in, "out": out_opt,
               "radius": {"type": "integer", "minimum": 1, "maximum": REFINE_MAX_RADIUS, "default": REFINE_RADIUS, "description": "Filter radius in pixels."},
               "eps": {"type": "number", "minimum": 1e-6, "maximum": 1.0, "default": REFINE_EPS,
                       "description": "Regularization; smaller follows edges more closely."}},
              ("name",), self._t_refine),
            T("morph", "snap", "Morphological operation on a mask with a disc: dilate, erode, open, close, or fill_holes.",
              {"name": name_in, "op": {"type": "string", "enum": list(MORPH_OPS), "description": "Operation."},
               "radius": {"type": "integer", "minimum": 1, "maximum": MORPH_RADIUS_MAX, "default": MORPH_RADIUS_DEFAULT,
                          "description": "Disc radius in pixels (ignored by fill_holes)."},
               "out": out_opt},
              ("name", "op"), self._t_morph),
            T("components", "snap", "List the 8-connected components of a mask (largest first) with area, box and center; optionally keep only some of them.",
              {"name": name_in,
               "keep": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": MAX_POINTS,
                        "description": "Ranks (1 = largest) of the components to keep."},
               "out": out_opt},
              ("name",), self._t_components),
            # check
            T("show", "check", "Return the original image with a named mask overlaid (fill and outline), optionally zoomed to a region, with the mask's area, box and number of components.",
              {"name": name_in, "region": dict(_REGION)}, ("name",), self._t_show),
            # code
            T("run_python", "code",
              "Run Python code in a sandbox with numpy, scipy, cv2, skimage, math, itertools, collections, functools, json, re and statistics "
              "(no files, network or learned models). Variables: image (H x W x 3 uint8 RGB array of the original image), masks (dict: name -> "
              "H x W bool array), H, W. Assign an H x W boolean array to result_mask to save it as a mask. Printed output is returned "
              f"(first {STDOUT_LIMIT_CHARS} characters). Limits: {self._sandbox.timeout_s:g} s, {self._sandbox.memory_bytes / 1024**3:g} GB.",
              {"code": {"type": "string", "description": "Python source code."},
               "out": {**_p_name("Name of the mask to save result_mask to."), "default": CODE_RESULT_DEFAULT}},
              ("code",), self._t_run_python, raw_params=frozenset({"code"})),
            # submit
            T("submit_mask", "submit", "Submit a named mask as the final answer.", {"name": name_in}, ("name",), self._t_submit_mask),
            T("submit_box", "submit", "Submit a box as the final answer.", dict(_BOX_PARAMS), ("x1", "y1", "x2", "y2"), self._t_submit_box),
            T("submit_count", "submit", "Submit a non-negative integer as the final answer.",
              {"n": {"type": "integer", "minimum": 0, "maximum": COUNT_MAX, "description": "The count."}}, ("n",), self._t_submit_count),
        ]
        order = {g: i for i, g in enumerate(GROUPS)}
        assert all(t.group in order for t in tools)
        return sorted(tools, key=lambda t: order[t.group])  # sorted 是稳定的：组内保持定义顺序
