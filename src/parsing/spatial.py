"""坐标类输出的解析：0–1000 的 JSON 多边形，以及 Qwen/Gemma 共用的 JSON 框规则（DEV-0006）。

这里只有与模型无关的规则。框的键名、轴序由各家族适配器传入（src/adapters/qwen35.py、gemma4.py；简报 §4 第 6 条）。
数值规则（JSON 格式共用）：坐标必须是 JSON 数（整数或小数；true/false 不算），并且是有限值，否则 malformed；
小于 0 或大于 1000（constants.NORM1000_SCALE）→ out_of_range。检查顺序是先查全部结构与类型，再查全部量程，
最后查颠倒（框）或点数（多边形），所以同一条输出里有几种毛病时，报的是排在前面的那种。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Sequence

import numpy as np

from src.constants import NORM1000_SCALE
from src.executors.geometry import check_box, norm1000_to_abs
from src.parsing.core import BoxParse, ParseFailure, ParseResult
from src.parsing.json_extract import extract_json

AxisOrder = Literal["xy", "yx"]
AXIS_ORDERS: tuple[str, ...] = ("xy", "yx")
MIN_POLYGON_POINTS = 3  # 任务书：多边形至少 3 个点（与 executors.raster 的要求相同）


def _check_wh(width: float, height: float) -> None:
    if not (width > 0 and height > 0):
        raise ValueError(f"原图宽高必须为正：W={width}, H={height}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numbers(v: Any, k: int) -> list[float] | None:
    """v 是恰好 k 个有限 JSON 数的列表 → 数值列表；否则 None（调用方判 malformed）。"""
    if not isinstance(v, list) or len(v) != k or not all(_is_number(x) for x in v):
        return None
    if not all(math.isfinite(x) for x in v):  # 例如 1e999 被 json 读成 inf
        return None
    return [float(x) for x in v]


def _in_range(values: Sequence[float]) -> bool:
    return all(0.0 <= x <= NORM1000_SCALE for x in values)


# ---------------------------------------------------------------------------
# 多边形
# ---------------------------------------------------------------------------


def _is_scalar(v: Any) -> bool:
    return not isinstance(v, (list, dict))


def parse_polygons(text: str, *, width: float, height: float, axis_order: AxisOrder = "xy") -> ParseResult:
    """0–1000 坐标的 JSON 多边形 → 绝对像素的多边形。成功时 value 是多边形的元组，每个多边形是 (x, y) 浮点对的元组，
    可直接交给 executors.raster.polygons_to_mask 与 geometry.polygon_iou。

    没有哪个家族有原生的多边形格式；E1 用统一的提示要求 0–1000 坐标的 JSON。规则：
    - 经 extract_json 取值（代码块与裸 JSON 的规则见那里）；没有 JSON → no_polygon，ambiguous / malformed 原样给出；
    - 只接受两种形状：`[[a, b], …]` 是单个多边形；`[[[a, b], …], …]` 是多个多边形。空列表 [] → no_polygon；
      其他形状（扁平的 [x0, y0, x1, …]、带键的对象、层数混杂）→ malformed；
    - 每个点必须恰好是两个有限的 JSON 数，否则 malformed；任一坐标 < 0 或 > 1000 → out_of_range；
      任一多边形少于 3 个点 → too_few_points。任何一个点坏了整条判失败，不返回其余的多边形；
    - axis_order="xy"：点是 [x, y]；"yx"：点是 [y, x]，先交换再换算。换算用 geometry.norm1000_to_abs
      （x = a / 1000 · W，y = b / 1000 · H；浮点，不取整）。width、height 是原图的宽高。
    不查自交等几何合法性：光栅化（pycocotools）照样能处理，是否合法由用到它的指标决定。
    """
    if axis_order not in AXIS_ORDERS:
        raise ValueError(f"axis_order 必须是 {AXIS_ORDERS} 之一，收到 {axis_order!r}")
    _check_wh(width, height)
    got = extract_json(text)
    if got.failure is ParseFailure.NO_JSON:
        return ParseResult(None, ParseFailure.NO_POLYGON)
    if got.failure is not None:
        return got
    v = got.value
    if not isinstance(v, list):
        return ParseResult(None, ParseFailure.MALFORMED)
    if not v:
        return ParseResult(None, ParseFailure.NO_POLYGON)
    if all(isinstance(p, list) and all(_is_scalar(x) for x in p) for p in v):
        raw_polys = [v]  # [[a, b], …]
    elif all(isinstance(poly, list) and all(isinstance(p, list) for p in poly) for poly in v):
        raw_polys = v  # [[[a, b], …], …]
    else:
        return ParseResult(None, ParseFailure.MALFORMED)
    polys: list[list[list[float]]] = []
    for raw in raw_polys:
        pts = [_numbers(p, 2) for p in raw]
        if any(p is None for p in pts):
            return ParseResult(None, ParseFailure.MALFORMED)
        polys.append(pts)
    if not all(_in_range(p) for poly in polys for p in poly):
        return ParseResult(None, ParseFailure.OUT_OF_RANGE)
    if any(len(poly) < MIN_POLYGON_POINTS for poly in polys):
        return ParseResult(None, ParseFailure.TOO_FEW_POINTS)
    out = []
    for poly in polys:
        arr = np.asarray(poly, dtype=np.float64)
        if axis_order == "yx":
            arr = arr[:, ::-1]
        xy = norm1000_to_abs(arr, width=width, height=height)
        out.append(tuple((float(x), float(y)) for x, y in xy))
    return ParseResult(tuple(out), None)


# ---------------------------------------------------------------------------
# JSON 框（Qwen 的 bbox_2d、Gemma 的 box_2d 共用这一套规则；键名与轴序由适配器给）
# ---------------------------------------------------------------------------

BOX_LABEL_KEY = "label"


def parse_json_boxes(
    text: str,
    *,
    key: str,
    to_xyxy: Callable[..., Any],
    width: float,
    height: float,
) -> BoxParse:
    """JSON 形式的原生框 → 内部绝对像素 xyxy。key 是本家族的坐标键（如 "bbox_2d"），to_xyxy(quad, width=, height=) 把
    本家族的原生四元组（0–1000，轴序由家族决定）换成绝对像素 xyxy。二者都由适配器给出。

    规则（不补齐、不截断、不猜）：
    - 经 extract_json 取值（代码块与裸 JSON 的规则见那里）；没有 JSON → no_box，ambiguous / malformed 原样给出；
    - 接受的形状：单个对象 {key: [4 个数], "label": …}；对象的列表；裸的四元组 [a, b, c, d]；四元组的列表。
      对象里 "label" 可缺（或为 null），记 None；有就必须是字符串；其他键忽略。对象与四元组混在一个列表里、对象缺 key、
      key 的值不是恰好 4 个有限 JSON 数、label 不是字符串、其他形状 → malformed；
    - 空列表 [] → 成功，零个框（模型明确回答"没有"；只要一个框时 one_box 判 no_box）；
    - 数值是有限的 JSON 数，整数或小数都接受（JSON 本身允许小数）；任一坐标 < 0 或 > 1000 → out_of_range；
    - 任一框颠倒 → inverted：原生四元组的第 3 个数小于第 1 个、或第 4 个数小于第 2 个。两个家族的轴序不同，但都是
      (轴1 起点, 轴2 起点, 轴1 终点, 轴2 终点)，所以这一条在原生四元组上判，与轴序无关；
    - 任何一个框坏了整条判失败，不返回其余的框。退化框（起点 = 终点）照样返回。
    """
    if not isinstance(text, str):
        raise TypeError(f"text 必须是 str，收到 {type(text).__name__}")
    _check_wh(width, height)
    got = extract_json(text)
    if got.failure is ParseFailure.NO_JSON:
        return BoxParse(None, None, ParseFailure.NO_BOX)
    if got.failure is not None:
        return BoxParse(None, None, got.failure)
    v = got.value
    if isinstance(v, dict):
        items: list[Any] = [v]
    elif isinstance(v, list) and v and all(_is_scalar(x) for x in v):
        items = [v]  # 裸的四元组
    elif isinstance(v, list) and (all(isinstance(x, dict) for x in v) or all(isinstance(x, list) for x in v)):
        items = v  # 对象的列表、四元组的列表，或空列表
    else:
        return BoxParse(None, None, ParseFailure.MALFORMED)
    quads: list[list[float]] = []
    labels: list[str | None] = []
    for item in items:
        if isinstance(item, dict):
            if key not in item:
                return BoxParse(None, None, ParseFailure.MALFORMED)
            label = item.get(BOX_LABEL_KEY)
            if label is not None and not isinstance(label, str):
                return BoxParse(None, None, ParseFailure.MALFORMED)
            quad = _numbers(item[key], 4)
        else:
            label = None
            quad = _numbers(item, 4)
        if quad is None:
            return BoxParse(None, None, ParseFailure.MALFORMED)
        quads.append(quad)
        labels.append(label)
    if not all(_in_range(q) for q in quads):
        return BoxParse(None, None, ParseFailure.OUT_OF_RANGE)
    if any(q[2] < q[0] or q[3] < q[1] for q in quads):
        return BoxParse(None, None, ParseFailure.INVERTED)
    boxes = tuple(check_box(to_xyxy(q, width=width, height=height)) for q in quads)
    return BoxParse(boxes, tuple(labels), None)
