"""探索 X-2 的两个新任务（DEV-0069；预注册 X）：描 B 样条、指修正框。提示原文、输出格式、解析与校验都是常量与纯函数，训练
（src/w2/w2c_train.py 的 spline / correct 流）与推理（src/explore/x2_runner.py）共用这一份，两边逐字相同。

**坐标**：都是 0–1000 网格的整数，x 在前（同 E1 与 src/explore/oracle_sim）：
- 原图像素 → 网格：oracle_sim.quantize_xy / quantize_box（src/e3/c3.norm1000：floor(x / W · 1000 + 0.5)，夹到 [0, 1000]）；
- 网格 → 原图像素：oracle_sim.dequantize_xy / dequantize_box（src/executors/geometry.norm1000_to_abs：x = q / 1000 · W，与适配器
  src/adapters/qwen35.native_box_to_xyxy 同一个函数）。
- 提出的框来自 E1 refexp_box 的原生输出（本来就是网格整数按 norm1000_to_abs 换算的），所以 box_q(提出的框) 就是模型写的那四个整数
  （tests/test_x2_prompts.py 钉住往返）。

**描样条**（spline_prompt；n 是参数，缺省 SPLINE_N_DEFAULT = 16）：
- 输入：原图 + 提示（指代句、提出的框的四个网格整数）；
- 目标与输出：一条闭合周期均匀三次 B 样条的 n 个控制点，JSON `[[x, y], …]`（spline_text：oracle_sim.spline_text 的单条外环写法，
  json.dumps 的分隔符 (", ", ": ")，同 E1 的统一多边形格式）；
- 解析（parse_spline，严格）：整个回答（去掉首尾空白）必须是一个 JSON 值，是恰好 n 个 [x, y] 的列表，坐标是 JSON 整数（不收小数、
  布尔、字符串）且在 [0, 1000] 内；否则按 X2Failure 分类记失败（先结构、再类型、再范围、最后点数）。
- 光栅化：spline_mask（guidance.rasterize_loops，m 由提出的框定，采样点逐坐标夹到提出的框内，偶奇填充），与 X-1 的 B 样条通道同一套。

**指修正框**（correct_prompt；m 是参数，缺省 CORRECT_M_DEFAULT = 2）：
- 输入：一张叠加图（src/explore/x2_render.render_overlay：原图上画当前 mask 的半透明填充与轮廓，再画提出的框）+ 提示（指代句、提出的框）；
- 目标与输出：至多 m 个带符号的框，JSON `[{"sign": "+", "box": [x1, y1, x2, y2]}, …]`（corrections_text：oracle_sim.corrections_text）；
  "+" = 漏掉、要补，"-" = 多出、要删；没有要改的写 `[]`；
- 解析（parse_corrections，严格）：整个回答必须是一个 JSON 列表，至多 m 项；每项恰有 sign 与 box 两个键；sign 恰是 "+" 或 "-"
  （ASCII；Unicode 的减号也算 bad_sign）；box 是 4 个 JSON 整数、在 [0, 1000] 内、x2 ≥ x1 且 y2 ≥ y1（同 E1 的 parse_boxes：面积为 0
  的框不算失败，施加时被约束框或 effect gate 去掉）；`[]` 是成功、零个框。

两个提示都关思考（同 W2-c 的提出与挑选）；生成上限见 max_new_tokens_spline / max_new_tokens_correct（只在推理时用，截断记失败，
同 E1 的 parse_generation）。

只 import 不含学习的模块（numpy 与本仓库的几何、栅格函数；tests/test_x2_prompts.py 在干净的子进程里核对）。
"""

from __future__ import annotations

import enum
import json
from typing import Any, NamedTuple, Sequence

import numpy as np

from src.constants import NORM1000_SCALE
from src.explore import guidance as G
from src.explore import oracle_sim as OS

PROMPT_VERSION = "x2-prompts-v1"
GRID_MAX = int(NORM1000_SCALE)  # 1000
SPLINE_N_DEFAULT = 16
SPLINE_N_MIN = 4  # 三次闭合样条至少 4 个控制点（guidance.eval_closed_bspline）
CORRECT_M_DEFAULT = 2
SIGNS: tuple[str, ...] = (G.SIGN_ADD, G.SIGN_REMOVE)  # "+"、"-"（ASCII）
CORRECTION_KEYS = ("sign", "box")
JSON_SEPARATORS = OS.JSON_SEPARATORS  # (", ", ": ")

SPLINE_TEMPLATE = (
    "The object {expression} lies inside the box [{x1}, {y1}, {x2}, {y2}]. Trace its outer contour with a closed periodic "
    "uniform cubic B-spline of exactly {n} control points. Coordinates are integers on a 0-1000 grid of the image, x first. "
    "Output only JSON: [[x, y], ...] with exactly {n} points."
)
CORRECT_TEMPLATE = (
    "The image shows the current mask for {expression} as a blue overlay with a yellow outline, and the box "
    "[{x1}, {y1}, {x2}, {y2}] that contains the object as a red rectangle. Point out at most {m} local regions where the mask "
    "needs fixing, each with a sign and a box: \"+\" for a missed part that should be added, \"-\" for an extra part that should "
    "be removed. If nothing needs fixing, output []. Coordinates are integers on a 0-1000 grid of the image, x first. "
    "Output only JSON: [{{\"sign\": \"+\", \"box\": [x1, y1, x2, y2]}}, ...]."
)

# 推理时的生成上限：基数 + 每单位（n 个控制点 / m 个框）。X-1 的实数：12 个控制点约 119 个 token、24 个约 237 个（约 10 个 / 点），
# 每轮 2 个修正框约 56 个；上限给到约 2 倍。
SPLINE_TOKENS_BASE = 64
SPLINE_TOKENS_PER_POINT = 20
CORRECT_TOKENS_BASE = 32
CORRECT_TOKENS_PER_BOX = 48


class X2Failure(str, enum.Enum):
    """解析失败的分类（parse_generation 另给 truncated、empty、thinking_unclosed，见 src/parsing/core.ParseFailure）。"""

    EMPTY = "empty"  # 去掉首尾空白后为空
    MALFORMED_JSON = "malformed_json"  # 整个回答不是一个 JSON 值（含代码块、前后有别的字、多个 JSON 值）
    NOT_LIST = "not_list"  # 顶层不是列表
    BAD_POINT = "bad_point"  # 样条：某一项不是两个元素的列表
    BAD_ITEM = "bad_item"  # 修正：某一项不是恰有 sign、box 两个键的对象，或 box 不是 4 个元素的列表
    BAD_SIGN = "bad_sign"  # 修正：sign 不是 "+" 或 "-"
    NON_INTEGER = "non_integer"  # 坐标不是 JSON 整数（小数、布尔、字符串……）
    OUT_OF_RANGE = "out_of_range"  # 坐标 < 0 或 > 1000
    INVERTED = "inverted"  # 修正：x2 < x1 或 y2 < y1
    WRONG_COUNT = "wrong_count"  # 样条：点数不是 n
    TOO_MANY = "too_many"  # 修正：多于 m 个框


class X2Parse(NamedTuple):
    """解析结果：成功时 failure 为 None；失败时 value 为 None（parse_generation 要求的形状）。"""

    value: Any
    failure: Any


def failure_name(f: Any) -> str | None:
    """失败原因 → 字符串（X2Failure 与 src.parsing.ParseFailure 都是 str 枚举）。"""
    if f is None:
        return None
    return str(f.value) if isinstance(f, enum.Enum) else str(f)


def _posint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_n(n: Any) -> int:
    if not _posint(n) or n < SPLINE_N_MIN:
        raise ValueError(f"控制点数 n 必须是 ≥ {SPLINE_N_MIN} 的整数，收到 {n!r}")
    return int(n)


def check_m(m: Any) -> int:
    if not _posint(m):
        raise ValueError(f"修正框数 m 必须是正整数，收到 {m!r}")
    return int(m)


def check_box_q(box_q: Sequence[Any]) -> tuple[int, int, int, int]:
    q = tuple(box_q)
    if len(q) != 4 or not all(_is_int(v) and 0 <= v <= GRID_MAX for v in q) or q[2] < q[0] or q[3] < q[1]:
        raise ValueError(f"网格框必须是 [0, {GRID_MAX}] 内的 4 个整数且 x2 ≥ x1、y2 ≥ y1，收到 {box_q!r}")
    return int(q[0]), int(q[1]), int(q[2]), int(q[3])


# ---------------------------------------------------------------------------
# 坐标
# ---------------------------------------------------------------------------


def box_q(box_abs: Sequence[float], width: float, height: float) -> list[int]:
    """原图像素的框 → 网格整数（oracle_sim.quantize_box）。"""
    return OS.quantize_box(box_abs, width, height)


def box_abs(q: Sequence[int], width: float, height: float) -> list[float]:
    """网格整数的框 → 原图像素（oracle_sim.dequantize_box = geometry.norm1000_to_abs）。"""
    return OS.dequantize_box(q, width, height)


def points_abs(points_q: Sequence[Sequence[int]], width: float, height: float) -> np.ndarray:
    return OS.dequantize_xy(points_q, width, height)


# ---------------------------------------------------------------------------
# 提示与目标文本
# ---------------------------------------------------------------------------


def _expr(expression: str) -> str:
    e = str(expression).strip()
    if not e:
        raise ValueError("指代句为空")
    return e


def spline_prompt(expression: str, box_q_: Sequence[int], n: int = SPLINE_N_DEFAULT) -> str:
    x1, y1, x2, y2 = check_box_q(box_q_)
    return SPLINE_TEMPLATE.format(expression=_expr(expression), x1=x1, y1=y1, x2=x2, y2=y2, n=check_n(n))


def correct_prompt(expression: str, box_q_: Sequence[int], m: int = CORRECT_M_DEFAULT) -> str:
    x1, y1, x2, y2 = check_box_q(box_q_)
    return CORRECT_TEMPLATE.format(expression=_expr(expression), x1=x1, y1=y1, x2=x2, y2=y2, m=check_m(m))


def spline_text(ctrl_q: Sequence[Sequence[int]]) -> str:
    """n 个控制点（网格整数）→ 目标文本 [[x, y], …]（oracle_sim.spline_text 的单条外环写法）。"""
    pts = [[int(p[0]), int(p[1])] for p in ctrl_q]
    return OS.spline_text([{"role": "outer", "ctrl": pts}])


def corrections_text(items: Sequence[Any]) -> str:
    """修正框（[{"sign", "box"（网格整数）}]）→ 目标文本（oracle_sim.corrections_text）；零个写 []。"""
    return OS.corrections_text([{"sign": d["sign"], "box": [int(v) for v in d["box"]]} for d in items])


def max_new_tokens_spline(n: int = SPLINE_N_DEFAULT) -> int:
    return SPLINE_TOKENS_BASE + SPLINE_TOKENS_PER_POINT * check_n(n)


def max_new_tokens_correct(m: int = CORRECT_M_DEFAULT) -> int:
    return CORRECT_TOKENS_BASE + CORRECT_TOKENS_PER_BOX * check_m(m)


# ---------------------------------------------------------------------------
# 解析（严格）
# ---------------------------------------------------------------------------


def _load(text: str) -> tuple[Any, X2Failure | None]:
    s = str(text).strip()
    if not s:
        return None, X2Failure.EMPTY
    try:
        return json.loads(s), None
    except ValueError:
        return None, X2Failure.MALFORMED_JSON


def _coord_failure(vals: Sequence[Any]) -> X2Failure | None:
    if not all(_is_int(v) for v in vals):
        return X2Failure.NON_INTEGER
    if not all(0 <= v <= GRID_MAX for v in vals):
        return X2Failure.OUT_OF_RANGE
    return None


def parse_spline(text: str, *, n: int = SPLINE_N_DEFAULT) -> X2Parse:
    """描样条的回答 → X2Parse(value = [[x, y], …]（恰 n 个网格整数点）, failure)。规则见模块说明；空串判 empty。"""
    n = check_n(n)
    doc, f = _load(text)
    if f is not None:
        return X2Parse(None, f)
    if not isinstance(doc, list):
        return X2Parse(None, X2Failure.NOT_LIST)
    if any(not isinstance(p, list) or len(p) != 2 for p in doc):
        return X2Parse(None, X2Failure.BAD_POINT)
    f = _coord_failure([v for p in doc for v in p])
    if f is not None:
        return X2Parse(None, f)
    if len(doc) != n:
        return X2Parse(None, X2Failure.WRONG_COUNT)
    return X2Parse([[int(p[0]), int(p[1])] for p in doc], None)


def parse_corrections(text: str, *, m: int = CORRECT_M_DEFAULT) -> X2Parse:
    """指修正框的回答 → X2Parse(value = [{"sign", "box"（网格整数）}]（至多 m 个，可以为空）, failure)。规则见模块说明。"""
    m = check_m(m)
    doc, f = _load(text)
    if f is not None:
        return X2Parse(None, f)
    if not isinstance(doc, list):
        return X2Parse(None, X2Failure.NOT_LIST)
    out = []
    for it in doc:
        if not isinstance(it, dict) or sorted(it) != sorted(CORRECTION_KEYS) or not isinstance(it["box"], list) or len(it["box"]) != 4:
            return X2Parse(None, X2Failure.BAD_ITEM)
        if it["sign"] not in SIGNS:
            return X2Parse(None, X2Failure.BAD_SIGN)
        f = _coord_failure(it["box"])
        if f is not None:
            return X2Parse(None, f)
        b = [int(v) for v in it["box"]]
        if b[2] < b[0] or b[3] < b[1]:
            return X2Parse(None, X2Failure.INVERTED)
        out.append({"sign": it["sign"], "box": b})
    if len(out) > m:
        return X2Parse(None, X2Failure.TOO_MANY)
    return X2Parse(out, None)


# ---------------------------------------------------------------------------
# 网格 → 工具的输入
# ---------------------------------------------------------------------------


def spline_loops(ctrl_q: Sequence[Sequence[int]], width: float, height: float) -> list[dict]:
    """网格整数的控制点 → guidance 的样条环（一条外环，控制点为原图像素）。"""
    return [{"role": "outer", "ctrl": points_abs(ctrl_q, width, height)}]


def spline_mask(ctrl_q: Sequence[Sequence[int]], box_abs_: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
    """直接光栅化（X-1 的 B 样条通道）：guidance.rasterize_loops，采样点夹在提出的框内。shape = (H, W)。"""
    h, w = int(shape[0]), int(shape[1])
    return G.rasterize_loops(spline_loops(ctrl_q, w, h), (h, w), box=[float(v) for v in box_abs_])


def proposals_abs(items: Sequence[Any], width: float, height: float) -> list[tuple[list[float], str]]:
    """修正框（网格）→ guidance.correction_round 的 proposals：[(原图像素的框, 符号)]。"""
    return [(box_abs(d["box"], width, height), str(d["sign"])) for d in items]


__all__ = [
    "CORRECT_M_DEFAULT", "CORRECT_TEMPLATE", "GRID_MAX", "JSON_SEPARATORS", "PROMPT_VERSION", "SIGNS", "SPLINE_N_DEFAULT",
    "SPLINE_TEMPLATE", "X2Failure", "X2Parse", "box_abs", "box_q", "check_box_q", "check_m", "check_n", "correct_prompt",
    "corrections_text", "failure_name", "max_new_tokens_correct", "max_new_tokens_spline", "parse_corrections", "parse_spline",
    "points_abs", "proposals_abs", "spline_loops", "spline_mask", "spline_prompt", "spline_text",
]
