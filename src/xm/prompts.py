"""X-M 的两次调用（DEV-0080；指令第三节）：提示原文、目标文本与解析都是常量与纯函数，训练（DEV-0081）与推理（src/xm/runner.py）共用这一份。

坐标同 X-2（src/explore/x2_prompts）：0–1000 网格的整数，x 在前；原图像素 ↔ 网格用 x2_prompts.box_q / box_abs（oracle_sim 的量化与
geometry.norm1000_to_abs）。JSON 的分隔符同 X-2（", "、": "）。

**调用 1：列出实例**（list_prompt）：原图 + 类别名 → 该类所有实例的框，阅读顺序，至多 MAX_INSTANCES 个，`[[x1, y1, x2, y2], …]`；
没有就 `[]`。
- 解析（parse_boxes，结构严格、数量与范围宽松）：整个回答（去掉首尾空白）必须是一个 JSON 列表，每项是 4 个 JSON 整数（不收小数、布尔、
  字符串）；否则整条失败（empty / malformed_json / not_list / bad_box / non_integer）。之后：
  - 多于 MAX_INSTANCES 个只取前 MAX_INSTANCES 个（n_capped 记丢掉的个数）；
  - 坐标越界截到 [0, 1000]（n_clipped 记被截过的框数）；
  - 截过之后 x2 < x1 或 y2 < y1 的框丢掉（n_inverted 记数；面积为 0 的框照样保留，同 E1 的 parse_boxes）。
  `[]` 是成功、零个框。

**调用 2：描轮廓**（trace_prompt）：原图 + 类别名 + 调用 1 的框（NMS 之后，按编号一行一个）→ 每个框一条 SPLINE_N 个控制点的闭合周期
均匀三次 B 样条，`[[[x, y] × n], …]`，与框一一对应、顺序相同。
- 解析（parse_splines）：整个回答必须是一个 JSON 列表（否则整条失败：empty / malformed_json / not_list），且恰有 k 项（k = 框数；不符
  记 count_mismatch，整条失败——条数对不上时哪条对哪个框无从知道）。逐条：每项是 [x, y] 的列表、坐标是 JSON 整数（bad_point /
  non_integer 记这一条失败）；越界截到 [0, 1000]（n_clipped）；点数不是 n 记 wrong_count。失败的条按 src/xm/post 的规则退回
  （SPLINE_FAILURE_FALLBACK），并计数。

生成上限（推理时；截断记失败，同 E1 的 parse_generation）：max_new_tokens_list、max_new_tokens_trace(k)。两个提示都关思考。

提示词原定可在调参集 xm_tune 上改；D168 起 xm-prompts-v1 冻结（FROZEN = True，预注册 XM-4a）。真要再改，换 PROMPT_VERSION（进运行
身份，另记两段原文的 sha256）并重新冻结。判定集、最终报告集与 LVIS 子集只用冻结的提示（src/xm/gate 核对）。只 import 不含学习的模块。
"""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any, NamedTuple, Sequence

from src.explore import x2_prompts as XP
from src.xm import targets as XT

PROMPT_VERSION = "xm-prompts-v1"
FROZEN = True  # D168：xm-prompts-v1 在 X-M 训练之前冻结（预注册 XM-4a）；之后不再改提示
GRID_MAX = XP.GRID_MAX  # 1000
SPLINE_N = XT.SPLINE_N  # 16（指令第三节；同 X-2 与训练目标）
MAX_INSTANCES = XT.MAX_BOXES  # 30（指令第三节；训练目标同一个上限）
JSON_SEPARATORS = XP.JSON_SEPARATORS

LIST_TEMPLATE = (
    "Find every instance of the category \"{category}\" in the image. Give the bounding box of each instance in reading order "
    "(top to bottom, then left to right), at most {max_n} boxes. Coordinates are integers on a 0-1000 grid of the image, x first. "
    "If there is no instance, output []. Output only JSON: [[x1, y1, x2, y2], ...]."
)
TRACE_TEMPLATE = (
    "Each numbered box below contains one instance of the category \"{category}\" in the image ({k} listed):\n{box_lines}\n"
    "For each box, in the same order, trace the outer contour of its instance with a closed periodic uniform cubic B-spline of "
    "exactly {n} control points. Coordinates are integers on a 0-1000 grid of the image, x first. "
    "Output only JSON: a list of {k} splines, one per box, each [[x, y], ...] with exactly {n} points."
)
BOX_LINE = "{i}: [{x1}, {y1}, {x2}, {y2}]"

# 生成上限：基数 + 每个框 / 每条样条。DEV-0079 用 Qwen3.5 的分词器实测：一条 16 点样条约 157 个 token（30 个实例约 4,700），所以调用 2
# 按框数给 200 / 条 + 基数（管理员 D167 的口径）；一个框 4 个数约 20 个 token，给到约 2 倍（基座的原生格式带 label，约 35 个 / 框）。
LIST_TOKENS_BASE = 64
LIST_TOKENS_PER_BOX = 48
TRACE_TOKENS_BASE = 64
TRACE_TOKENS_PER_SPLINE = 200
BUDGET_RULE = (f"调用 1：{LIST_TOKENS_BASE} + {LIST_TOKENS_PER_BOX} × {MAX_INSTANCES}；调用 2：{TRACE_TOKENS_BASE} + {TRACE_TOKENS_PER_SPLINE} × 框数"
               "（一批取批内各对的最大值）；截断记失败")


class XMFailure(str, enum.Enum):
    """解析失败的分类（parse_generation 另给 truncated、empty、thinking_unclosed）。"""

    EMPTY = "empty"
    MALFORMED_JSON = "malformed_json"
    NOT_LIST = "not_list"
    BAD_BOX = "bad_box"  # 调用 1：某一项不是 4 个元素的列表
    NON_INTEGER = "non_integer"  # 坐标不是 JSON 整数
    COUNT_MISMATCH = "count_mismatch"  # 调用 2：条数与框数不符
    BAD_POINT = "bad_point"  # 调用 2 的某一条：不是 [x, y] 的列表
    WRONG_COUNT = "wrong_count"  # 调用 2 的某一条：点数不是 n


class XMParse(NamedTuple):
    """解析结果：成功时 failure 为 None、value 是 dict；失败时 value 为 None（parse_generation 要求的形状）。"""

    value: Any
    failure: Any


def failure_name(f: Any) -> str | None:
    return XP.failure_name(f)


def identity() -> dict[str, Any]:
    """进运行身份的提示口径。"""
    h = hashlib.sha256((LIST_TEMPLATE + "\n\x00\n" + TRACE_TEMPLATE + "\n\x00\n" + BOX_LINE).encode("utf-8")).hexdigest()
    return {"version": PROMPT_VERSION, "templates_sha256": h, "frozen": FROZEN, "n": SPLINE_N, "max_instances": MAX_INSTANCES,
            "max_new_tokens": {"list": max_new_tokens_list(), "trace_base": TRACE_TOKENS_BASE, "trace_per_spline": TRACE_TOKENS_PER_SPLINE,
                               "rule": BUDGET_RULE}}


# ---------------------------------------------------------------------------
# 提示与目标文本
# ---------------------------------------------------------------------------


def _category(category: str) -> str:
    c = str(category).strip()
    if not c:
        raise ValueError("类别名为空")
    return c


def list_prompt(category: str) -> str:
    return LIST_TEMPLATE.format(category=_category(category), max_n=MAX_INSTANCES)


def trace_prompt(category: str, boxes_q: Sequence[Sequence[int]], n: int = SPLINE_N) -> str:
    if not boxes_q:
        raise ValueError("调用 2 至少要一个框")
    lines = []
    for i, b in enumerate(boxes_q, start=1):
        x1, y1, x2, y2 = XP.check_box_q(b)
        lines.append(BOX_LINE.format(i=i, x1=x1, y1=y1, x2=x2, y2=y2))
    return TRACE_TEMPLATE.format(category=_category(category), k=len(boxes_q), box_lines="\n".join(lines), n=XP.check_n(n))


def boxes_text(boxes_q: Sequence[Sequence[int]]) -> str:
    """调用 1 的目标文本：[[x1, y1, x2, y2], …]（网格整数；零个写 []）。训练目标的同一个函数（src/xm/targets.box_list_text）。"""
    return XT.box_list_text(boxes_q)


def splines_text(ctrls_q: Sequence[Sequence[Sequence[int]]]) -> str:
    """调用 2 的目标文本：[s₁, s₂, …]，每条逐字是 X-2 的 x2_prompts.spline_text，与框一一对应。训练目标的同一个函数
    （src/xm/targets.spline_list_text）。"""
    return XT.spline_list_text(ctrls_q)


def max_new_tokens_list() -> int:
    return LIST_TOKENS_BASE + LIST_TOKENS_PER_BOX * MAX_INSTANCES


def max_new_tokens_trace(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"框数 k 必须是正整数，收到 {k!r}")
    return TRACE_TOKENS_BASE + TRACE_TOKENS_PER_SPLINE * k


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _clip(v: int) -> int:
    return min(max(int(v), 0), GRID_MAX)


def _load(text: str) -> tuple[Any, XMFailure | None]:
    s = str(text).strip()
    if not s:
        return None, XMFailure.EMPTY
    try:
        return json.loads(s), None
    except ValueError:
        return None, XMFailure.MALFORMED_JSON


def parse_boxes(text: str) -> XMParse:
    """调用 1 的回答 → XMParse(value = {"boxes_q", "n_raw", "n_capped", "n_clipped", "n_inverted"}, failure)。规则见模块说明。"""
    doc, f = _load(text)
    if f is not None:
        return XMParse(None, f)
    if not isinstance(doc, list):
        return XMParse(None, XMFailure.NOT_LIST)
    if any(not isinstance(b, list) or len(b) != 4 for b in doc):
        return XMParse(None, XMFailure.BAD_BOX)
    if not all(_is_int(v) for b in doc for v in b):
        return XMParse(None, XMFailure.NON_INTEGER)
    kept = doc[:MAX_INSTANCES]
    out, n_clip, n_inv = [], 0, 0
    for b in kept:
        c = [_clip(v) for v in b]
        n_clip += int(c != [int(v) for v in b])
        if c[2] < c[0] or c[3] < c[1]:
            n_inv += 1
            continue
        out.append(c)
    return XMParse({"boxes_q": out, "n_raw": len(doc), "n_capped": len(doc) - len(kept), "n_clipped": n_clip, "n_inverted": n_inv}, None)


def _one_spline(item: Any, n: int) -> tuple[list[list[int]] | None, str | None, bool]:
    if not isinstance(item, list) or any(not isinstance(p, list) or len(p) != 2 for p in item):
        return None, XMFailure.BAD_POINT.value, False
    if not all(_is_int(v) for p in item for v in p):
        return None, XMFailure.NON_INTEGER.value, False
    pts = [[_clip(p[0]), _clip(p[1])] for p in item]
    clipped = pts != [[int(p[0]), int(p[1])] for p in item]
    if len(pts) != n:
        return None, XMFailure.WRONG_COUNT.value, clipped
    return pts, None, clipped


def parse_splines(text: str, *, k: int, n: int = SPLINE_N) -> XMParse:
    """调用 2 的回答 → XMParse(value = {"splines"（k 项，失败的为 None）, "failures"（k 项，成功的为 None）, "n_clipped"}, failure)。
    整条失败（结构、条数）时 value 为 None。规则见模块说明。"""
    n = XP.check_n(n)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"框数 k 必须是正整数，收到 {k!r}")
    doc, f = _load(text)
    if f is not None:
        return XMParse(None, f)
    if not isinstance(doc, list):
        return XMParse(None, XMFailure.NOT_LIST)
    if len(doc) != k:
        return XMParse(None, XMFailure.COUNT_MISMATCH)
    splines, fails, n_clip = [], [], 0
    for item in doc:
        pts, why, clipped = _one_spline(item, n)
        splines.append(pts)
        fails.append(why)
        n_clip += int(clipped)
    return XMParse({"splines": splines, "failures": fails, "n_clipped": n_clip}, None)


# DEV-0081 任务书用的名字（训练样本的提示与目标；同一组函数）。调用 2 的提示不说框是"全部"实例：训练时一条样本至多 12 个框，
# 多的按阅读顺序切成几条（DEV-0081 第 1 条），推理时一次至多 30 个
call1_prompt = list_prompt
call2_prompt = trace_prompt
format_boxes = boxes_text
format_splines = splines_text


__all__ = [
    "call1_prompt", "call2_prompt", "format_boxes", "format_splines", "BOX_LINE", "FROZEN", "GRID_MAX", "JSON_SEPARATORS", "LIST_TEMPLATE", "MAX_INSTANCES", "PROMPT_VERSION", "SPLINE_N", "TRACE_TEMPLATE",
    "XMFailure", "XMParse", "boxes_text", "failure_name", "identity", "list_prompt", "max_new_tokens_list", "max_new_tokens_trace",
    "parse_boxes", "parse_splines", "splines_text", "trace_prompt",
]
