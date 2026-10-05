"""结构强制的多边形（DEV-0036；D72-3，N26 → 问题 2 选 (a)；事后诊断，不改 G1 的结论）。

口径照 configs/e1_posthoc.json 的 format_forced_poly（规则文字在配置里，这里实现）：
- 输出必须恰好是 open 点 (point_sep 点)… close，再接一个停止 token；点 = 整数 xy_sep 整数；整数是 0–1000 的十进制整数，
  不带前导零（"0" 本身可以），所以最多 4 位、4 位时只能是 1000。四段结构 open / xy_sep / point_sep / close 是 "[["、", "、"], ["、"]]"，
  即提示里 JSON 格式的写法。
- token 化（poly_format_ids）：数字逐位，每个十进制数字是本家族分词器里的一个单 token（Qwen 的分词器本来就把数字逐位切开）；四段结构
  各取分词器单独编码它们得到的 token 序列（Qwen3.5：[[、"," + " "、"]," + " ["、]]）。运行时用探针多边形核对这就是本家族对整段文字的
  规范切分（整段编码 = 各段编码与逐位数字的拼接，解码回来逐字相同），不成立就报错，不静默换写法。
- 语法（PolyGrammar，纯 Python 的状态机）：开头只留 open；写数字时只留合乎整数规则的数字，已写了至少一位时另留下一段的第一个 token
  （x 之后是 xy_sep；y 之后是 point_sep（已写完的点数 + 1 < max_points）与 close（已写完的点数 + 1 ≥ min_points））；一段写到一半时只留
  这一段的下一个 token；close 之后只留本家族的停止 token。点数由模型决定，上限 max_points。
- PolyStructureProcessor（HF 的 LogitsProcessor，经 generate_batch 的 logits_processor 传入，不改模型）：逐行维护状态，把不合语法的
  token 的 logit 置为 −inf。已经停下的行不再限制（HF 给它补 pad）。一个实例只用于一次 generate（每批新建一个）。
- 入口断言（check_poly_output）：每行以停止 token 结束，token 序列被同一份语法逐个接受，点数在 [min_points, max_points]；
  不成立说明处理器没生效，报错而不是当成模型的输出。
- 近矩形（near_rectangle）：只有一个多边形、恰好 n_points 个点，且多边形面积（鞋带公式取绝对值）与外接框面积之比 ≥ min_area_ratio。

本模块只有纯函数与处理器（不碰文件）；运行（批、续跑、输出）在 src/e1/posthoc_runner.py。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.e1.forced import ForcedOutputError
from src.e1.posthoc import FORCED_POLY_PIECES

try:  # 与适配器相同的基类（HF generate 的处理器列表只认它）
    from transformers import LogitsProcessor as _LogitsProcessorBase
except ImportError:  # pragma: no cover — 开发机与服务器都装了 transformers
    _LogitsProcessorBase = object

DIGITS = "0123456789"
# 探针多边形：覆盖 1–4 位数、0 与 1000、前后各种组合；整段的编码必须是规范切分（见模块说明）
_PROBES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1000, 999), (12, 3)),
    ((7, 100), (45, 1000), (999, 10), (5, 60)),
    ((1000, 1000), (0, 1), (10, 0), (100, 101), (9, 99), (250, 750), (31, 4), (600, 8)),
)


class PolyFormatError(RuntimeError):
    """本家族的分词器不支持这套规范切分（数字不是逐位、四段结构的切分随上下文变、两段互为前缀等）。"""


def polygon_text(points: Sequence[Sequence[int]], pieces: Mapping[str, str]) -> str:
    """点 → 规范的多边形文字（open 点 (point_sep 点)… close）。"""
    body = pieces["point_sep"].join(f"{int(x)}{pieces['xy_sep']}{int(y)}" for x, y in points)
    return pieces["open"] + body + pieces["close"]


def poly_format_ids(tokenizer: Any, pieces: Mapping[str, str]) -> dict[str, Any]:
    """数字与四段结构的 token（规则见模块说明与配置的 tokenization）。返回写进 config.yaml 的记录。"""

    def enc(s: str) -> list[int]:
        return [int(i) for i in tokenizer(s, add_special_tokens=False)["input_ids"]]

    if set(pieces) != set(FORCED_POLY_PIECES):
        raise PolyFormatError(f"pieces 必须恰好是 {FORCED_POLY_PIECES}，收到 {sorted(pieces)}")
    digits = []
    for d in DIGITS:
        ids = enc(d)
        if len(ids) != 1:
            raise PolyFormatError(f"数字 {d!r} 在本家族的分词器里不是单个 token：{ids}")
        digits.append(ids[0])
    if len(set(digits)) != len(DIGITS):
        raise PolyFormatError(f"十个数字撞到了同一个 token：{digits}")
    seq = {k: tuple(enc(pieces[k])) for k in FORCED_POLY_PIECES}
    for k, s in seq.items():
        if not s:
            raise PolyFormatError(f"{k} = {pieces[k]!r} 编码为空")
        if set(s) & set(digits):
            raise PolyFormatError(f"{k} = {pieces[k]!r} 的编码里有数字 token：{s}")
    ps, cl = seq["point_sep"], seq["close"]
    if ps[: len(cl)] == cl or cl[: len(ps)] == ps:
        raise PolyFormatError(f"point_sep {ps} 与 close {cl} 互为前缀：y 之后的两种写法分不开")
    probes = []
    for pts in _PROBES:
        text = polygon_text(pts, pieces)
        want: list[int] = list(seq["open"])
        for k, (x, y) in enumerate(pts):
            if k:
                want += seq["point_sep"]
            want += [digits[int(c)] for c in str(x)] + list(seq["xy_sep"]) + [digits[int(c)] for c in str(y)]
        want += seq["close"]
        got = enc(text)
        if got != want:
            raise PolyFormatError(f"{text!r} 的编码 {got} 不是各段编码与逐位数字的拼接 {want}（不是规范切分）")
        if tokenizer.decode(want) != text:
            raise PolyFormatError(f"{want} 解码回来不是 {text!r}")
        probes.append({"text": text, "n_tokens": len(want)})
    conv = tokenizer.convert_ids_to_tokens
    return {"digit_ids": digits, "digit_tokens": [conv(i) for i in digits],
            "pieces": {k: {"text": pieces[k], "ids": list(seq[k]), "tokens": [conv(i) for i in seq[k]]} for k in FORCED_POLY_PIECES},
            "probes": probes,
            "rule": "数字逐位（每个十进制数字一个单 token）；四段结构各取单独编码的 token 序列；探针多边形的整段编码 = 各段编码与逐位"
                    "数字的拼接、解码回来逐字相同（configs/e1_posthoc.json 的 format_forced_poly.tokenization）"}


@dataclass(frozen=True)
class PolyState:
    """一行的语法状态。phase：piece（正在写某一段：cands 是还可能的段名，pos 是段内已写的 token 数）、num（正在写一个整数）、
    end（close 已写完，只留停止 token）、done（已停下）。coord：0 = x，1 = y；digits：当前整数已写的数字；n_points：已写完的点数。"""

    phase: str
    cands: tuple[str, ...] = ()
    pos: int = 0
    coord: int = 0
    digits: str = ""
    n_points: int = 0


class PolyGrammarError(ForcedOutputError):
    """生成的 token 不合语法（处理器没生效，或入口断言抓到的序列不完整）。"""


class PolyGrammar:
    """结构强制的多边形语法（规则见模块说明）。fmt = poly_format_ids 的结果。"""

    def __init__(self, fmt: Mapping[str, Any], *, min_points: int, max_points: int, stop_ids: Sequence[int],
                 coord_max: int = 1000):
        from src.parsing.spatial import MIN_POLYGON_POINTS

        if isinstance(min_points, bool) or isinstance(max_points, bool) or not MIN_POLYGON_POINTS <= min_points <= max_points:
            raise ValueError(f"要 {MIN_POLYGON_POINTS} ≤ min_points ≤ max_points，收到 {min_points}、{max_points}")
        self.digit_ids = [int(i) for i in fmt["digit_ids"]]
        self.digit_of = {t: DIGITS[k] for k, t in enumerate(self.digit_ids)}
        self.seq = {k: tuple(int(i) for i in fmt["pieces"][k]["ids"]) for k in FORCED_POLY_PIECES}
        self.stop_ids = tuple(int(s) for s in stop_ids)
        if not self.stop_ids or set(self.stop_ids) & (set(self.digit_ids) | {t for s in self.seq.values() for t in s}):
            raise ValueError(f"停止 token {self.stop_ids} 为空或与数字、四段结构的 token 重叠")
        self.min_points, self.max_points, self.coord_max = int(min_points), int(max_points), int(coord_max)
        self.max_digits = len(str(self.coord_max))

    # -- 状态机 ---------------------------------------------------------------

    def start(self) -> PolyState:
        return PolyState(phase="piece", cands=("open",))

    def _digit_ok(self, digits: str, d: str) -> bool:
        if digits == "0":  # 不允许前导零
            return False
        return int(digits + d) <= self.coord_max

    def _next_pieces(self, st: PolyState) -> tuple[str, ...]:
        if st.coord == 0:
            return ("xy_sep",)
        out = []
        if st.n_points + 1 < self.max_points:
            out.append("point_sep")
        if st.n_points + 1 >= self.min_points:
            out.append("close")
        return tuple(out)

    def allowed(self, st: PolyState) -> list[int]:
        """这一步允许的 token（升序）。done 状态不限制，调用方不该再问。"""
        if st.phase == "piece":
            return sorted({self.seq[c][st.pos] for c in st.cands})
        if st.phase == "num":
            out = {t for t, d in self.digit_of.items() if self._digit_ok(st.digits, d)}
            if st.digits:
                out |= {self.seq[c][0] for c in self._next_pieces(st)}
            return sorted(out)
        if st.phase == "end":
            return sorted(set(self.stop_ids))
        raise PolyGrammarError("已停下的行没有下一步")

    def _after_piece(self, st: PolyState, name: str) -> PolyState:
        if name == "open":
            return PolyState(phase="num", coord=0, n_points=0)
        if name == "xy_sep":
            return PolyState(phase="num", coord=1, n_points=st.n_points)
        if name == "point_sep":
            return PolyState(phase="num", coord=0, n_points=st.n_points + 1)
        return PolyState(phase="end", n_points=st.n_points + 1)  # close

    def advance(self, st: PolyState, tok: int) -> PolyState:
        tok = int(tok)
        if st.phase == "done":
            return st
        if tok not in self.allowed(st):
            raise PolyGrammarError(f"token {tok} 不合语法（状态 {st}，允许 {self.allowed(st)[:12]}）")
        if st.phase == "end":
            return dataclasses.replace(st, phase="done")
        if st.phase == "num":
            if tok in self.digit_of:
                return dataclasses.replace(st, digits=st.digits + self.digit_of[tok])
            st = dataclasses.replace(st, phase="piece", cands=self._next_pieces(st), pos=0, digits="")
        cands = tuple(c for c in st.cands if self.seq[c][st.pos] == tok)
        pos = st.pos + 1
        done = [c for c in cands if len(self.seq[c]) == pos]
        if done:
            return self._after_piece(st, done[0])
        return dataclasses.replace(st, cands=cands, pos=pos)

    def max_tokens(self) -> int:
        """最长的生成（含停止 token）：open + max_points 个点（两个 max_digits 位的数与 xy_sep）+ (max_points − 1) 个 point_sep + close + 1。"""
        s = self.seq
        return (len(s["open"]) + self.max_points * (2 * self.max_digits + len(s["xy_sep"]))
                + (self.max_points - 1) * len(s["point_sep"]) + len(s["close"]) + 1)

    def accept(self, ids: Sequence[int]) -> PolyState:
        """把一整行生成的 token 走一遍语法，返回最后的状态（不合语法就报 PolyGrammarError）。"""
        st = self.start()
        for t in ids:
            if st.phase == "done":
                raise PolyGrammarError("停止 token 之后还有 token")
            st = self.advance(st, t)
        return st

    def as_dict(self) -> dict[str, Any]:
        return {"min_points": self.min_points, "max_points": self.max_points, "coord_max": self.coord_max,
                "stop_ids": list(self.stop_ids), "max_tokens": self.max_tokens()}


class PolyStructureProcessor(_LogitsProcessorBase):
    """逐行的语法限制（见模块说明）。t = 当前已生成的 token 数：第一次调用时记下 input_ids 的宽度（左填充后整批相同），之后每步先用
    上一步生成的 token 推进每一行的状态，再把不合语法的 token 置 −inf。一个实例只用于一次 generate。"""

    def __init__(self, grammar: PolyGrammar):
        self.g = grammar
        self.start: int | None = None
        self.states: list[PolyState] = []
        self.n_calls = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.start is None:
            self.start = int(input_ids.shape[1])
            self.states = [self.g.start() for _ in range(int(input_ids.shape[0]))]
        t = int(input_ids.shape[1]) - self.start
        if t != self.n_calls or len(self.states) != int(input_ids.shape[0]):
            raise ForcedOutputError(f"处理器第 {self.n_calls} 次调用时 t = {t}：一个实例只能用于一次 generate")
        if t > 0:
            last = input_ids[:, -1].tolist()
            self.states = [self.g.advance(st, tok) for st, tok in zip(self.states, last)]
        self.n_calls += 1
        keep = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        for b, st in enumerate(self.states):
            if st.phase == "done":
                keep[b] = True  # 已停下的行：HF 给它补 pad，不再限制
            else:
                keep[b, torch.tensor(self.g.allowed(st), dtype=torch.long, device=scores.device)] = True
        return scores.masked_fill(~keep, float("-inf"))


def check_poly_output(res: Any, grammar: PolyGrammar, *, where: str = "") -> int:
    """入口断言（见模块说明）：以停止 token 结束，token 序列合语法，点数在 [min_points, max_points]。返回点数。"""
    ids = [int(i) for i in res.token_ids]
    if res.stopped is not True or not ids or ids[-1] not in grammar.stop_ids:
        raise PolyGrammarError(f"{where}：结构强制的多边形应以停止 token 结束，得到 stopped={res.stopped}、末尾 {ids[-3:]}")
    try:
        st = grammar.accept(ids)
    except PolyGrammarError as e:
        raise PolyGrammarError(f"{where}：{e}") from None
    if st.phase != "done" or not grammar.min_points <= st.n_points <= grammar.max_points:
        raise PolyGrammarError(f"{where}：语法没有走完（{st}）或点数 {st.n_points} 不在 [{grammar.min_points}, {grammar.max_points}]")
    return st.n_points


def polygon_area_ratio(points: Sequence[Sequence[float]]) -> float | None:
    """多边形面积（鞋带公式取绝对值，点按给出的顺序）/ 外接框面积；外接框面积为 0 时 None。"""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return None
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
    box = float((x.max() - x.min()) * (y.max() - y.min()))
    return area / box if box > 0 else None


def near_rectangle(polygons: Sequence[Sequence[Sequence[float]]] | None, *, n_points: int, min_area_ratio: float
                   ) -> dict[str, Any]:
    """近矩形的判定（configs/e1_posthoc.json 的 format_forced_poly.near_rectangle）。polygons = 记录的 parsed.polygons（解析失败为 None）。"""
    if polygons is None:
        return {"near_rect": False, "n_polygons": 0, "n_points": None, "area_ratio": None}
    pts = [len(p) for p in polygons]
    ratio = polygon_area_ratio(polygons[0]) if len(polygons) == 1 and pts[0] == n_points else None
    return {"near_rect": ratio is not None and ratio >= min_area_ratio, "n_polygons": len(polygons),
            "n_points": pts[0] if len(pts) == 1 else pts, "area_ratio": ratio}


def near_rect_summary(flags: Sequence[Mapping[str, Any]], valid: Sequence[bool]) -> dict[str, Any]:
    """近矩形占比：占全部 n 与占有效输出。flags 是逐条的 near_rectangle 结果。"""
    n = len(flags)
    k = sum(1 for f in flags if f["near_rect"])
    nv = sum(1 for v in valid if v)
    kv = sum(1 for f, v in zip(flags, valid) if v and f["near_rect"])
    return {"n": n, "n_near_rect": k, "share_of_all": (k / n) if n else None, "n_valid": nv,
            "n_near_rect_valid": kv, "share_of_valid": (kv / nv) if nv else None}


__all__ = [
    "PolyFormatError",
    "PolyGrammar",
    "PolyGrammarError",
    "PolyState",
    "PolyStructureProcessor",
    "check_poly_output",
    "near_rect_summary",
    "near_rectangle",
    "poly_format_ids",
    "polygon_area_ratio",
    "polygon_text",
]
