"""解析层的公共部分：失败原因、结果类型、解析失败率记账、生成结果的统一入口（简报 §8.1；DEV-0006）。

与模型无关。各家族原生格式的知识（框的键名、轴序、标签语法）在 src/adapters/<family>.py 里（简报 §4 第 6 条）。

总原则（DEV-0006 任务书、D10、D19）：
- 每个解析函数返回"值 + 失败原因"的 NamedTuple，失败原因是最后一个字段 `failure`：成功时为 None；失败时取自有限枚举
  ParseFailure（网格解析沿用 src.grid.codec.GridParseFailure），其余字段一律为 None。
- 规则确定，写在各函数的 docstring 里：对真实输出里常见的包装（```json 代码块、前后的说明文字、剥离停止 token 后的残留空白）
  鲁棒；对含糊的情况（多个候选、坐标越界、框颠倒）判失败，不挑一个猜。任何一个框（或点）坏了，整条判失败，不返回其余的。
- 解析器只看剥离思考段之后的 GenerationResult.text（D10）；思考段未闭合与空回答由 parse_generation 先判。
"""

from __future__ import annotations

import enum
from collections import Counter
from typing import Any, Callable, Iterable, NamedTuple


class ParseFailure(str, enum.Enum):
    """解析失败原因（有限枚举，供统计解析失败率；简报 §8.1）。

    前 6 个取值与 DEV-0005 的 internvl35.BoxParseFailure 相同，原样保留（那个名字现在是本枚举的别名）。
    """

    # —— 框（DEV-0005 原有取值）——
    NO_BOX = "no_box"  # 找不到框：没有 <box> 标签（InternVL）；没有 JSON（Qwen/Gemma）；parse_one_box 遇到零个框
    UNBALANCED_TAGS = "unbalanced_tags"  # InternVL：<box> 与 </box> 没有一开一闭地交替出现
    MALFORMED = "malformed"  # 找到了唯一的候选，但内容不合格式：语法错、结构不符、非数值、个数不对、计数是小数或负数等
    OUT_OF_RANGE = "out_of_range"  # 坐标越界：大于 1000；JSON 格式里的负数也算（JSON 数本身合法，只是不在 [0, 1000]）
    INVERTED = "inverted"  # 框颠倒：x2 < x1 或 y2 < y1
    MULTIPLE_BOXES = "multiple_boxes"  # 只要一个框（REC）却给了多个，不猜用哪个
    # —— 通用（DEV-0006 新增）——
    EMPTY = "empty"  # 回答去掉首尾空白后为空（parse_generation）
    THINKING_UNCLOSED = "thinking_unclosed"  # 思考段开了没闭合，没有最终回答（parse_generation；判定规则在适配器，D19）
    AMBIGUOUS = "ambiguous"  # 多个候选：多个代码块、多个顶层 JSON 值、JSON 对象里的重复键
    NO_JSON = "no_json"  # 只由 extract_json 给出：找不到 JSON；框与多边形解析把它换成 no_box / no_polygon
    # —— 计数 ——
    NO_NUMBER = "no_number"  # 没有数
    MULTIPLE_NUMBERS = "multiple_numbers"  # 两个以上不同的数值
    # —— 选择（多选字母、具名二选一）——
    NO_CHOICE = "no_choice"  # 认不出任何一个选项
    MULTIPLE_CHOICES = "multiple_choices"  # 认出两个以上不同的选项（例如回显了选项列表）
    # —— 多边形 ——
    NO_POLYGON = "no_polygon"  # 没有 JSON，或 JSON 是空列表
    TOO_FEW_POINTS = "too_few_points"  # 某个多边形少于 3 个点
    # —— 停止原因（DEV-0011；追加在末尾，前面的取值与顺序不变）——
    TRUNCATED = "truncated"  # 被 max_new_tokens 截断：GenerationResult.stopped is False（parse_generation 判定，D21、D22-3）


class ParseResult(NamedTuple):
    """通用解析结果：成功时 failure 为 None；失败时 value 为 None。"""

    value: Any
    failure: ParseFailure | None


class BoxParse(NamedTuple):
    """框解析结果（适配器的 parse_boxes）。boxes 是内部格式（constants.BOX_FORMAT：绝对像素、浮点、xyxy），按出现顺序；
    labels 与 boxes 等长，原生格式里没有标签的框记 None。失败时 boxes、labels 都是 None。

    成功时 boxes 可以是空元组：只在原生格式明确写出"没有框"时出现（Qwen/Gemma 的 JSON 空列表 []），见各家族的 parse_boxes。
    """

    boxes: tuple[tuple[float, float, float, float], ...] | None
    labels: tuple[str | None, ...] | None
    failure: ParseFailure | None


class OneBoxParse(NamedTuple):
    """只要一个框的解析结果（适配器的 parse_one_box，见 one_box）。"""

    box: tuple[float, float, float, float] | None
    failure: ParseFailure | None


def one_box(parsed: BoxParse) -> OneBoxParse:
    """只要一个框的场景（如 RefCOCO 的 REC），建在 parse_boxes 的结果之上：

    - parse_boxes 失败 → 原样给出它的失败原因（包括 no_box）；
    - 成功但一个框都没有（原生格式明确写了"没有框"，如 JSON 空列表）→ no_box；
    - 多于一个框 → multiple_boxes，不猜用哪个（同 D10 对网格多块的规则），即使几个框完全相同。
    """
    if parsed.failure is not None:
        return OneBoxParse(None, parsed.failure)
    if len(parsed.boxes) == 0:
        return OneBoxParse(None, ParseFailure.NO_BOX)
    if len(parsed.boxes) > 1:
        return OneBoxParse(None, ParseFailure.MULTIPLE_BOXES)
    return OneBoxParse(parsed.boxes[0], None)


def parse_generation(result: Any, parse_fn: Callable[..., Any], **kw: Any) -> Any:
    """一次生成的统一解析入口。result 是 GenerationResult（只用它的 thinking_unclosed、stopped 与 text 三个字段）。

    判定顺序（DEV-0011 加第 2 步）：
    1. result.thinking_unclosed → thinking_unclosed（思考段开了没闭合，没有最终回答；D19）；
    2. result.stopped is False → truncated（被 max_new_tokens 截断，D21、D22-3）；
    3. result.text.strip() == "" → empty；
    4. 否则返回 parse_fn(result.text, **kw)，原样不动。

    **E1 的协议：截断一律判失败，不尝试从截断的输出里解析**——截断的多边形、网格或框列表可能恰好是一个更短的合法结果，
    解析出来会被当成模型的回答。截断与思考段未闭合在解析失败率里各自单列（ParseTally 按原因计数）。
    stopped 为 None（字段未记录：旧的构造方式与测试替身，或没有这个字段的对象，例如健康检查的 _Answer）时跳过第 2 步，
    行为与 DEV-0011 之前相同；三个适配器的 generate / generate_batch 一律填写 True/False。

    返回值与 parse_fn 的返回类型相同（ParseResult、BoxParse、OneBoxParse、GridParse……），调用方不必区分失败来自哪一步：
    前三种失败的做法是取 parse_fn("", **kw)（对空串，每个解析函数都返回值字段全为 None 的失败），只把 failure 换成
    thinking_unclosed / truncated / empty。所以 parse_fn 必须返回带 failure 字段的 NamedTuple，且对空串判失败；否则抛 TypeError。
    """
    if result.thinking_unclosed:
        reason = ParseFailure.THINKING_UNCLOSED
    elif getattr(result, "stopped", None) is False:
        reason = ParseFailure.TRUNCATED
    elif result.text.strip() == "":
        reason = ParseFailure.EMPTY
    else:
        return parse_fn(result.text, **kw)
    base = parse_fn("", **kw)
    fields = getattr(base, "_asdict", None)
    if fields is None or "failure" not in fields():
        raise TypeError(f"parse_fn 必须返回带 failure 字段的 NamedTuple，收到 {type(base).__name__}")
    values = fields()
    if values["failure"] is None or any(v is not None for k, v in values.items() if k != "failure"):
        raise TypeError(f"parse_fn 对空串必须返回值字段全为 None 的失败，收到 {base!r}")
    return base._replace(failure=reason)


def _failure_of(item: Any) -> enum.Enum | None:
    """ParseTally.add 的输入 → 失败原因。接受带 failure 字段的解析结果，或直接给出失败原因（None 表示成功）。"""
    failure = item.failure if hasattr(item, "failure") else item
    if failure is None:
        return None
    if isinstance(failure, enum.Enum) and isinstance(failure, str):
        return failure
    raise TypeError(f"失败原因必须是 str 枚举（ParseFailure、GridParseFailure 等）或 None，收到 {failure!r}")


class ParseTally:
    """累计解析结果，给出解析失败率（简报 §8.1、§8.2"解析失败率"）。

    - add(x)：x 是任何带 failure 字段的解析结果（ParseResult、BoxParse、OneBoxParse、GridParse……），或者直接是失败原因
      （str 枚举的成员；None 表示成功）。失败原因按枚举的取值（字符串）计数，所以 GridParseFailure 这类其他 str 枚举也能收；
      普通字符串不收（防止拼写错误悄悄成为新的"原因"）。
    - n、n_ok、n_failed、failure_rate（n = 0 时为 None：空集合报 null，不与"全错"同形）、counts（原因 → 次数）。
    - 可合并：merge(other) / a + b 返回新的计数，与相加的顺序无关。
    - to_dict()：可直接写进 metrics.json。键顺序固定为 n、n_ok、n_failed、failure_rate、failures；
      failures 只列出现过的原因，按原因字符串的字典序排列，所以相同的计数总是得到逐字相同的 JSON。
    """

    def __init__(self, items: Iterable[Any] = ()) -> None:
        self._n = 0
        self._failures: Counter[str] = Counter()
        for item in items:
            self.add(item)

    def add(self, item: Any) -> None:
        failure = _failure_of(item)
        self._n += 1
        if failure is not None:
            self._failures[failure.value] += 1

    @property
    def n(self) -> int:
        return self._n

    @property
    def n_failed(self) -> int:
        return sum(self._failures.values())

    @property
    def n_ok(self) -> int:
        return self._n - self.n_failed

    @property
    def failure_rate(self) -> float | None:
        return None if self._n == 0 else self.n_failed / self._n

    @property
    def counts(self) -> dict[str, int]:
        return {k: self._failures[k] for k in sorted(self._failures)}

    def merge(self, other: ParseTally) -> ParseTally:
        if not isinstance(other, ParseTally):
            raise TypeError(f"只能与 ParseTally 合并，收到 {type(other).__name__}")
        out = ParseTally()
        out._n = self._n + other._n
        out._failures = self._failures + other._failures
        return out

    def __add__(self, other: ParseTally) -> ParseTally:
        return self.merge(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParseTally):
            return NotImplemented
        return self._n == other._n and self.counts == other.counts

    __hash__ = None  # 可变对象，不可哈希

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self._n,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "failure_rate": self.failure_rate,
            "failures": self.counts,
        }

    def __repr__(self) -> str:
        return f"ParseTally(n={self._n}, failures={self.counts})"
