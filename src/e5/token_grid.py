"""E5 训练后系统的输出格式：E3b 的像素 token 网格（DEV-0037 §2；简报 §8.8"输出一律用像素 token 网格"）。

**token**：同 E3b 的 67 个网格 token（configs/e3b_train.json 的 vocab；src/e3/tokens）、同一种序列格式（src/e3/targets.grid_to_ids：光栅顺序，
每行 N 个值 token 之后一个行分隔，最后接本家族的回合结束 token，长度 N(N + 1) + 1 = 1057）。值 token 的下标就是 E5 格子表示里的值：
- 二值网格（c、d、e：mask）：E3b 的二值 token（mask[0]、mask[1]），值 = 0 / 1，即 sample.target_grid；
- 起伏网格（a、b：背景 + 32 档起伏，D28-13；格子表示见 src/grid/codec.py 的起伏代码）：值 = 起伏代码 c ∈ {0, …, 32}，
  c = 0（背景）→ 二值 token mask[0]；c = k + 1（起伏档 k，0 最近）→ 深度 token depth[RELIEF_DEPTH_OFFSET + k]，
  RELIEF_DEPTH_OFFSET = (DEPTH_NUM_BINS − RELIEF_NUM_LEVELS) / 2 = 16，即 64 档深度 token 正中间连续的 32 档（映射与理由写在
  configs/e5_train.json 的 format.relief_basis）。目标 = sample.extras["relief_target"]。
受限 logits 的允许集合（训练损失与受限解码同一个集合，升序）：allowed_scope = task 时 = 本任务的值 token + 行分隔 + 停止 token；
all 时 = 全部 67 个网格 token + 停止 token（同 E3b 的 tokens.allowed_ids）。

**提示**：生成器的指令原文 + E3b 网格指令的结尾句（configs/e5_train.json 的 format；结尾句从 E3b 配置的模板里取最后一句，只在那里写一处）；
指令在前；a–d 一张图（sample.image 转成 PIL 图，与零样本的 src/e5/runner 同一个转换），e 没有图。

**解析与评分**：token 序列 → src/e3/targets.parse_grid_ids（严格：不补齐、不截断、不猜）→ 值网格 → E5 的 Pred（二值：scoring.from_grid；
起伏：scoring.from_relief）→ src/e5/scoring.score，与零样本同一套评分。解析失败的原因：截断（没有以停止 token 结尾）记 ParseFailure.TRUNCATED
（同零样本的 truncated），其余三种记 scoring.E5Failure 里同名的取值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.constants import DEPTH_NUM_BINS
from src.e3.targets import GridIdsFailure, grid_to_ids, parse_grid_ids, seq_len
from src.e3.tokens import ALLOWED_SCOPES, GridTokens
from src.e5 import scoring as SC
from src.grid.codec import RELIEF_NUM_LEVELS
from src.parsing import ParseFailure
from src.synth.relief import RELIEF_TARGET_KEY, RELIEF_TASKS
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

BINARY, RELIEF = "binary", "relief"
FORMATS: tuple[str, ...] = (BINARY, RELIEF)
RELIEF_DEPTH_OFFSET = (DEPTH_NUM_BINS - RELIEF_NUM_LEVELS) // 2
if RELIEF_DEPTH_OFFSET < 0 or 2 * RELIEF_DEPTH_OFFSET + RELIEF_NUM_LEVELS != DEPTH_NUM_BINS:  # pragma: no cover - 常量写错
    raise ImportError("起伏档放不进深度 token 的正中间（DEPTH_NUM_BINS − RELIEF_NUM_LEVELS 必须是非负偶数）")
RELIEF_DEPTH_BINS: tuple[int, ...] = tuple(RELIEF_DEPTH_OFFSET + k for k in range(RELIEF_NUM_LEVELS))
RELIEF_MAPPINGS: tuple[str, ...] = ("centered",)  # configs/e5_train.json 的 format.relief_depth_bins 只认这一种

# 网格 token 序列的解析失败 → E5 评分的失败原因（取值都在 scoring 的 _FAILURES 里）
FAILURE_OF: Mapping[GridIdsFailure, str] = {
    GridIdsFailure.TRUNCATED: ParseFailure.TRUNCATED.value,
    GridIdsFailure.WRONG_LENGTH: SC.E5Failure.GRID_WRONG_LENGTH.value,
    GridIdsFailure.ROW_SEPARATOR: SC.E5Failure.GRID_ROW_SEPARATOR.value,
    GridIdsFailure.FOREIGN_TOKEN: SC.E5Failure.GRID_FOREIGN_TOKEN.value,
}


class TokenGridError(ValueError):
    pass


def task_format(task: str) -> str:
    """a、b（带起伏的任务）用起伏网格，其余用二值网格。"""
    return RELIEF if canonical_task(task) in RELIEF_TASKS else BINARY


def target_values(sample: Sample) -> np.ndarray:
    """样本的目标值网格（N×N int64）：二值任务 = target_grid（0/1），起伏任务 = 起伏代码（0 背景，k + 1 = 档 k）。"""
    if task_format(sample.task) == RELIEF:
        return np.asarray(sample.extras[RELIEF_TARGET_KEY], dtype=np.int64)
    return np.asarray(sample.target_grid, dtype=np.int64)


def ending_of(template: str) -> str:
    """模板的最后一句（最后一个 '. ' 之后的部分）；只允许占位符 {n}。"""
    import string

    s = str(template).strip()
    i = s.rfind(". ")
    end = s[i + 2:] if i >= 0 else s
    used = {f for _, f, _, _ in string.Formatter().parse(end) if f is not None}
    if not end or used - {"n"}:
        raise TokenGridError(f"取不出结尾句，或结尾句里有 {{n}} 以外的占位符：{end!r}")
    return end


def _lookup(doc: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class PromptFormat:
    """提示的格式：template（占位符 instruction、ending）、两种网格的结尾句、指令顺序。"""

    template: str
    endings: Mapping[str, str]
    order: str
    endings_from: Mapping[str, str]

    @classmethod
    def from_configs(cls, fmt: Mapping[str, Any], e3b_doc: Mapping[str, Any]) -> "PromptFormat":
        src = dict(fmt["endings_from"])
        if set(src) != set(FORMATS):
            raise TokenGridError(f"format.endings_from 必须恰好给 {FORMATS}，收到 {sorted(src)}")
        endings = {k: ending_of(_lookup(e3b_doc, v)) for k, v in src.items()}
        tmpl = str(fmt["template"])
        import string

        used = {f for _, f, _, _ in string.Formatter().parse(tmpl) if f is not None}
        if used != {"instruction", "ending"}:
            raise TokenGridError(f"format.template 的占位符必须恰好是 instruction、ending，收到 {sorted(used)}")
        if fmt["order"] not in ("instruction_first", "image_first"):
            raise TokenGridError(f"format.order 不认识：{fmt['order']!r}")
        if fmt.get("relief_depth_bins") not in RELIEF_MAPPINGS:
            raise TokenGridError(f"format.relief_depth_bins 只实现了 {RELIEF_MAPPINGS}，收到 {fmt.get('relief_depth_bins')!r}")
        return cls(template=tmpl, endings=endings, order=str(fmt["order"]), endings_from=src)

    def render(self, sample: Sample, n: int) -> str:
        ending = self.endings[task_format(sample.task)].format(n=int(n))
        return self.template.format(instruction=sample.instruction.strip(), ending=ending).strip()

    def as_dict(self) -> dict[str, Any]:
        return {"template": self.template, "endings": dict(self.endings), "endings_from": dict(self.endings_from),
                "order": self.order}


def sample_images(sample: Sample) -> list[Any]:
    """送进 build_inputs 的图：a–d 一张（PIL，与零样本 src/e5/runner 的转换相同），e 没有图。"""
    return [] if sample.image is None else [Image.fromarray(sample.image)]


@dataclass(frozen=True)
class E5Tokens:
    """E3b 的网格 token 布局（GridTokens）上的 E5 值 token 与允许集合（规则见模块说明）。"""

    tokens: GridTokens
    scope: str

    def __post_init__(self) -> None:
        if self.scope not in ALLOWED_SCOPES:
            raise TokenGridError(f"allowed_scope 必须是 {ALLOWED_SCOPES} 之一，收到 {self.scope!r}")

    @property
    def stop_id(self) -> int:
        return int(self.tokens.stop_id)

    @property
    def row_id(self) -> int:
        return int(self.tokens.row_id)

    def value_ids_of(self, fmt: str) -> tuple[int, ...]:
        mask = self.tokens.value_ids("refseg")
        if fmt == BINARY:
            return tuple(int(i) for i in mask)
        if fmt == RELIEF:
            depth = self.tokens.value_ids("depth")
            return (int(mask[0]),) + tuple(int(depth[b]) for b in RELIEF_DEPTH_BINS)
        raise TokenGridError(f"未知格式 {fmt!r}")

    def value_ids(self, task: str) -> tuple[int, ...]:
        """任务的值 token：下标就是 E5 格子表示里的值（二值 0/1；起伏代码 0..32）。"""
        return self.value_ids_of(task_format(task))

    def allowed_ids_of(self, fmt: str) -> list[int]:
        base = self.value_ids_of(fmt) + (self.row_id,) if self.scope == "task" else self.tokens.grid_ids
        return sorted(set(int(i) for i in base) | {self.stop_id})

    def allowed_ids(self, task: str) -> list[int]:
        return self.allowed_ids_of(task_format(task))

    def target_ids(self, task: str, values: np.ndarray, n: int) -> list[int]:
        return grid_to_ids(np.asarray(values), self.value_ids(task), self.row_id, self.stop_id, n=n)

    def sample_target_ids(self, sample: Sample) -> list[int]:
        v = target_values(sample)
        return self.target_ids(sample.task, v, int(v.shape[0]))

    def parse(self, task: str, ids: Sequence[int], n: int) -> tuple[np.ndarray | None, str | None]:
        """token 序列 → (值网格 uint8, None) 或 (None, E5 评分的失败原因)。"""
        p = parse_grid_ids(ids, self.value_ids(task), self.row_id, self.stop_id, n=n)
        if p.failure is not None:
            return None, FAILURE_OF[p.failure]
        return p.grid.astype(np.uint8), None

    def pred(self, task: str, ids: Sequence[int], n: int, height: int, width: int) -> SC.Pred:
        """token 序列 → E5 的 Pred（scoring 的同一组构造函数）。"""
        grid, failure = self.parse(task, ids, n)
        if failure is not None:
            return SC.failed(failure)
        return SC.from_relief(grid, height, width) if task_format(task) == RELIEF else SC.from_grid(grid, height, width)

    def layout(self) -> dict[str, Any]:
        """写进运行身份的映射：每种格式的值 token、允许集合，起伏到深度 token 的档号。"""
        return {"scope": self.scope, "stop": self.stop_id, "row": self.row_id,
                "binary": {"value_ids": list(self.value_ids_of(BINARY)), "allowed": self.allowed_ids_of(BINARY),
                           "values": "0 / 1 → E3b mask token 0 / 1"},
                "relief": {"value_ids": list(self.value_ids_of(RELIEF)), "allowed": self.allowed_ids_of(RELIEF),
                           "depth_bins": [RELIEF_DEPTH_BINS[0], RELIEF_DEPTH_BINS[-1]], "offset": RELIEF_DEPTH_OFFSET,
                           "values": "起伏代码 0（背景）→ mask token 0；代码 k + 1（起伏档 k）→ depth token 第 offset + k 档"}}


def max_new_tokens(n: int) -> int:
    """恰好写满一个合格网格（含停止 token）。"""
    return seq_len(n)


__all__ = [
    "BINARY",
    "E5Tokens",
    "FAILURE_OF",
    "FORMATS",
    "PromptFormat",
    "RELIEF",
    "RELIEF_DEPTH_BINS",
    "RELIEF_DEPTH_OFFSET",
    "TokenGridError",
    "ending_of",
    "max_new_tokens",
    "sample_images",
    "target_values",
    "task_format",
]
