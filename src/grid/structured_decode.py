"""网格的结构强制解码（DEV-0041 §2；D83：E3b+C3 与 E5 全部网格任务的正式口径，现有各臂另报作诊断）。

**结构**（只强制网格的形状，不注入任何任务知识）：n 行，每行 n 个格子位置接一个行末位置，最后一个停止位置。生成的第 t 个 token
（t 从 0 起）的允许集合是固定的：
- t = r·(n + 1) + c，c < n（第 r 行的格子）：只允许第 r 行的值 token（value_ids[r]；E3b 的分割 2 个、深度 64 个，各行相同）；
- t = r·(n + 1) + n（第 r 行的行末）：只允许行分隔 row_id；
- t = n·(n + 1)（最后一个位置）：只允许停止 token stop_id。stop_id = None 时没有这个位置（给以后不以停止 token 收尾的网格用）。
长度恒为 n(n + 1) + 1（n = 32 时 1057；stop_id = None 时 n(n + 1)）。与 src/e3/targets 的序列格式（grid_to_ids / parse_grid_ids）
逐位一致：结构强制下的输出必然解析成功。

**参数化**（E5 的二值 c、d、e 与起伏 a、b 以后也用）：每行的值 token 集合、行分隔 id、停止 id、n（GridStructure）；不写死任何任务。

**实现**：StructuredGridProcessor 是 HF 的 LogitsProcessor，经适配器 generate_batch(allowed_token_ids=任务的允许集合,
logits_processor=[它]) 传入（适配器把 allowed 的处理器排在最前，本处理器在它之后再逐位置收窄；不改 src/adapters/）。贪心
（do_sample=False）与受限采样（do_sample=True，温度在 −inf 之后作用，不改变允许集合）都适用。一个实例只用于一次 generate：
第一次被调用时记下 input_ids 的宽度（提示的宽度，左填充后整批相同），之后 t = 宽度 − 它（同 src/e1/forced.GridStructureProcessor）。

**与受限解码的关系**：不加结构约束时，generate_batch(allowed_token_ids) 与 adapter.generate_restricted 走两条计算路径（前者是 HF
generate 的全词表 logits 再屏蔽，后者是 restricted_logits 只算允许集合的行，fp32）。CPU fp32 上两者逐 token 相同（tests 钉住，含挂一个
"不收窄"的处理器）；bf16 上在近似平局处可能分岔，差异的读数见每个检查点合并核对的 restricted 一节
（by_thinking.thinking_off.decode_vs_generate_batch；RUN-0038 的 2B E3b：exact_rows 0.955、token_agreement 0.981）。所以结构强制的
结果 ≠ "受限解码的结果再加结构"逐位相同，只是同一套允许集合上的贪心；新旧臂之间的比较一律用同一种解码（D83）。

**入口断言**：check_structured_ids 核对每行恰好 length 个 token、逐位置都在允许集合里、以停止 token 收尾（有停止位置时）——
处理器没生效就报 StructuredDecodeError，不当成模型的输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:  # 与适配器相同的基类（HF generate 的处理器列表只认它）
    from transformers import LogitsProcessor as _LogitsProcessorBase
except ImportError:  # pragma: no cover — 开发机与服务器都装了 transformers
    _LogitsProcessorBase = object

STRUCTURED_RULE = ("结构强制（D83，DEV-0041）：第 r 行的格子位置只允许第 r 行的值 token，行末只允许行分隔，最后一个位置只允许停止 "
                   "token；长度恒为 n(n + 1) + 1；只强制网格的形状，不注入任务知识。经 generate_batch(allowed_token_ids = 任务的允许集合, "
                   "logits_processor = [StructuredGridProcessor]) 实现（src/grid/structured_decode.py）")


class StructuredDecodeError(RuntimeError):
    """结构强制的输出不合约定（长度、逐位置的允许集合、停止）：处理器没有生效，或调用方式不对。"""


def _ids(xs: Sequence[Any], what: str) -> tuple[int, ...]:
    if isinstance(xs, (str, bytes)) or torch.is_tensor(xs):
        raise ValueError(f"{what} 必须是整数序列")
    out = tuple(int(x) for x in xs)
    if any(isinstance(x, bool) for x in xs):
        raise ValueError(f"{what} 不能含 bool")
    return out


@dataclass(frozen=True)
class GridStructure:
    """网格的形状：n 行 × n 格，每行的值 token 集合（value_ids[r]），行分隔 id，停止 id（None = 没有停止位置）。"""

    n: int
    value_ids: tuple[tuple[int, ...], ...]
    row_id: int
    stop_id: int | None

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"n 必须是正整数，收到 {self.n!r}")
        if len(self.value_ids) != self.n:
            raise ValueError(f"value_ids 要有 n = {self.n} 行，收到 {len(self.value_ids)} 行")
        special = {int(self.row_id)} | ({int(self.stop_id)} if self.stop_id is not None else set())
        if self.stop_id is not None and int(self.stop_id) == int(self.row_id):
            raise ValueError("行分隔与停止 token 不能是同一个 id")
        for r, vals in enumerate(self.value_ids):
            if not vals:
                raise ValueError(f"第 {r} 行的值 token 集合是空的")
            if len(set(vals)) != len(vals):
                raise ValueError(f"第 {r} 行的值 token 有重复")
            if special & set(vals):
                raise ValueError(f"第 {r} 行的值 token 与行分隔 / 停止 token 重叠：{sorted(special & set(vals))}")

    @classmethod
    def uniform(cls, value_ids: Sequence[int], row_id: int, stop_id: int | None, n: int) -> "GridStructure":
        """各行的值 token 集合相同（E3b 的分割、深度）。"""
        vals = _ids(value_ids, "value_ids")
        return cls(n=int(n), value_ids=tuple(vals for _ in range(int(n))), row_id=int(row_id),
                   stop_id=None if stop_id is None else int(stop_id))

    @classmethod
    def per_row(cls, value_ids: Sequence[Sequence[int]], row_id: int, stop_id: int | None) -> "GridStructure":
        """逐行给值 token 集合（行数就是 n）。"""
        rows = tuple(_ids(v, f"value_ids[{r}]") for r, v in enumerate(value_ids))
        return cls(n=len(rows), value_ids=rows, row_id=int(row_id), stop_id=None if stop_id is None else int(stop_id))

    @property
    def length(self) -> int:
        """生成的 token 数：n(n + 1)，有停止位置时 + 1。"""
        return self.n * (self.n + 1) + (0 if self.stop_id is None else 1)

    def kind_at(self, t: int) -> tuple[str, int]:
        """第 t 个位置的种类：("cell", 行号)、("row", 行号) 或 ("stop", n)。越界报错。"""
        if not 0 <= t < self.length:
            raise StructuredDecodeError(f"位置 t = {t} 超出结构的长度 {self.length}")
        r, c = divmod(int(t), self.n + 1)
        if r == self.n:
            return "stop", r
        return ("row", r) if c == self.n else ("cell", r)

    def allowed_at(self, t: int) -> tuple[int, ...]:
        kind, r = self.kind_at(t)
        if kind == "cell":
            return self.value_ids[r]
        if kind == "row":
            return (int(self.row_id),)
        return (int(self.stop_id),)

    def all_ids(self) -> list[int]:
        """结构里出现的全部 token（升序）：调用方传给 generate_batch 的 allowed_token_ids 必须包含它们。"""
        s = {int(self.row_id)} | {int(v) for vals in self.value_ids for v in vals}
        if self.stop_id is not None:
            s.add(int(self.stop_id))
        return sorted(s)

    def as_dict(self) -> dict[str, Any]:
        """写进运行身份（各行相同时只列一次）。"""
        same = all(v == self.value_ids[0] for v in self.value_ids)
        return {"n": self.n, "length": self.length, "row_id": int(self.row_id),
                "stop_id": None if self.stop_id is None else int(self.stop_id),
                "value_ids": list(self.value_ids[0]) if same else [list(v) for v in self.value_ids],
                "value_ids_per_row": not same, "rule": STRUCTURED_RULE}


class StructuredGridProcessor(_LogitsProcessorBase):
    """逐位置的允许集合（GridStructure.allowed_at）；其他 token 的 logit 置 −inf。不改模型，只改送进 argmax / 采样的 scores。
    一个实例只用于一次 generate（每批新建一个）。"""

    def __init__(self, structure: GridStructure):
        if not isinstance(structure, GridStructure):
            raise TypeError(f"structure 必须是 GridStructure，收到 {type(structure).__name__}")
        self.structure = structure
        self.start: int | None = None
        self.n_calls = 0
        self._masks: dict[tuple[str, int], torch.Tensor] = {}
        self._uniform = all(v == structure.value_ids[0] for v in structure.value_ids)

    def _keep(self, t: int, width: int, device: torch.device) -> torch.Tensor:
        kind, r = self.structure.kind_at(t)
        key = (kind, r if kind == "cell" and not self._uniform else -1)
        m = self._masks.get(key)
        if m is None or m.device != device or m.shape[0] != width:
            ids = self.structure.allowed_at(t)
            bad = [i for i in ids if not 0 <= i < width]
            if bad:
                raise StructuredDecodeError(f"允许集合里的 id {bad[:4]} 超出 logits 宽度 {width}")
            m = torch.zeros(width, dtype=torch.bool, device=device)
            m[torch.tensor(ids, dtype=torch.long, device=device)] = True
            self._masks[key] = m
        return m

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.start is None:
            self.start = int(input_ids.shape[1])
        t = int(input_ids.shape[1]) - self.start
        if t != self.n_calls:
            raise StructuredDecodeError(f"处理器第 {self.n_calls} 次调用时 t = {t}：一个实例只能用于一次 generate")
        self.n_calls += 1
        keep = self._keep(t, int(scores.shape[-1]), scores.device)
        return scores.masked_fill(~keep, float("-inf"))


def check_structured_ids(token_ids: Sequence[int], structure: GridStructure, *, stopped: bool | None = None,
                         where: str = "") -> None:
    """结构强制输出的入口断言：恰好 length 个 token，逐位置都在允许集合里；有停止位置时 stopped 必须是 True（给了的话）。"""
    ids = [int(i) for i in token_ids]
    if len(ids) != structure.length:
        raise StructuredDecodeError(f"{where}：结构强制应恰好生成 {structure.length} 个 token，得到 {len(ids)} 个（stopped={stopped}）")
    wrong = [t for t, i in enumerate(ids) if i not in structure.allowed_at(t)]
    if wrong:
        t = wrong[0]
        raise StructuredDecodeError(f"{where}：第 {t} 个 token {ids[t]} 不在该位置（{structure.kind_at(t)}）的允许集合里，"
                                    f"共 {len(wrong)} 处")
    if stopped is not None and structure.stop_id is not None and stopped is not True:
        raise StructuredDecodeError(f"{where}：以停止 token 收尾却没有记为 stopped（{stopped}）")


__all__ = [
    "GridStructure",
    "STRUCTURED_RULE",
    "StructuredDecodeError",
    "StructuredGridProcessor",
    "check_structured_ids",
]
