"""E3b 像素 token 通道的网格词表（DEV-0025 §1；简报 §4 第 4 条、§7.2、§8.5）。

- 词表：二值 2 个（mask 的 0 / 1）、深度 64 档（src/constants.DEPTH_NUM_BINS）、行分隔 1 个，共 67 个新 token。名字写在
  configs/e3b_train.json 的 vocab 里（深度档用模板展开），这里只读它；名字过 base.check_grid_token_names，个数与常量核对。
- 预算（简报 §4 第 4 条、§7.2）：本词表 + 配置里登记的计划追加（ADE20K 的 ADE_NUM_CLASSES 个类别）累计不超过 src/constants.MAX_NEW_GRID_TOKENS；
  适配器的 add_grid_tokens 另按"跨调用累计"再核一次（D10）。
- 加进模型：GridTokens = add_vocab(adapter, vocab)，按 vocab.names 的顺序调一次 add_grid_tokens；id 由适配器的分配规则决定
  （Qwen / InternVL 复用预留行，Gemma 复用 <unusedN>，见各适配器）。token_table() 由训练器记进运行身份。检查点恢复之后用
  attach_vocab(adapter, vocab) 取回同一份布局（restore_tokens 已按表登记，id 必须与名字一一对应）。
- 停止 token：本家族的回合结束 token = 适配器停止 token 列表（_stop_ids()）的第一个（Qwen / InternVL 是 <|im_end|>，Gemma 是
  <turn|>）。适配器没有公开接口，这里读私有钩子，接口需求写进 DEV-0025 的完成总结。
- 受限 logits 的允许集合（allowed_ids）：scope = "task"（默认）时是本任务的值 token（分割 2 个、深度 64 个）+ 行分隔 + 停止；
  scope = "all" 时是全部 67 个 grid token + 停止。训练损失与受限解码用同一个函数给出的同一个集合（升序）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.adapters.base import check_grid_token_names
from src.constants import DEPTH_NUM_BINS, GRID_SIZE_E3B, MAX_NEW_GRID_TOKENS

GRID_TASKS: tuple[str, ...] = ("refseg", "depth")  # 有网格目标的任务；值 token 的种类：refseg → mask，depth → depth
N_MASK_VALUES = 2  # 二值网格：0 = 背景、1 = 前景（src/grid/codec.mask_to_grid 的取值）
ALLOWED_SCOPES: tuple[str, ...] = ("task", "all")


class GridTokenError(ValueError):
    """词表配置不合规，或适配器里的 grid token 与词表对不上。"""


@dataclass(frozen=True)
class GridVocab:
    """网格词表（名字）。mask[v] 是二值网格取值 v 的 token；depth[k] 是深度档 k（src/grid/codec.depth_to_bins 的档号）；
    row 是行分隔。names = mask + depth + (row,)，即 add_grid_tokens 的顺序。"""

    mask: tuple[str, ...]
    depth: tuple[str, ...]
    row: str
    planned_extra: Mapping[str, int]

    def __post_init__(self) -> None:
        if len(self.mask) != N_MASK_VALUES:
            raise GridTokenError(f"mask 词表应有 {N_MASK_VALUES} 个 token，收到 {len(self.mask)}")
        if len(self.depth) != DEPTH_NUM_BINS:
            raise GridTokenError(f"depth 词表应有 DEPTH_NUM_BINS = {DEPTH_NUM_BINS} 个 token，收到 {len(self.depth)}")
        try:
            check_grid_token_names(list(self.names))
        except ValueError as e:
            raise GridTokenError(f"网格词表的名字不合规：{e}") from None
        extra = sum(int(v) for v in self.planned_extra.values())
        if any(int(v) < 0 for v in self.planned_extra.values()):
            raise GridTokenError(f"planned_extra 不能为负：{dict(self.planned_extra)}")
        if len(self.names) + extra > MAX_NEW_GRID_TOKENS:
            raise GridTokenError(f"网格词表 {len(self.names)} 个 + 计划追加 {extra} 个 = {len(self.names) + extra}，"
                                 f"超过上限 MAX_NEW_GRID_TOKENS = {MAX_NEW_GRID_TOKENS}（简报 §4 第 4 条）")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.mask) + tuple(self.depth) + (self.row,)

    def value_names(self, task: str) -> tuple[str, ...]:
        """任务的值 token（网格格子里出现的 token），按取值 / 档号的顺序。"""
        if task == "refseg":
            return tuple(self.mask)
        if task == "depth":
            return tuple(self.depth)
        raise GridTokenError(f"任务 {task!r} 没有网格目标（有：{GRID_TASKS}）")

    def budget(self) -> dict[str, int]:
        """预算核对的记录：本词表、计划追加、累计、上限。"""
        extra = sum(int(v) for v in self.planned_extra.values())
        return {"n_vocab": len(self.names), "n_planned_extra": extra, "n_total_planned": len(self.names) + extra,
                "limit": MAX_NEW_GRID_TOKENS}


def vocab_from_config(doc: Mapping[str, Any]) -> GridVocab:
    """configs/e3b_train.json 的 vocab 节 → GridVocab。depth_template 用 str.format(k=档号) 展开。"""
    v = doc["vocab"]
    tmpl = str(v["depth_template"])
    depth = tuple(tmpl.format(k=k) for k in range(DEPTH_NUM_BINS))
    return GridVocab(mask=tuple(v["mask"]), depth=depth, row=str(v["row"]), planned_extra=dict(v.get("planned_extra") or {}))


def check_grid_size(n: int) -> int:
    """网格边长必须是 src/constants.GRID_SIZE_E3B（简报 §7.2 默认 32×32；配置里写的值与常量逐数相同，测试钉住）。"""
    if isinstance(n, bool) or int(n) != n or int(n) != GRID_SIZE_E3B:
        raise GridTokenError(f"网格边长 {n!r} 与 GRID_SIZE_E3B = {GRID_SIZE_E3B} 不同")
    return int(n)


def family_stop_id(adapter: Any) -> int:
    """本家族的回合结束 token（目标序列的最后一个 token）：适配器停止 token 列表的第一个。"""
    ids = list(adapter._stop_ids())
    if not ids:
        raise GridTokenError("适配器没有停止 token")
    return int(ids[0])


@dataclass(frozen=True)
class GridTokens:
    """词表加进某个适配器之后的 id 布局。"""

    vocab: GridVocab
    ids: Mapping[str, int]
    stop_id: int

    def __post_init__(self) -> None:
        names = self.vocab.names
        if set(self.ids) != set(names):
            raise GridTokenError(f"id 表的名字与词表不同：多 {sorted(set(self.ids) - set(names))[:3]}，"
                                 f"少 {sorted(set(names) - set(self.ids))[:3]}")
        vals = [int(self.ids[n]) for n in names]
        if len(set(vals)) != len(vals):
            raise GridTokenError("不同的 grid token 拿到了同一个 id")
        if int(self.stop_id) in vals:
            raise GridTokenError(f"停止 token {self.stop_id} 与某个 grid token 的 id 相同")

    def id_of(self, name: str) -> int:
        return int(self.ids[name])

    @property
    def row_id(self) -> int:
        return self.id_of(self.vocab.row)

    @property
    def grid_ids(self) -> tuple[int, ...]:
        """全部 67 个 grid token 的 id（按 vocab.names 的顺序）。"""
        return tuple(self.id_of(n) for n in self.vocab.names)

    def value_ids(self, task: str) -> tuple[int, ...]:
        """任务的值 token 的 id：下标就是网格取值（二值 0 / 1、深度档号 0..63）。"""
        return tuple(self.id_of(n) for n in self.vocab.value_names(task))

    def allowed_ids(self, task: str, scope: str) -> list[int]:
        """受限 logits 的允许集合（升序）：训练损失与受限解码都用它（规则见模块说明）。"""
        if scope not in ALLOWED_SCOPES:
            raise GridTokenError(f"allowed scope 必须是 {ALLOWED_SCOPES} 之一，收到 {scope!r}")
        base = self.value_ids(task) + (self.row_id,) if scope == "task" else self.grid_ids
        return sorted(set(base) | {int(self.stop_id)})

    def layout(self) -> dict[str, Any]:
        """写进运行身份与完成总结的 id 布局（名字 → id；连续区间的概括）。"""
        def runs(ids: Sequence[int]) -> list[list[int]]:
            out: list[list[int]] = []
            for i in ids:
                if out and i == out[-1][1] + 1:
                    out[-1][1] = i
                else:
                    out.append([i, i])
            return out

        return {"mask": {n: self.id_of(n) for n in self.vocab.mask},
                "depth_first": self.id_of(self.vocab.depth[0]), "depth_last": self.id_of(self.vocab.depth[-1]),
                "depth_contiguous": runs(self.value_ids("depth")) == [[self.id_of(self.vocab.depth[0]),
                                                                      self.id_of(self.vocab.depth[-1])]],
                "row": self.row_id, "stop": int(self.stop_id), "grid_id_runs": runs(sorted(self.grid_ids)),
                "n_grid_tokens": len(self.vocab.names)}


def add_vocab(adapter: Any, vocab: GridVocab) -> GridTokens:
    """把词表加进适配器（必须在 inject_lora 之前；顺序约束由适配器检查）。"""
    ids = adapter.add_grid_tokens(list(vocab.names))
    return GridTokens(vocab=vocab, ids={n: int(ids[n]) for n in vocab.names}, stop_id=family_stop_id(adapter))


def attach_vocab(adapter: Any, vocab: GridVocab) -> GridTokens:
    """检查点恢复（restore_tokens）之后取回布局：适配器登记的 grid token 必须恰好是这份词表。"""
    have = dict(adapter.grid_token_ids)
    if set(have) != set(vocab.names):
        raise GridTokenError(f"适配器的 grid token（{len(have)} 个）与词表（{len(vocab.names)} 个）不同")
    return GridTokens(vocab=vocab, ids={n: int(have[n]) for n in vocab.names}, stop_id=family_stop_id(adapter))


__all__ = [
    "ALLOWED_SCOPES",
    "GRID_TASKS",
    "GridTokenError",
    "GridTokens",
    "GridVocab",
    "N_MASK_VALUES",
    "add_vocab",
    "attach_vocab",
    "check_grid_size",
    "family_stop_id",
    "vocab_from_config",
]
