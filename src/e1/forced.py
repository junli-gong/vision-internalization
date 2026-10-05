"""格式强制诊断（E1 预注册 §10.2，D66-2；DEV-0033）：多边形预填、网格受限解码（固定长度；主读法与结构强制）。

口径照 configs/e1_posthoc.json 的 format_forced（规则文字在配置里，这里实现）。三种读法（variant）：
- prefill（refexp_poly）：提示不变，作答开头预填 "[["：adapter.generate_batch(answer_prefix="[[")（适配器把预填接在助手头之后，作为输入、
  不计入生成；text = split_answer(预填 + 生成)，即"解析时拼上预填"），上限同 E1 多边形；解析与截断规则同 E1。
- restricted（grid16，主读法）：允许集合只有 0、1、换行三个 token（grid_format_ids，不加停止 token），固定生成 GRID_FIXED_TOKENS = 271
  个 token。这三个 token 的构造只在本模块的 grid_format_ids 写一份，被 scripts/check_training_interfaces.py import（它的 grid16 允许集合
  = 这三个 + 本家族的停止 token；D68-4(b)、D70-5）。
- structured（grid16，补充）：在 restricted 之上用 GridStructureProcessor（HF 的 LogitsProcessor，经 generate_batch 的 logits_processor 传入）
  逐位置限制：生成的第 t 个 token（t 从 0 起）在 t mod 17 = 16 时只允许换行，其余只允许 0、1；必然是合法的 16×16 网格。

固定长度的两种网格读法不判截断（fixed_length_rule）：记录的 truncated 记 False、另记 fixed_length；解析时把结果的 stopped 置为 None
（parse_generation 只判思考段未闭合与空回答，再交给 parse_grid_text）。入口断言：每行恰好 271 个 token、都在允许集合里；structured 另核对
换行恰在 t mod 17 = 16 处——处理器没生效就报错，不当成模型的输出。

本模块只有纯函数与处理器（不碰文件）；运行（批、续跑、输出）在 src/e1/posthoc_runner.py。
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.constants import GRID_SIZE_E1
from src.e1.posthoc import GRID_FIXED_TOKENS, GRID_FORMAT_TEXTS, GRID_NEWLINE_PERIOD
from src.grid.codec import grid_to_text

try:  # 与适配器相同的基类（HF generate 的处理器列表只认它）
    from transformers import LogitsProcessor as _LogitsProcessorBase
except ImportError:  # pragma: no cover — 开发机与服务器都装了 transformers
    _LogitsProcessorBase = object


class ForcedOutputError(RuntimeError):
    """格式强制读法的输出不合约定（长度、允许集合、换行位置）：处理器没有生效。"""


def grid_format_ids(tokenizer: Any) -> dict[str, Any]:
    """数字网格格式的 0、1、换行三个 token（不含停止 token）：每个字符编码恰为一个 token；用本家族分词器编码一张 16×16 的示例网格
    （src/grid/codec.grid_to_text），用到的 token 必须恰好是这三个。约定只在这里写一份，被 scripts/check_training_interfaces.grid_allowed
    import（那边另加停止 token；D68-4(b)、D70-5），tests 核对两边的 id 相同。返回值里的 rule 文字是运行身份的一部分（格式强制的
    config.yaml 记它），保持原样。"""

    def enc(s: str) -> list[int]:
        return [int(i) for i in tokenizer(s, add_special_tokens=False)["input_ids"]]

    probe = np.zeros((GRID_SIZE_E1, GRID_SIZE_E1), dtype=bool)
    probe[GRID_SIZE_E1 // 4: GRID_SIZE_E1 // 2, GRID_SIZE_E1 // 4: 3 * GRID_SIZE_E1 // 4] = True
    text = grid_to_text(probe)
    rows = []
    for role, s in zip(("digit_0", "digit_1", "newline"), GRID_FORMAT_TEXTS):
        ids = enc(s)
        if len(ids) != 1:
            raise RuntimeError(f"{s!r} 在本家族的 tokenizer 里不是单个 token：{ids}")
        rows.append({"role": role, "text": s, "id": ids[0], "token": tokenizer.convert_ids_to_tokens(ids[0])})
    fmt = {r["id"] for r in rows}
    if len(fmt) != len(rows):
        raise RuntimeError(f"0、1、换行撞到了同一个 token：{rows}")
    probe_ids = enc(text)
    if set(probe_ids) != fmt:
        raise RuntimeError(f"示例网格的编码用到了 {sorted(set(probe_ids))}，不只是 0、1、换行 {sorted(fmt)}")
    return {"ids": sorted(fmt), "digit_ids": [rows[0]["id"], rows[1]["id"]], "newline_id": rows[2]["id"], "tokens": rows,
            "probe": {"grid_size": GRID_SIZE_E1, "n_tokens": len(probe_ids), "distinct_ids": sorted(set(probe_ids))},
            "rule": "同 scripts/check_training_interfaces.grid_allowed 的构造（编码 16×16 示例网格必须恰好由 0、1、换行三个单 token 组成），"
                    "不加停止 token"}


def newline_position(t: int, period: int = GRID_NEWLINE_PERIOD) -> bool:
    """生成的第 t 个 token（t 从 0 起）是否是每行第 period 个位置（只允许换行）。"""
    return t % period == period - 1


class GridStructureProcessor(_LogitsProcessorBase):
    """结构强制（structured）的逐位置允许集合：t mod period = period − 1 处只留换行，其余只留两个数字；其他 token 的 logit 置 −inf。

    t = 当前已生成的 token 数：第一次被调用时记下 input_ids 的宽度（提示的宽度，左填充后整批相同），之后 t = 宽度 − 它。所以一个实例
    只用于一次 generate（每批新建一个）。不改模型，只改送进 argmax 的 scores。"""

    def __init__(self, digit_ids: Sequence[int], newline_id: int, *, period: int = GRID_NEWLINE_PERIOD):
        self.digits = [int(i) for i in digit_ids]
        self.newline = int(newline_id)
        if len(self.digits) != 2 or self.newline in self.digits:
            raise ValueError(f"要两个数字 token 与一个不同的换行 token，收到 {digit_ids}、{newline_id}")
        self.period = int(period)
        self.start: int | None = None
        self.n_calls = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.start is None:
            self.start = int(input_ids.shape[1])
        t = int(input_ids.shape[1]) - self.start
        if t != self.n_calls:
            raise ForcedOutputError(f"处理器第 {self.n_calls} 次调用时 t = {t}：一个实例只能用于一次 generate")
        self.n_calls += 1
        ids = [self.newline] if newline_position(t, self.period) else self.digits
        keep = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        keep[torch.tensor(ids, dtype=torch.long, device=scores.device)] = True
        return scores.masked_fill(~keep, float("-inf"))


def check_fixed_output(res: Any, *, variant: str, fmt: Mapping[str, Any], n_tokens: int = GRID_FIXED_TOKENS,
                       period: int = GRID_NEWLINE_PERIOD, where: str = "") -> None:
    """固定长度网格读法的入口断言：恰好 n_tokens 个 token、没有停止、都在 0/1/换行里；structured 另核对换行的位置。"""
    ids = [int(i) for i in res.token_ids]
    if len(ids) != n_tokens or res.stopped is not False:
        raise ForcedOutputError(f"{where}：{variant} 应恰好生成 {n_tokens} 个 token 且不停止，得到 {len(ids)} 个、stopped={res.stopped}")
    allowed = set(fmt["ids"])
    bad = sorted({i for i in ids if i not in allowed})
    if bad:
        raise ForcedOutputError(f"{where}：{variant} 生成了允许集合 {sorted(allowed)} 以外的 token {bad[:8]}")
    if variant == "structured":
        nl = int(fmt["newline_id"])
        wrong = [t for t, i in enumerate(ids) if (i == nl) != newline_position(t, period)]
        if wrong:
            raise ForcedOutputError(f"{where}：structured 的换行不在每行第 {period} 个位置（t = {wrong[:8]}）")


def parse_view(res: Any, *, fixed_length: bool) -> Any:
    """交给解析的结果：固定长度的读法把 stopped 置为 None（不判截断，见模块说明）；其余原样。"""
    return dataclasses.replace(res, stopped=None) if fixed_length else res


__all__ = [
    "ForcedOutputError",
    "GridStructureProcessor",
    "check_fixed_output",
    "grid_format_ids",
    "newline_position",
    "parse_view",
]
