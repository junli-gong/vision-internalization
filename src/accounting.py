"""计算记账（简报 §4 第 11 条）。

每次模型调用记一条 Compute；同一样本的多次调用相加，再用 as_sample(wall_time_s=…) 标记为一个样本；
样本之间再相加，得到一次运行的总量，写成 compute.json。

两种时间：
- call_time_s：逐次调用耗时之和（每次 generate 自己的计时）。并发调用时它大于实际墙钟。
- wall_time_s：实际墙钟，只在样本级由调用方给出（as_sample 的必填参数）；调用级记录恒为 0。
  例：E3c 的 4 个块并发各 2 s，call_time_s = 8，wall_time_s ≈ 2。
  多个样本相加时 wall_time_s 是各样本墙钟之和（样本间也并发时它仍大于整次运行的墙钟）。

轮数：一次调用若开启新一轮（叠加反馈的下一轮、单轮流程的唯一一轮）记 new_round=True；
同一轮内的其他调用（如分块）记 new_round=False。单轮流程里 n_rounds == n_calls。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

COMPUTE_SCHEMA_VERSION = 2

_INT_FIELDS = ("n_samples", "n_calls", "n_rounds", "generated_tokens", "prompt_tokens", "visual_tokens")
_FLOAT_FIELDS = ("call_time_s", "wall_time_s")
# compute.json 的字段名，固定；改动须升 COMPUTE_SCHEMA_VERSION
COMPUTE_JSON_FIELDS = ("schema_version",) + _INT_FIELDS + _FLOAT_FIELDS + ("per_sample_mean",)


@dataclass(frozen=True)
class Compute:
    n_samples: int = 0
    n_calls: int = 0
    n_rounds: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    visual_tokens: int = 0
    call_time_s: float = 0.0
    wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} 必须是非负整数，收到 {v!r}")
        for name in _FLOAT_FIELDS:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} 必须是非负数，收到 {v!r}")
        if self.n_samples == 0 and self.wall_time_s != 0:
            raise ValueError("wall_time_s 只能在样本级给出（as_sample）")

    @classmethod
    def call(
        cls, *, generated_tokens: int, prompt_tokens: int, visual_tokens: int, call_time_s: float, new_round: bool
    ) -> "Compute":
        """一次模型调用。call_time_s 是这次调用自己的耗时。"""
        return cls(
            n_calls=1,
            n_rounds=int(new_round),
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            visual_tokens=visual_tokens,
            call_time_s=float(call_time_s),
        )

    @classmethod
    def from_generation(cls, result: Any, *, new_round: bool) -> "Compute":
        """从 src.adapters.base.GenerationResult 记一次调用（它的 wall_time_s 是单次调用耗时，记入 call_time_s）。"""
        return cls.call(
            generated_tokens=result.n_generated_tokens,
            prompt_tokens=result.n_prompt_tokens,
            visual_tokens=result.n_visual_tokens,
            call_time_s=result.wall_time_s,
            new_round=new_round,
        )

    def __add__(self, other: "Compute") -> "Compute":
        if not isinstance(other, Compute):
            return NotImplemented
        return Compute(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def as_sample(self, *, wall_time_s: float) -> "Compute":
        """把一个样本的调用总和标记为 1 个样本，并记下这个样本的实际墙钟。已含样本计数的记录不能再标记。"""
        if self.n_samples != 0:
            raise ValueError("该记录已含样本计数，不能再次 as_sample")
        return Compute(**{**asdict(self), "n_samples": 1, "wall_time_s": float(wall_time_s)})

    def per_sample_mean(self) -> dict[str, float] | None:
        if self.n_samples == 0:
            return None
        return {name: getattr(self, name) / self.n_samples for name in _INT_FIELDS[1:] + _FLOAT_FIELDS}

    def to_json_dict(self) -> dict[str, Any]:
        d = {"schema_version": COMPUTE_SCHEMA_VERSION, **asdict(self), "per_sample_mean": self.per_sample_mean()}
        if tuple(d) != COMPUTE_JSON_FIELDS:
            raise RuntimeError(f"compute.json 字段 {tuple(d)} 偏离固定字段 {COMPUTE_JSON_FIELDS}")
        return d


def merge(records: Iterable[Compute]) -> Compute:
    """多条记录相加。空迭代返回全零记录。"""
    total = Compute()
    for r in records:
        total = total + r
    return total


def write_compute_json(path: str | Path, compute: Compute) -> None:
    Path(path).write_text(json.dumps(compute.to_json_dict(), indent=2, ensure_ascii=False) + "\n")


def read_compute_json(path: str | Path) -> Compute:
    d = json.loads(Path(path).read_text())
    if d.get("schema_version") != COMPUTE_SCHEMA_VERSION:
        raise ValueError(f"compute.json schema_version={d.get('schema_version')}，期望 {COMPUTE_SCHEMA_VERSION}")
    if tuple(d) != COMPUTE_JSON_FIELDS:
        raise ValueError(f"compute.json 字段 {tuple(d)} 与固定字段 {COMPUTE_JSON_FIELDS} 不一致")
    return Compute(**{k: d[k] for k in _INT_FIELDS + _FLOAT_FIELDS})
