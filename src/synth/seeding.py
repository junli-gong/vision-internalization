"""种子派生：样本 = 纯函数(任务, 划分, 序号, 该任务的生成器版本)。

- 场景种子 scene_seed(split, index) = SPLIT_CODES[split] * SCENE_SEED_STRIDE + index。三个划分的场景种子
  落在互不相交的整数区间里（train [0, S)、val [S, 2S)、test [2S, 3S)，S = 2^48），按构造不相交。
- 每个样本的随机数生成器 = PCG64(SeedSequence(entropy=[根, 版本, 任务码, 场景种子]))；
  版本默认是该任务的 TASK_GENERATOR_VERSIONS[任务]（DEV-0010 起按任务记，c、d、e 仍是 2，种子不变）。
  任务码是 constants 里的固定整数，不用 hash()（字符串哈希每个进程随机化）；不读全局随机状态、不读墙钟。
- 生成器内只用 Generator.random() 与 Generator.integers()，其余分布都在它们之上用初等运算构造，
  避免依赖 numpy 各版本之间可能变化的分布算法。
"""

from __future__ import annotations

import numpy as np

from src.synth.constants import (
    SCENE_SEED_STRIDE,
    SPLIT_CODES,
    SPLITS,
    SYNTH_SEED_ROOT,
    TASK_ALIASES,
    TASK_CODES,
    TASK_GENERATOR_VERSIONS,
    TASKS,
)


def canonical_task(task: str) -> str:
    """任务名或字母（a–e）→ 规范任务名。"""
    t = TASK_ALIASES.get(task, task)
    if t not in TASKS:
        raise ValueError(f"未知任务 {task!r}；可选 {TASKS} 或字母 {tuple(TASK_ALIASES)}")
    return t


def check_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"未知划分 {split!r}；可选 {SPLITS}")
    return split


def check_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError(f"序号必须是整数，收到 {type(index).__name__}")
    i = int(index)
    if not 0 <= i < SCENE_SEED_STRIDE:
        raise ValueError(f"序号必须在 [0, {SCENE_SEED_STRIDE}) 内，收到 {i}")
    return i


def scene_seed(split: str, index: int) -> int:
    """划分内序号 → 全局场景种子；不同划分的场景种子互不相交。"""
    return SPLIT_CODES[check_split(split)] * SCENE_SEED_STRIDE + check_index(index)


def task_version(task: str) -> int:
    """该任务当前的生成器版本（TASK_GENERATOR_VERSIONS）。"""
    return TASK_GENERATOR_VERSIONS[canonical_task(task)]


def seed_entropy(task: str, split: str, index: int, version: int | None = None) -> list[int]:
    """进入 SeedSequence 的 entropy：[根, 版本, 任务码, 场景种子]。version 默认是该任务的当前版本。"""
    t = canonical_task(task)
    v = task_version(t) if version is None else int(version)
    return [SYNTH_SEED_ROOT, v, TASK_CODES[t], scene_seed(split, index)]


def sample_rng(task: str, split: str, index: int, version: int | None = None) -> np.random.Generator:
    """该样本专用的随机数生成器（PCG64）。同一输入在任何机器、任何进程里给出同一个流。"""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_entropy(task, split, index, version))))


# ---------------------------------------------------------------------------
# 只建立在 random() 与 integers() 上的抽样小工具
# ---------------------------------------------------------------------------


def uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng.random())


def randint(rng: np.random.Generator, lo: int, hi_inclusive: int) -> int:
    """[lo, hi_inclusive] 上的均匀整数。"""
    return int(rng.integers(lo, hi_inclusive + 1))


def pick(rng: np.random.Generator, items):
    """从非空序列里均匀取一个。"""
    if len(items) == 0:
        raise ValueError("pick：序列为空")
    return items[int(rng.integers(len(items)))]


def coin(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)
