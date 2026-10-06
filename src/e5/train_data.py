"""E5 的训练数据（DEV-0037 §1）：从生成器的 train 划分即时取样，不落盘（N1）。

**训练集**（TrainSet；数字在 configs/e5_train.json 的 data 节）：
- 数据量档 N（五类合计，1K、4K、16K、64K，暂定）；每类 per_task = N / 5 条（等量）。只训练部分任务（tasks，例如 a–d）时每类条数不变、
  各类样本与五类全训的运行逐条相同（合计 per_task × 任务数）。
- 第 t 类的第 j 条（0 ≤ j < per_task）= src.synth.generate(t, "train", 序号, bin = 档)，序号 = 数据种子 × seed_stride + j，
  档 = j mod 档数（同评测集的轮转，档内均匀：各档条数至多差 1）。样本由 (数据种子, 任务, j) 唯一决定、无状态；
  小档的每一类都是大档同一类的前缀（同一个数据种子下嵌套）。不同数据种子的序号区间不相交（seed_stride = 2^32 ≥ 每类最大条数）。
- 划分：train 划分的场景种子落在 [0, 2^48)，与 val、test 不相交；物体身份只取 train 的身份池（D23-2；src/synth/seeding.py、
  shapes.identity_split_table、solids.solid_identity_split_table），tests 钉住。
- d（physics）的数据只在规范平台上生成（D24）：非规范平台上只允许 --tiny（src/e5/runner.check_platform，与评测同一个检查）。
- 训练集的清单 = 先按 tasks 的顺序（E5_TASKS 的 a–e 顺序）、再按 j 列出的 (任务, j)；清单的 sha256（items_digest）进运行身份。

**样本日程**（Schedule；无状态，续跑按步号重算）：固定 epoch（configs/e5_train.json 的 training.epochs_basis）。第 k 个训练样本
（k = 步 × 批大小 + 批内下标，0 ≤ k < 总条数）取第 e = k // 训练集条数 个 epoch 的置换
np.random.default_rng([ORDER_SALT, 运行种子, 训练集条数, e]).permutation(训练集条数) 的第 k mod 训练集条数 个位置。总条数 = epochs × 训练集
条数（最后一步可能不满）；步数下限生效时（DEV-0038，training.min_steps）总条数 = 步数 × 批大小，epoch 照样往后数（置换与总条数无关，
所以前 epochs 个 epoch 与不设下限时逐条相同）。

**一条训练样本**（ExampleBuilder.build → src/e3/data.Example，与 E3b 的训练器同一个接口）：图像（a–d 一张 PIL 图，e 没有）、提示
（src/e5/token_grid.PromptFormat：指令原文 + E3b 网格指令的结尾句，指令在前）、目标 token（src/e5/token_grid.E5Tokens：二值网格或起伏网格
→ E3b 的网格 token 序列，长度 1057）、逐 token 权重（全部为 1：所有格子、行分隔与停止 token 都监督）。生成器偶发的 SynthGenerationError
（拒绝采样超限）记为跳过（训练器把它当 NoSupervision 一样计数），确定的。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.e3.data import Example, SlotPlan
from src.e3.targets import NoSupervision
from src.e5.protocol import E5_TASKS, TASK_LETTER, num_bins, sample_id
from src.e5.runner import check_platform
from src.e5.token_grid import E5Tokens, PromptFormat, sample_images, target_values
from src.e5.train_config import MAX_DATA_SEED, E5TrainConfig, per_task
from src.synth import SynthGenerationError, generate
from src.synth.sample import Sample
from src.synth.seeding import canonical_task

ORDER_SALT = 20261005  # 样本顺序的随机数流（与数据种子、运行种子一起进 default_rng）；固定不变
SPLIT = "train"


def canonical_tasks(tasks: Sequence[str] | None) -> tuple[str, ...]:
    """任务子集 → 规范名，按 E5_TASKS（a–e）的顺序；None 或 ["all"] = 五类。"""
    if tasks is None or list(tasks) == ["all"]:
        return tuple(E5_TASKS)
    ts = {canonical_task(t) for t in tasks}
    if len(ts) != len(list(tasks)):
        raise ValueError(f"--tasks 有重复：{list(tasks)}")
    return tuple(t for t in E5_TASKS if t in ts)


def tasks_tag(tasks: Sequence[str]) -> str:
    """叶目录里的任务子集标记：五类为空串，否则是字母（如 abcd）。"""
    ts = canonical_tasks(tasks)
    return "" if ts == tuple(E5_TASKS) else "".join(TASK_LETTER[t] for t in ts)


@dataclass(frozen=True)
class TrainItem:
    task: str
    j: int
    index: int
    bin: int

    @property
    def id(self) -> str:
        return sample_id(self.task, SPLIT, self.index)


@dataclass(frozen=True)
class TrainSet:
    """一个数据量档 × 任务子集 × 数据种子的训练集（规则见模块说明）。"""

    n_train: int
    tasks: tuple[str, ...]
    data_seed: int
    seed_stride: int

    def __post_init__(self) -> None:
        if isinstance(self.data_seed, bool) or not isinstance(self.data_seed, int) or not 0 <= self.data_seed < MAX_DATA_SEED:
            raise ValueError(f"数据种子必须是 [0, {MAX_DATA_SEED}) 内的整数，收到 {self.data_seed!r}")
        if not self.tasks or tuple(canonical_tasks(self.tasks)) != tuple(self.tasks):
            raise ValueError(f"tasks 必须是规范名、按 a–e 的顺序且不重复，收到 {self.tasks}")
        if self.per_task > self.seed_stride:
            raise ValueError(f"每类 {self.per_task} 条超过 seed_stride {self.seed_stride}：不同数据种子的序号区间会相交")

    @classmethod
    def build(cls, cfg: E5TrainConfig, n_train: int, tasks: Sequence[str] | None, data_seed: int) -> "TrainSet":
        return cls(n_train=int(n_train), tasks=canonical_tasks(tasks), data_seed=int(data_seed), seed_stride=cfg.seed_stride)

    @property
    def per_task(self) -> int:
        return per_task(self.n_train)

    def __len__(self) -> int:
        return self.per_task * len(self.tasks)

    @property
    def tag(self) -> str:
        """叶目录名：n<数据量>，部分任务时加 _<字母>（如 n1000_abcd）。"""
        t = tasks_tag(self.tasks)
        return f"n{self.n_train}" + (f"_{t}" if t else "")

    def item_of(self, task: str, j: int) -> TrainItem:
        t = canonical_task(task)
        if not 0 <= int(j) < self.per_task:
            raise IndexError(f"{t} 的 j = {j} 不在 [0, {self.per_task})")
        return TrainItem(task=t, j=int(j), index=self.data_seed * self.seed_stride + int(j), bin=int(j) % num_bins(t))

    def item(self, pos: int) -> TrainItem:
        """清单的第 pos 个（先按任务、再按 j）。"""
        if not 0 <= int(pos) < len(self):
            raise IndexError(pos)
        return self.item_of(self.tasks[int(pos) // self.per_task], int(pos) % self.per_task)

    def items(self) -> list[TrainItem]:
        return [self.item(p) for p in range(len(self))]

    def digest(self) -> str:
        return items_digest(self.items())

    def identity(self) -> dict[str, Any]:
        return {"generator": "src.synth.generate", "split": SPLIT, "n_train": self.n_train, "tasks": list(self.tasks),
                "per_task": self.per_task, "n_examples": len(self), "data_seed": self.data_seed,
                "seed_stride": self.seed_stride, "items_sha256": self.digest(),
                "index_rule": "序号 = 数据种子 × seed_stride + j，档 = j mod 档数（src/e5/train_data.py）",
                "bins_per_task": {t: num_bins(t) for t in self.tasks}}


def items_digest(items: Sequence[TrainItem]) -> str:
    return hashlib.sha256("\n".join(f"{it.id}#{it.bin}" for it in items).encode("utf-8")).hexdigest()


def make_train_sample(it: TrainItem) -> Sample:
    return generate(it.task, SPLIT, it.index, bin=it.bin)


class Schedule:
    """固定 epoch 的样本日程（模块说明）。接口同 src/e3/data.Schedule：step_plans(step)。"""

    def __init__(self, ts: TrainSet, *, seed: int, epochs: int, batch_size: int, total: int | None = None) -> None:
        """total（DEV-0038，步数下限生效时 = 步数 × 批大小）：日程的总条数；不给时 = epochs × 训练集条数。给了 total 时 epoch 数
        = ceil(total / 训练集条数)（最后一个 epoch 可能只取置换的前一段）；第 e 个 epoch 的置换与 total 无关。"""
        self.ts = ts
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.n = len(ts)
        self.total = int(epochs) * self.n if total is None else int(total)
        if self.total < 1:
            raise ValueError(f"日程的总条数必须 ≥ 1，收到 {self.total}")
        self.epochs = -(-self.total // self.n)
        self._perm: dict[int, np.ndarray] = {}

    def permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm:
            self._perm[epoch] = np.random.default_rng([ORDER_SALT, self.seed, self.n, int(epoch)]).permutation(self.n)
        return self._perm[epoch]

    def plan(self, k: int) -> SlotPlan:
        if not 0 <= int(k) < self.total:
            raise IndexError(f"k = {k} 不在 [0, {self.total})")
        ep, pos = divmod(int(k), self.n)
        it = self.ts.item(int(self.permutation(ep)[pos]))
        return SlotPlan(k=int(k), step=int(k) // self.batch_size, task=it.task, j=it.j, index=it.index, epoch=ep,
                        sample_id=it.id, kind="grid")

    def step_plans(self, step: int) -> list[SlotPlan]:
        lo = int(step) * self.batch_size
        return [self.plan(k) for k in range(lo, min(lo + self.batch_size, self.total))]


class ExampleBuilder:
    """SlotPlan → Example（接口同 src/e3/data.ExampleBuilder.build）。"""

    def __init__(self, ts: TrainSet, *, tokens: E5Tokens, prompt: PromptFormat, n: int, tiny: bool) -> None:
        self.ts = ts
        self.tokens = tokens
        self.prompt = prompt
        self.n = int(n)
        self.tiny = bool(tiny)
        for t in ts.tasks:  # d 的数据只在规范平台上生成（D24）
            check_platform(t, self.tiny)

    def example(self, plan: SlotPlan, sample: Sample) -> Example:
        """一条生成好的样本 → Example（测试也用它直接构造）。"""
        v = target_values(sample)
        if v.shape != (self.n, self.n):
            raise ValueError(f"{plan.sample_id}：目标网格 {v.shape} 不是 {self.n}×{self.n}")
        ids = self.tokens.target_ids(sample.task, v, self.n)
        info = {"n_fg_cells": int(np.count_nonzero(v)), "bin": int(sample.meta["difficulty"]["bin"])}
        return Example(plan, sample_images(sample), self.prompt.render(sample, self.n), self.prompt.order, ids, [1] * len(ids),
                       info=info)

    def build(self, plan: SlotPlan) -> Example:
        it = self.ts.item_of(plan.task, plan.j)
        if it.id != plan.sample_id:
            raise RuntimeError(f"日程与训练集对不上：{plan.sample_id} ≠ {it.id}")
        try:
            sample = generate(it.task, SPLIT, it.index, bin=it.bin, grid_size=self.n)
        except SynthGenerationError as e:  # 拒绝采样超限：确定的，记为跳过（训练器按 NoSupervision 计数）
            raise NoSupervision(f"{it.id}：生成器没有得到合格样本（{e}）") from None
        return self.example(plan, sample)


__all__ = ["ExampleBuilder", "ORDER_SALT", "SPLIT", "Schedule", "TrainItem", "TrainSet", "canonical_tasks", "items_digest",
           "make_train_sample", "tasks_tag"]
