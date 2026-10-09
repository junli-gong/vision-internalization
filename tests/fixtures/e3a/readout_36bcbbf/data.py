"""E3a 的训练样本（DEV-0034 §2）：训练池、无状态的样本日程、样本构造。只调用 src/data、src/e3/data 的函数，不改它们。

**训练池**（硬约束 7：没有去重剔除清单就报错，由 src/data/train_loaders 负责；开跑前另核对剔除清单比较过 dedupe.required_eval_sets，
src/e3/data.check_dedupe 的同一规则）：
- refseg、depth：与 E3b 同一个训练池（src/e3/data.load_pool：RefCOCO 家族三者的 train 句子；NYU 深度 BTS 清单）；
- semseg：ade20k/train（load_ade20k）；normals：nyu_normals/train（load_nyu_normals）。
--tiny：refseg 用 tests/fixtures/e3/root，其余用 tests/fixtures/e2/root；四个池共用 tests/fixtures/e3a/dedupe/train_exclusions.json
（四个训练集都有键，eval_sets 同 RUN-0026）。

**样本日程**（无状态，与 E3b 同一规则：src/e3/data 的模块说明）：第 k 个样本（k = 步 × batch_size + 批内下标）的任务由 mix.weights
展开的固定循环（mix_cycle，平滑加权轮转，平局按 TASKS 的顺序）决定，任务内的序号 j 取第 j // N 个 epoch 的置换
np.random.default_rng([种子, 任务号, epoch]).permutation(N) 的第 j mod N 个。任务号 TASK_INDEX：refseg 0、depth 1（与 E3b 的
GRID_TASKS 相同）、semseg 2、normals 3。所以 mix.weights = {refseg: 1, depth: 1} 时每一步的样本 id 与 E3b（同一种子、同一批大小）
逐个相同（简报 §8.5 的同预算对照；tests 钉住）。

**一条样本**：原图（RGB uint8）+ 指令（configs 的 tasks.<任务>.instruction_template；指令在前）+ 128 × 128 的目标（src/readout/targets）。
E3a 不做数据增强，样本只由 (种子, k) 决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data import train_loaders as TL
from src.data.records import LoadResult, Sample
from src.e3 import data as E3D
from src.readout.config import E3aConfig
from src.readout.model import TASKS
from src.readout.targets import DenseTarget, depth_target, normals_target, refseg_target, semseg_target

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_REFSEG_ROOT = REPO_ROOT / "tests" / "fixtures" / "e3" / "root"
FIXTURE_DENSE_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2" / "root"
FIXTURE_EXCLUSIONS = REPO_ROOT / "tests" / "fixtures" / "e3a" / "dedupe" / "train_exclusions.json"
TASK_INDEX = {t: i for i, t in enumerate(TASKS)}
OBJECT_INFO = "objectInfo150.txt"


@dataclass(frozen=True)
class E3aPaths:
    """四个训练池的落地根（真实运行都是 DEV-0022 的同一个根，--data-root）与剔除清单（"auto" = <根>/dedupe/train_exclusions.json）。"""

    refseg_root: Path
    depth_root: Path
    semseg_root: Path
    normals_root: Path
    exclusions: str | Path | None = "auto"
    require_dedupe: bool = True
    strict: bool = True

    @classmethod
    def single_root(cls, root: str | Path) -> "E3aPaths":
        r = Path(root)
        return cls(refseg_root=r, depth_root=r, semseg_root=r, normals_root=r)

    @classmethod
    def fixtures(cls) -> "E3aPaths":
        return cls(refseg_root=FIXTURE_REFSEG_ROOT, depth_root=FIXTURE_DENSE_ROOT, semseg_root=FIXTURE_DENSE_ROOT,
                   normals_root=FIXTURE_DENSE_ROOT, exclusions=FIXTURE_EXCLUSIONS)

    def root_for(self, task: str) -> Path:
        return Path({"refseg": self.refseg_root, "depth": self.depth_root, "semseg": self.semseg_root,
                     "normals": self.normals_root}[task])

    def e3_paths(self) -> E3D.E3Paths:
        return E3D.E3Paths(refseg_root=Path(self.refseg_root), depth_root=Path(self.depth_root), exclusions=self.exclusions,
                           require_dedupe=self.require_dedupe, strict=self.strict)

    def as_dict(self) -> dict[str, Any]:
        return {"refseg_root": str(self.refseg_root), "depth_root": str(self.depth_root), "semseg_root": str(self.semseg_root),
                "normals_root": str(self.normals_root), "exclusions": str(self.exclusions),
                "require_dedupe": self.require_dedupe, "strict": self.strict}


# ---------------------------------------------------------------------------
# 训练池
# ---------------------------------------------------------------------------


def _strip(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if k != "dedupe_excluded"}


def load_pool(task: str, paths: E3aPaths, cfg: E3aConfig) -> LoadResult:
    if task in ("refseg", "depth"):
        return E3D.load_pool(task, paths.e3_paths(), cfg.task(task))
    kw = {"exclusions": paths.exclusions, "require_dedupe": paths.require_dedupe, "strict": paths.strict}
    if task == "semseg":
        res = TL.load_ade20k(paths.semseg_root, "train", **kw)
    elif task == "normals":
        res = TL.load_nyu_normals(paths.normals_root, "train", **kw)
    else:
        raise ValueError(f"未知任务 {task!r}（有：{TASKS}）")
    return LoadResult(res.records, _strip(res.stats))


def parse_object_info(text: str) -> list[str]:
    """objectInfo150.txt（制表符分隔，首行是表头 Idx Ratio Train Val Name）→ 按 Idx 排序的每类第一个名字（名字列按 ',' 或 ';' 切开
    取第一个，去首尾空白）。"""
    rows = []
    for line in text.splitlines():
        parts = [p for p in line.rstrip("\n").split("\t")]
        if len(parts) < 5 or not parts[0].strip().isdigit():
            continue
        name = parts[-1].replace(";", ",").split(",")[0].strip()
        rows.append((int(parts[0]), name))
    return [n for _, n in sorted(rows)]


def check_class_names(root: str | Path, names: Sequence[str]) -> dict[str, Any]:
    """数据根里有 ade20k/objectInfo150.txt 时，逐个核对配置的类别名就是它每类的第一个名字（不一致报错）；没有这个文件（--tiny 的
    fixture）时记 checked = False。"""
    path = Path(root) / "ade20k" / OBJECT_INFO
    if not path.is_file():
        return {"checked": False, "file": None}
    import hashlib

    raw = path.read_bytes()
    got = parse_object_info(raw.decode("utf-8", errors="replace"))
    if got != list(names):
        diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(got, names)) if a != b][:10]
        raise ValueError(f"配置的 ADE 类别名与 {path} 不一致（{len(got)} 对 {len(names)} 个；前几处 (Idx, 文件, 配置)：{diff}）")
    return {"checked": True, "file": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


# ---------------------------------------------------------------------------
# 无状态的样本日程
# ---------------------------------------------------------------------------


def mix_cycle(weights: Mapping[str, int]) -> tuple[str, ...]:
    """整数权重 → 长度 Σw 的固定循环（src/e3/data.mix_cycle 的同一规则，任务表换成 TASKS）。"""
    extra = sorted(set(weights) - set(TASKS))
    if extra:
        raise ValueError(f"mix.weights 里有未知任务 {extra}")
    items = [(t, int(weights[t])) for t in TASKS if int(weights.get(t, 0)) > 0]
    total = sum(w for _, w in items)
    if total < 1:
        raise ValueError("mix.weights 至少要有一个正权重")
    cur = {t: 0 for t, _ in items}
    out = []
    for _ in range(total):
        for t, w in items:
            cur[t] += w
        pick = max(items, key=lambda tw: (cur[tw[0]], -TASK_INDEX[tw[0]]))[0]
        cur[pick] -= total
        out.append(pick)
    return tuple(out)


def pool_position(j: int, n: int, *, seed: int, task: str) -> tuple[int, int]:
    """任务内的第 j 个样本 → (训练池下标, epoch)（src/e3/data.pool_position 的同一规则）。"""
    if n < 1:
        raise ValueError(f"{task}：训练池为空")
    ep = j // n
    perm = np.random.default_rng([int(seed), TASK_INDEX[task], ep]).permutation(n)
    return int(perm[j % n]), ep


@dataclass(frozen=True)
class SlotPlan:
    k: int
    step: int
    task: str
    j: int
    index: int
    epoch: int
    sample_id: str


class Schedule:
    def __init__(self, pools: Mapping[str, Sequence[Sample]], cfg: E3aConfig, *, seed: int, batch_size: int) -> None:
        self.pools = {t: list(v) for t, v in pools.items()}
        self.cycle = mix_cycle(cfg.mix_weights)
        missing = sorted(t for t in set(self.cycle) if t not in self.pools or not self.pools[t])
        if missing:
            raise ValueError(f"任务 {missing} 的训练池为空或没有加载")
        self.seed = int(seed)
        self.batch_size = int(batch_size)

    def plan(self, k: int) -> SlotPlan:
        task, j = E3D.slot_task(k, self.cycle)
        pool = self.pools[task]
        idx, ep = pool_position(j, len(pool), seed=self.seed, task=task)
        return SlotPlan(k=k, step=k // self.batch_size, task=task, j=j, index=idx, epoch=ep, sample_id=pool[idx].id)

    def step_plans(self, step: int) -> list[SlotPlan]:
        return [self.plan(step * self.batch_size + i) for i in range(self.batch_size)]


# ---------------------------------------------------------------------------
# 一条训练样本
# ---------------------------------------------------------------------------


@dataclass
class Example:
    plan: SlotPlan
    image: np.ndarray
    instruction: str
    order: str
    target: DenseTarget
    image_key: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.plan.task


def sample_target(task: str, sample: Sample, image_hw: tuple[int, int], cfg: E3aConfig, out: int) -> DenseTarget:
    """一条训练或评测样本的 128 × 128 目标（GT 的尺寸必须与图像相同）。"""
    tc = cfg.task(task)
    if task == "refseg":
        g = E3D.refseg_mask(sample)
        _check_hw(sample, g.shape, image_hw)
        return refseg_target(g, out)
    if task == "semseg":
        g = TL.read_ade_label(sample.gt["label"])
        _check_hw(sample, g.shape, image_hw)
        return semseg_target(g, out)
    if task == "depth":
        g = TL.read_depth_m(sample.gt["depth"])
        _check_hw(sample, g.shape, image_hw)
        return depth_target(g, out, min_valid_fraction=float(tc["min_valid_fraction"]))
    if task == "normals":
        n, valid = TL.read_normals(sample.gt["normal"])
        _check_hw(sample, valid.shape, image_hw)
        return normals_target(n, valid, out, min_valid_fraction=float(tc["min_valid_fraction"]))
    raise ValueError(f"未知任务 {task!r}")


def _check_hw(sample: Sample, gt_hw: Sequence[int], image_hw: Sequence[int]) -> None:
    if tuple(gt_hw[:2]) != tuple(image_hw[:2]):
        raise ValueError(f"{sample.id}：GT {tuple(gt_hw[:2])} 与图像 {tuple(image_hw[:2])} 尺寸不同")


class ExampleBuilder:
    """SlotPlan → Example（读图、解码 GT、降到 out × out）。没有可监督像素时抛 targets.NoSupervision（训练循环记 skipped）。"""

    def __init__(self, cfg: E3aConfig, pools: Mapping[str, Sequence[Sample]], *, out_size: int) -> None:
        self.cfg = cfg
        self.pools = {t: list(v) for t, v in pools.items()}
        self.out = int(out_size)

    def build(self, plan: SlotPlan) -> Example:
        sample = self.pools[plan.task][plan.index]
        if sample.id != plan.sample_id:
            raise RuntimeError(f"日程与训练池对不上：{plan.sample_id} ≠ {sample.id}")
        image = E3D.load_rgb(sample.images[0])
        target = sample_target(plan.task, sample, image.shape[:2], self.cfg, self.out)
        expr = sample.fields.get("expression") if plan.task == "refseg" else None
        return Example(plan, image, self.cfg.instruction(plan.task, expr), self.cfg.order(plan.task), target,
                       sample.image_keys[0] if sample.image_keys else sample.id, info=dict(target.info))


__all__ = [
    "E3aPaths",
    "Example",
    "ExampleBuilder",
    "FIXTURE_EXCLUSIONS",
    "Schedule",
    "SlotPlan",
    "TASK_INDEX",
    "check_class_names",
    "load_pool",
    "mix_cycle",
    "parse_object_info",
    "pool_position",
    "sample_target",
]
