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

**E3a+C3 的两条流**（MixedSchedule、MixedBuilder；DEV-0044，配置的 mixed 节）：第 t 步 = C3 种子 s 的样本流
（src/e3/data.Schedule(text_cfg, supervision = text, batch_size = per_step.text)）的 step_plans(t)，接着 E3a full 种子 s 的样本流
（本模块的 Schedule(batch_size = per_step.dense)）的 step_plans(t)。两者的 plan(k) 都只由 (种子, k) 决定、与 batch_size 无关，所以文本流
第 t 步就是 C3 运行的第 8t … 8t + 7 个槽位，稠密流就是 E3a full 运行的第 8t … 8t + 7 个槽位。文本样本由 src/e3/data.ExampleBuilder
（supervision = text，C3 的构造，原样调用）构造，训练池是本模块加载的 refseg、depth 两个池（与 C3 同一个加载函数）；稠密样本由本模块的
ExampleBuilder 构造。样本属于哪条流由 stream_of(plan) 判定：src/e3/data.SlotPlan（kind = text）是文本流，本模块的 SlotPlan 是稠密流。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data import train_loaders as TL
from src.data.records import LoadResult, Sample
from src.e3 import data as E3D
from src.e3.config import E3Config
from src.readout.config import MIXED_STREAMS, TEXT_SUPERVISION, E3aConfig
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


# ---------------------------------------------------------------------------
# E3a+C3 的两条流（模块说明）
# ---------------------------------------------------------------------------


def stream_of(plan: Any) -> str:
    """E3a+C3 里样本属于哪条流：src/e3/data.SlotPlan（C3，kind = text）是 text，本模块的 SlotPlan 是 dense。"""
    if isinstance(plan, E3D.SlotPlan):
        if plan.kind != TEXT_SUPERVISION:
            raise ValueError(f"文本流的样本 kind 应为 {TEXT_SUPERVISION!r}，收到 {plan.kind!r}")
        return "text"
    if isinstance(plan, SlotPlan):
        return "dense"
    raise TypeError(f"不认识的样本计划 {type(plan).__name__}")


def text_pools(pools: Mapping[str, Sequence[Sample]], text_cfg: E3Config) -> dict[str, list[Sample]]:
    """文本流（C3）用的训练池：E3a 已加载的、C3 混合里的那几个任务（refseg、depth；与 C3 同一个加载函数，配置核对见
    src/readout/config._check_mixed_compat）。"""
    missing = [t for t in text_cfg.tasks if t not in pools]
    if missing:
        raise ValueError(f"文本流要的训练池 {missing} 没有加载")
    return {t: list(pools[t]) for t in text_cfg.tasks}


class MixedSchedule:
    """E3a+C3 的样本日程（模块说明）：第 t 步 = 各条流（按 order）的 step_plans(t) 依次拼接。"""

    def __init__(self, pools: Mapping[str, Sequence[Sample]], cfg: E3aConfig, text_cfg: E3Config, *, seed: int,
                 per_step: Mapping[str, int], order: Sequence[str] = MIXED_STREAMS) -> None:
        if sorted(order) != sorted(MIXED_STREAMS) or set(per_step) != set(MIXED_STREAMS):
            raise ValueError(f"E3a+C3 的样本流必须恰好是 {MIXED_STREAMS}，收到 order {list(order)}、per_step {dict(per_step)}")
        self.order = tuple(order)
        self.per_step = {s: int(per_step[s]) for s in MIXED_STREAMS}
        self.streams: dict[str, Any] = {
            "text": E3D.Schedule(text_pools(pools, text_cfg), text_cfg, seed=seed, supervision=TEXT_SUPERVISION,
                                 batch_size=self.per_step["text"]),
            "dense": Schedule(pools, cfg, seed=seed, batch_size=self.per_step["dense"]),
        }
        self.batch_size = sum(self.per_step.values())

    def step_plans(self, step: int) -> list[Any]:
        return [p for s in self.order for p in self.streams[s].step_plans(step)]


def c3_text_builder(text_cfg: E3Config, pools: Mapping[str, Sequence[Sample]], *, seed: int, adapter: Any,
                    family: str) -> E3D.ExampleBuilder:
    """文本流的样本构造：src/e3/data.ExampleBuilder(supervision = text)，参数与 C3 的训练（src/e3/train.run_training）逐项相同：
    encode = 适配器 tokenizer 的无特殊 token 编码，停止 token = 本家族的回合结束 token，框用适配器的 format_boxes。"""
    from src.e3.tokens import family_stop_id

    return E3D.ExampleBuilder(text_cfg, text_pools(pools, text_cfg), seed=seed, supervision=TEXT_SUPERVISION,
                              stop_id=family_stop_id(adapter),
                              encode=lambda s: list(adapter.tokenizer(s, add_special_tokens=False)["input_ids"]),
                              family=family, format_boxes=adapter.format_boxes)


class MixedBuilder:
    """plan → 样本：文本流交给 C3 的构造（E3D.Example），稠密流交给 E3a 的构造（本模块的 Example）。"""

    def __init__(self, dense: ExampleBuilder, text: E3D.ExampleBuilder) -> None:
        self.dense = dense
        self.text = text

    def build(self, plan: Any) -> Any:
        return self.text.build(plan) if stream_of(plan) == "text" else self.dense.build(plan)


__all__ = [
    "E3aPaths",
    "Example",
    "ExampleBuilder",
    "FIXTURE_EXCLUSIONS",
    "MixedBuilder",
    "MixedSchedule",
    "Schedule",
    "SlotPlan",
    "TASK_INDEX",
    "c3_text_builder",
    "check_class_names",
    "load_pool",
    "mix_cycle",
    "parse_object_info",
    "pool_position",
    "sample_target",
    "stream_of",
    "text_pools",
]
