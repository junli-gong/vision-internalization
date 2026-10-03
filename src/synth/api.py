"""数据集接口：generate(task, split, index, **cfg) -> Sample，以及可迭代的 SynthDataset。

按需即时生成，不落盘（N1）；落盘只是 export.py 的可选功能。

cfg（全部可选）
- bin：难度档序号；默认 None = index % 档数，连续序号按档轮转，天然均衡。
- canvas_hw：(H, W)，默认 CANVAS_HW = (512, 512)。
- grid_size：网格边长 N，默认 src.constants.GRID_SIZE_E3B。
- holdout_family：留出形状族（整族只进 test），默认 None；三个划分必须传同一个值。
  c、d、e 取 2D 形状族（SHAPE_FAMILIES），a、b 取 3D 实体族（SOLID_FAMILIES）。

样本 = 纯函数(任务, 划分, 序号, 该任务的生成器版本, cfg)：随机数只来自 seeding.sample_rng(任务, 划分, 序号)，
cfg 不进种子（同一序号换档位得到同一随机流下的另一个样本）。
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from src.constants import GRID_SIZE_E3B
from src.synth.constants import (
    CANVAS_HW,
    IMAGERY_BINS,
    OCCLUSION_BINS,
    PHYSICS_K_BINS,
    ROT3D_BINS,
    SOLID_FAMILIES,
    VIEW_ORBIT_BINS,
)
from src.synth.sample import GenContext, Sample
from src.synth.seeding import canonical_task, check_index, check_split, sample_rng, scene_seed, task_version
from src.synth.shapes import SHAPE_FAMILIES

_TASK_MODULES = {
    "occlusion": "src.synth.occlusion",
    "physics": "src.synth.physics",
    "imagery": "src.synth.imagery",
    "rotation3d": "src.synth.rotation3d",
    "viewpoint": "src.synth.viewpoint",
}
# b 的第 k 档 = 环绕角的第 k 档或平移量的第 k 档（VIEW_SHIFT_BINS_CM），档数相同；这里列环绕角
TASK_BINS = {
    "occlusion": OCCLUSION_BINS,
    "physics": PHYSICS_K_BINS,
    "imagery": IMAGERY_BINS,
    "rotation3d": ROT3D_BINS,
    "viewpoint": VIEW_ORBIT_BINS,
}
# 留出族的取值范围：3D 任务用实体族，其余用 2D 形状族
_HOLDOUT_FAMILIES = {"rotation3d": SOLID_FAMILIES, "viewpoint": SOLID_FAMILIES}
# "当前 mask"（D59-5：输入图里目标在变换之前的 mask；DEV-0028）在样本里已经是某个平凡基线的 mask，不另存（不改摘要）：
# a 旋转前的 mask、b 旧视角里的可见 mask、d 第 0 步的 mask 都是 "copy"；c 是目标的可见部分 "visible"。e 没有图像，没有当前 mask。
CURRENT_MASK_BASELINE = {"rotation3d": "copy", "viewpoint": "copy", "occlusion": "visible", "physics": "copy"}


def num_bins(task: str) -> int:
    return len(TASK_BINS[canonical_task(task)])


def generate(
    task: str,
    split: str,
    index: int,
    *,
    bin: int | None = None,
    canvas_hw: tuple[int, int] = CANVAS_HW,
    grid_size: int = GRID_SIZE_E3B,
    holdout_family: str | None = None,
) -> Sample:
    t = canonical_task(task)
    check_split(split)
    i = check_index(index)
    n_bins = num_bins(t)
    b = i % n_bins if bin is None else int(bin)
    if not 0 <= b < n_bins:
        raise ValueError(f"{t} 只有 {n_bins} 档，收到 bin={bin}")
    h, w = (int(v) for v in canvas_hw)
    if h < 64 or w < 64:
        raise ValueError(f"画布太小：{canvas_hw}")
    if not 1 <= int(grid_size) <= min(h, w):
        raise ValueError(f"grid_size={grid_size} 必须在 [1, min(H, W)] 内")
    if holdout_family is not None and holdout_family not in _HOLDOUT_FAMILIES.get(t, SHAPE_FAMILIES):
        raise ValueError(f"未知的留出形状族 {holdout_family!r}（任务 {t}）")
    ctx = GenContext(
        task=t,
        split=split,
        index=i,
        rng=sample_rng(t, split, i),
        bin=b,
        height=h,
        width=w,
        grid_size=int(grid_size),
        holdout_family=holdout_family,
        scene_seed=scene_seed(split, i),
        version=task_version(t),
    )
    return importlib.import_module(_TASK_MODULES[t]).generate(ctx)


class SynthDataset:
    """序号 [start, start + size) 上的按需数据集：ds[i] = generate(task, split, start + i, **cfg)。"""

    def __init__(self, task: str, split: str, size: int, *, start: int = 0, **cfg) -> None:
        self.task = canonical_task(task)
        self.split = check_split(split)
        if size < 0 or start < 0:
            raise ValueError("size 与 start 必须非负")
        check_index(start + max(size - 1, 0))
        self.size = int(size)
        self.start = int(start)
        self.cfg = dict(cfg)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Sample:
        if not -self.size <= i < self.size:
            raise IndexError(i)
        return generate(self.task, self.split, self.start + (i % self.size), **self.cfg)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.size):
            yield self[i]


def current_mask(sample: Sample):
    """样本的"当前 mask"（Prediction：全分辨率 mask 与网格；定义见 CURRENT_MASK_BASELINE）；e 没有图像，返回 None。"""
    name = CURRENT_MASK_BASELINE.get(canonical_task(sample.task))
    return None if name is None else sample.baselines[name]
