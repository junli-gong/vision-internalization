"""E2 的数据：训练池、留出的验证子集、标准化用的样本、评测集（DEV-0023）。只调用 src/data 的加载器，不改它们。

- 训练（硬约束 7：没有去重剔除清单就报错，由 src/data/train_loaders 负责，这里不绕过）：
  depth = nyu_depth_train；semseg = ade20k/train；normals = nyu_normals/train；refseg = refcoco_train 的 refcoco（unc）句子。
- 评测（不读训练剔除清单；路径可配置，见 E2Paths）：
  depth = E1 的 nyuv2 落地里 split = test 的 654 张（depth_raw，与 BTS 的 Eigen 测试 GT 同源；load_nyu_test）；
  semseg = ade20k/val；normals = nyu_normals/test；refseg = E1 的 load_refcoco("refcoco", "val")，subset 取配置的 eval_subset
  （默认 eval：去掉 E1 的开发子集；D20 的探针用图一律剔除）。
- 留出（configs/e2_probes.json 的 holdout）：按组键（depth 基础场景、semseg 图、normals 图、refseg 图）的哈希排序依次取组，
  直到条数 ≥ min_items，再在这些组里按哈希排序取 n_val 条作验证子集；这些组整组不进训练。只用来记曲线。
- 标准化样本：训练池（去掉留出组之后）按哈希排序的前 n 条。
- 训练顺序：每个 epoch 一个置换，np.random.default_rng([种子, 任务号, epoch])；第 k 条（k = 步 × 批 + 批内下标）取第 k // N 个
  epoch 置换的第 k % N 个，无状态，续跑时直接按步号重算。
哈希排序一律用 src/data/subsets.hash_rank，盐是 E2_SALT（与 E1 的抽样互相独立）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data import loaders as L
from src.data import train_loaders as TL
from src.data.land.manifest import read_manifest
from src.data.records import LoadResult, Sample
from src.data.subsets import DEV_SUBSETS_PATH, hash_rank, sort_ids
from src.e2.config import TASKS

E2_SALT = "visint-e2-probes-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_TRAIN_ROOT = REPO_ROOT / "tests" / "fixtures" / "e2" / "root"
FIXTURE_E1_DIR = REPO_ROOT / "tests" / "fixtures" / "e1"
NYU_TEST_SOURCE = "nyuv2"  # E1 的落地源（src/data/land/nyuv2.py）
REFSEG_DATASET = "refcoco"  # RefCOCO（unc）


@dataclass(frozen=True)
class E2Paths:
    """train_root：DEV-0022 的落地根（含 dedupe/train_exclusions.json；ade20k/val 与 nyu_normals/test 与训练划分一起落在这里，
    评测也从这里读）；eval_root：E1 的落地根（nyuv2 的官方 654、refcoco 的 val；可与 train_root 相同）；
    dev_subsets：E1 的开发子集清单（RefCOCO val 去掉开发子集用）。"""

    train_root: Path
    eval_root: Path
    dev_subsets: Path = DEV_SUBSETS_PATH
    exclusions: str | Path | None = "auto"
    require_dedupe: bool = True
    strict: bool = True

    @classmethod
    def fixtures(cls) -> "E2Paths":
        """--tiny：tests/fixtures/e2/root 是训练根（ade20k、nyu_normals 的评测划分也在这里），tests/fixtures/e1/root 是 E1 的评测根。"""
        return cls(train_root=FIXTURE_TRAIN_ROOT, eval_root=FIXTURE_E1_DIR / "root",
                   dev_subsets=FIXTURE_E1_DIR / "dev_subsets.json")

    def as_dict(self) -> dict[str, Any]:
        return {"train_root": str(self.train_root), "eval_root": str(self.eval_root), "dev_subsets": str(self.dev_subsets),
                "exclusions": str(self.exclusions), "require_dedupe": self.require_dedupe, "strict": self.strict}


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def load_train_pool(task: str, paths: E2Paths) -> LoadResult:
    kw = {"exclusions": paths.exclusions, "require_dedupe": paths.require_dedupe, "strict": paths.strict}
    if task == "depth":
        return TL.load_nyu_depth_train(paths.train_root, **kw)
    if task == "semseg":
        return TL.load_ade20k(paths.train_root, "train", **kw)
    if task == "normals":
        return TL.load_nyu_normals(paths.train_root, "train", **kw)
    if task == "refseg":
        return TL.load_refcoco_train(paths.train_root, REFSEG_DATASET, **kw)
    raise ValueError(f"未知任务 {task!r}，可用：{TASKS}")


def load_nyu_test(eval_root: str | Path, *, strict: bool = True) -> LoadResult:
    """E1 的 nyuv2 落地里 split = test 的条目（官方 654）：RGB 与 depth/<idx>.npz 的 depth_raw（米，0 = 无效）。"""
    src = Path(eval_root) / NYU_TEST_SOURCE
    files = read_manifest(src)["files"]
    out, n_missing = [], 0
    for r in json.loads((src / "index.json").read_text())["images"]:
        if r["split"] != "test":
            continue
        if not all(rel in files and (src / rel).exists() for rel in (r["rgb"], r["depth"])):
            if strict:
                raise FileNotFoundError(f"nyuv2 test {r['idx']}：{r['rgb']} 或 {r['depth']} 不存在；落地不全")
            n_missing += 1
            continue
        out.append(Sample(
            id=f"{NYU_TEST_SOURCE}/test/{r['idx']}", task="depth", images=(str(src / r["rgb"]),),
            image_keys=(f"nyuv2:{r['idx']}",), phash=((files[r["rgb"]] or {}).get("phash"),),
            fields={"idx": r["idx"], "scene": r.get("scene")},
            gt={"depth_npz": str(src / r["depth"]), "key": "depth_raw", "units": "meters"},
            source=NYU_TEST_SOURCE, split="test"))
    order = {i: k for k, i in enumerate(sort_ids(str(s.fields["idx"]) for s in out))}
    out.sort(key=lambda s: order[str(s.fields["idx"])])
    return LoadResult(out, {"set": f"{NYU_TEST_SOURCE}/test", "n_kept": len(out), "n_missing": n_missing})


def load_eval_set(task: str, paths: E2Paths, *, eval_subset: str | None = None) -> LoadResult:
    if task == "depth":
        return load_nyu_test(paths.eval_root, strict=paths.strict)
    if task == "semseg":
        return TL.load_ade20k(paths.train_root, "val", strict=paths.strict)
    if task == "normals":
        return TL.load_nyu_normals(paths.train_root, "test", strict=paths.strict)
    if task == "refseg":
        return L.load_refcoco(paths.eval_root, REFSEG_DATASET, "val", subset=eval_subset or "eval",
                              dev_config=paths.dev_subsets, strict=paths.strict)
    raise ValueError(f"未知任务 {task!r}，可用：{TASKS}")


# ---------------------------------------------------------------------------
# 留出、标准化样本、训练顺序
# ---------------------------------------------------------------------------


def group_key(task: str, s: Sample) -> str:
    """留出的组：同一组的样本整组进训练或整组进验证。"""
    if task == "depth":
        return str(s.fields["base_scene"])
    if task == "semseg":
        return str(s.fields["stem"])
    if task == "normals":
        return str(s.fields["idx"])
    if task == "refseg":
        return str(s.fields["image_id"])
    raise ValueError(task)


@dataclass
class Split:
    train: list[Sample]
    val: list[Sample]
    stats: list[Sample]
    info: dict[str, Any]


def split_pool(task: str, pool: Sequence[Sample], holdout: Mapping[str, Any], *, n_stats: int) -> Split:
    """训练池 → (训练, 验证子集, 标准化样本)。规则见模块说明。"""
    if not pool:
        raise ValueError(f"{task}：训练池是空的")
    groups: dict[str, list[Sample]] = {}
    for s in pool:
        groups.setdefault(group_key(task, s), []).append(s)
    held: list[str] = []
    n_held = 0
    for g in hash_rank(groups, f"e2:{task}:holdout", E2_SALT):
        if n_held >= int(holdout["min_items"]):
            break
        held.append(g)
        n_held += len(groups[g])
    if len(held) == len(groups):
        raise ValueError(f"{task}：留出用掉了全部 {len(groups)} 个组，训练集为空（min_items={holdout['min_items']}）")
    held_set = set(held)
    cand = [s for g in held for s in groups[g]]
    by_id = {s.id: s for s in cand}
    val_ids = set(hash_rank(by_id, f"e2:{task}:val", E2_SALT)[: int(holdout["n_val"])])
    val = [s for s in pool if s.id in val_ids]
    train = [s for s in pool if group_key(task, s) not in held_set]
    stats_ids = set(hash_rank((s.id for s in train), f"e2:{task}:stats", E2_SALT)[: int(n_stats)])
    stats = [s for s in train if s.id in stats_ids]
    info = {"group": holdout["group"], "n_groups": len(groups), "n_held_groups": len(held), "held_groups": sort_ids(held),
            "n_held_items": n_held, "n_val": len(val), "n_train": len(train), "n_stats": len(stats),
            "val_ids": [s.id for s in val], "stats_ids": [s.id for s in stats]}
    return Split(train, val, stats, info)


TASK_INDEX = {t: i for i, t in enumerate(TASKS)}


def batch_indices(n: int, step: int, batch_size: int, *, seed: int, task: str) -> list[int]:
    """第 step 步（0 起）的批：训练集下标（规则见模块说明）。"""
    if n < 1:
        raise ValueError("训练集为空")
    out = []
    cache: dict[int, np.ndarray] = {}
    for k in range(step * batch_size, (step + 1) * batch_size):
        ep = k // n
        if ep not in cache:
            cache[ep] = np.random.default_rng([int(seed), TASK_INDEX[task], ep]).permutation(n)
        out.append(int(cache[ep][k % n]))
    return out


def ids_digest(ids: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


__all__ = [
    "E2Paths",
    "E2_SALT",
    "Split",
    "batch_indices",
    "group_key",
    "ids_digest",
    "load_eval_set",
    "load_nyu_test",
    "load_train_pool",
    "split_pool",
]
