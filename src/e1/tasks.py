"""E1 的 9 个任务：每个任务在各子集上选哪些样本、按什么分档报告（DEV-0014 任务书 §2）。

数据一律经 src/data/loaders（已剔除开发子集与 D20 探针图，剔除逐条记录在加载统计里），子集 id 取自入库的清单：
- eval：主口径的评测总体。定位跑全划分；多边形、网格取 configs/e1_eval_subsets.json 的 poly:*、grid16:*；检测是 COCO 500 + ODinW13。
- dev：开发子集（configs/e1_dev_subsets.json；D11-6），只核对格式、截断率、解析失败，以及批处理验收。
  检测没有开发子集（COCO 500 与 ODinW 都只有评测图）：这里用 RefCOCO unc val 开发子集里的图（COCO train2014）配 COCO 的 80 类
  做格式核对，没有 GT，不算指标（偏离，见完成总结）。
- think / think8192：think:<任务> / think8192:<任务>（每任务 200 / 50 条），开思考。
- extra：附加对照的子集（extra:refcoco_unc_val → 定位，extra:coco_val2017 → 检测），配 --policy qwen:1024 或 internvl:native。
样本顺序固定：多划分的任务按 POLY_SPLITS 的划分顺序，划分内按加载器的 id 自然顺序；断点续传与逐字节复现都依赖它。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.data import loaders as L
from src.data.eval_subsets import EVAL_SUBSETS_PATH, POLY_SPLITS, subset_ids
from src.data.nyu_pairs import PAIRS_PATH
from src.data.records import Sample
from src.data.subsets import DEV_SUBSETS_PATH
from src.e1.prompts import SUBSET_KINDS, TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "e1"

# 每个任务支持哪些子集（extra 只给附加对照的两个任务）
TASK_SUBSETS: dict[str, tuple[str, ...]] = {
    t: ("eval", "dev", "think", "think8192") + (("extra",) if t in ("refexp_box", "ovd") else ()) for t in TASKS
}
EXTRA_SUBSET_OF_TASK = {"refexp_box": "extra:refcoco_unc_val", "ovd": "extra:coco_val2017"}
REFCOCO_DATASETS = ("refcoco", "refcoco+", "refcocog")
OVD_DEV_SOURCE = "ovd_dev/refcoco_unc_val_images"


@dataclass(frozen=True)
class DataPaths:
    """数据与清单的位置。root 是落地目录（服务器上 $PROJ/data）；其余默认取入库的配置。"""

    root: Path
    dev_subsets: Path = DEV_SUBSETS_PATH
    eval_subsets: Path = EVAL_SUBSETS_PATH
    nyu_pairs: Path = PAIRS_PATH
    nyu_cache: Path | None = None
    strict: bool = True

    @classmethod
    def fixtures(cls, nyu_cache: Path | None = None) -> "DataPaths":
        """开发机彩排（--tiny）与测试：tests/fixtures/e1/ 的小数据。"""
        return cls(root=FIXTURE_DIR / "root", dev_subsets=FIXTURE_DIR / "dev_subsets.json",
                   eval_subsets=FIXTURE_DIR / "eval_subsets.json", nyu_pairs=FIXTURE_DIR / "nyu_pairs.json",
                   nyu_cache=nyu_cache)

    def config_files(self) -> list[Path]:
        """本次运行用到的清单文件（sha256 记进 meta.json）；不存在的跳过。"""
        return [p for p in (self.dev_subsets, self.eval_subsets, self.nyu_pairs) if Path(p).exists()]


@dataclass
class Selection:
    task: str
    subset: str
    samples: list[Sample]
    has_gt: bool = True
    stats: dict[str, Any] = field(default_factory=dict)


def _pick(samples: list[Sample], ids: list[str], unit: Callable[[Sample], str], what: str) -> list[Sample]:
    """按子集清单的 id 取样本（保持 samples 的顺序）；清单里有、总体里找不到的 id 直接报错（不静默丢）。"""
    want = set(ids)
    out = [s for s in samples if unit(s) in want]
    found = {unit(s) for s in out}
    missing = sorted(want - found)
    if missing:
        raise KeyError(f"{what}：清单里的 {len(missing)} 个 id 在总体里找不到，例如 {missing[:5]}")
    return out


def _sent_id(s: Sample) -> str:
    return str(s.fields["sent_id"])


def _stats(res: L.LoadResult) -> dict[str, Any]:
    st = res.stats
    return {"n_total": st.get("n_total"), "n_kept": st.get("n_kept"),
            "excluded_by_reason": st.get("exclusion", {}).get("excluded_by_reason", {})}


def _refcoco(paths: DataPaths, ds: str, split: str, subset: str) -> L.LoadResult:
    return L.load_refcoco(paths.root, ds, split, subset=subset, dev_config=paths.dev_subsets, strict=paths.strict)


def _refcoco_dev(paths: DataPaths, datasets=REFCOCO_DATASETS) -> tuple[list[Sample], dict[str, Any]]:
    out, st = [], {}
    for ds in datasets:
        res = _refcoco(paths, ds, "val", "dev")
        out += res.records
        st[f"{ds}/val/dev"] = _stats(res)
    return out, st


def _refcoco_unc_val(paths: DataPaths) -> L.LoadResult:
    return _refcoco(paths, "refcoco", "val", "eval")


def _select_refexp(task: str, subset: str, paths: DataPaths) -> Selection:
    st: dict[str, Any] = {}
    if subset == "dev":
        samples, st = _refcoco_dev(paths, ("refcoco",) if task == "grid16" else REFCOCO_DATASETS)
        return Selection(task, subset, samples, True, st)
    if subset == "eval" and task in ("refexp_box", "refexp_poly"):
        samples = []
        for ds, split_by, split in POLY_SPLITS:
            res = _refcoco(paths, ds, split, "eval")
            key = f"{ds}_{split_by}_{split}"
            st[key] = _stats(res)
            recs = res.records
            if task == "refexp_poly":
                recs = _pick(recs, subset_ids(f"poly:{key}", paths.eval_subsets), _sent_id, f"poly:{key}")
                st[key]["n_selected"] = len(recs)
            samples += recs
        return Selection(task, subset, samples, True, st)
    res = _refcoco_unc_val(paths)
    st["refcoco_unc_val"] = _stats(res)
    if subset == "eval":  # grid16
        name = "grid16:refcoco_unc_val"
    elif subset == "extra":
        name = EXTRA_SUBSET_OF_TASK[task]
    else:
        name = f"{subset}:{task}"
    samples = _pick(res.records, subset_ids(name, paths.eval_subsets), _sent_id, name)
    st["subset_name"] = name
    return Selection(task, subset, samples, True, st)


def _coco_categories(paths: DataPaths) -> tuple[list[str], list[int]]:
    from src.data.land.coco import SUBSET_FILE

    coco = json.loads((paths.root / "coco_val2017" / SUBSET_FILE).read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    return [c["name"] for c in cats], [c["id"] for c in cats]


def odinw_configs(root: Path) -> list[str]:
    """落地目录里的 ODinW 数据集（MANIFEST 里有 <config>/annotations_e1sub.json 的），按名字排序。"""
    from src.data.land.manifest import read_manifest

    files = read_manifest(root / "odinw13")["files"]
    return sorted({p.rsplit("/", 1)[0] for p in files if p.endswith("/annotations_e1sub.json")})


def _select_ovd(subset: str, paths: DataPaths) -> Selection:
    st: dict[str, Any] = {}
    if subset == "dev":
        names, ids = _coco_categories(paths)
        recs, st = _refcoco_dev(paths, ("refcoco",))
        seen: dict[int, Sample] = {}
        for s in recs:
            iid = int(s.fields["image_id"])
            if iid not in seen:
                seen[iid] = Sample(
                    id=f"{OVD_DEV_SOURCE}/{iid}", task="ovd", images=s.images, image_keys=s.image_keys,
                    fields={"categories": names, "category_ids": ids, "image_id": iid,
                            "image_size": list(s.fields["image_size"])},
                    gt={"boxes": None}, source=OVD_DEV_SOURCE, split="dev", phash=s.phash)
        st["note"] = "检测没有开发子集：用 RefCOCO unc val 开发子集的图（去重）配 COCO 80 类，只核对格式，没有 GT"
        return Selection("ovd", subset, [seen[k] for k in sorted(seen)], False, st)
    coco = L.load_coco_ovd(paths.root, strict=paths.strict)
    st["coco_val2017"] = _stats(coco)
    if subset == "eval":
        samples = list(coco.records)
        for cfg in odinw_configs(paths.root):
            res = L.load_odinw(paths.root, cfg, strict=paths.strict)
            st[f"odinw13/{cfg}"] = _stats(res)
            samples += res.records
        return Selection("ovd", subset, samples, True, st)
    name = EXTRA_SUBSET_OF_TASK["ovd"] if subset == "extra" else f"{subset}:ovd"
    st["subset_name"] = name
    samples = _pick(coco.records, subset_ids(name, paths.eval_subsets), lambda s: str(s.fields["image_id"]), name)
    return Selection("ovd", subset, samples, True, st)


def _select_simple(task: str, subset: str, paths: DataPaths) -> Selection:
    """计数、BLINK、CV-Bench、EmbSpatial、NYU 点对：eval / dev 直接取加载器的子集，think 按清单筛 eval。"""
    if task == "rel_depth_nyu":
        load = lambda sub: L.load_nyu_pairs(paths.root, subset=sub, pairs_config=paths.nyu_pairs,  # noqa: E731
                                            cache_dir=paths.nyu_cache, strict=paths.strict)
        unit = lambda s: s.id.rsplit("/", 1)[1]  # noqa: E731
    else:
        fn, unit = {
            "count": (L.load_countbench, lambda s: str(s.fields["row"])),
            "rel_depth_blink": (L.load_blink_reldepth, lambda s: str(s.fields["idx"])),
            "spatial_cvbench": (L.load_cvbench, lambda s: str(s.fields["idx"])),
            "spatial_embspatial": (L.load_embspatial, lambda s: str(s.fields["question_id"])),
        }[task]
        load = lambda sub: fn(paths.root, subset=sub, dev_config=paths.dev_subsets, strict=paths.strict)  # noqa: E731
    res = load("dev" if subset == "dev" else "eval")
    st = {"population": _stats(res)}
    samples = res.records
    if subset in ("think", "think8192"):
        name = f"{subset}:{task}"
        st["subset_name"] = name
        samples = _pick(samples, subset_ids(name, paths.eval_subsets), unit, name)
    return Selection(task, subset, samples, True, st)


def select(task: str, subset: str, paths: DataPaths) -> Selection:
    """任务 × 子集 → 样本（顺序固定）。不支持的组合报错。"""
    if task not in TASKS:
        raise ValueError(f"未知任务 {task!r}；可用：{TASKS}")
    if subset not in SUBSET_KINDS or subset not in TASK_SUBSETS[task]:
        raise ValueError(f"{task} 不支持子集 {subset!r}；可用：{TASK_SUBSETS[task]}")
    if task in ("refexp_box", "refexp_poly", "grid16"):
        sel = _select_refexp(task, subset, paths)
    elif task == "ovd":
        sel = _select_ovd(subset, paths)
    else:
        sel = _select_simple(task, subset, paths)
    ids = [s.id for s in sel.samples]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{task}/{subset}：样本 id 有重复")
    sel.stats["n"] = len(sel.samples)
    return sel


def default_nyu_cache() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp")) / "visint_nyu_pairs_rendered"


# ---------------------------------------------------------------------------
# 分档（metrics.json 的 breakdown）
# ---------------------------------------------------------------------------


def _cvbench_source(source_dataset: str) -> str:
    """CV-Bench 的 source_dataset → 官方合成准确率用的三个来源（README 的 calculate_accuracy(df, 'ADE20K'|'COCO'|'Omni3D')）。"""
    for key in ("ADE20K", "COCO", "Omni3D"):
        if source_dataset.startswith(key):
            return key
    raise ValueError(f"认不出 CV-Bench 的来源 {source_dataset!r}")


def groups_of(task: str, s: Sample) -> dict[str, str]:
    """一条样本在各分档维度上的取值。"""
    if task in ("refexp_box", "refexp_poly", "grid16"):
        return {"split": s.id.rsplit("/", 1)[0]}
    if task == "count":
        return {"count": str(s.gt["count"])}
    if task == "ovd":
        return {"dataset": s.source}
    if task == "spatial_cvbench":
        return {"type": s.fields["type"], "task": f"{s.fields['type']}/{s.fields['task']}",
                "source": _cvbench_source(s.fields["source_dataset"])}
    if task == "spatial_embspatial":
        return {"relation": s.fields["relation"], "data_source": s.fields["data_source"]}
    return {}


__all__ = [
    "DataPaths",
    "EXTRA_SUBSET_OF_TASK",
    "OVD_DEV_SOURCE",
    "Selection",
    "TASK_SUBSETS",
    "default_nyu_cache",
    "groups_of",
    "odinw_configs",
    "select",
]
