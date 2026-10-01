"""E1 的评测子集清单（configs/e1_eval_subsets.json；DEV-0012 审查第 1 轮，D28-3、D28-10）。

总体：各加载器 subset="eval" 的结果，即已剔除开发子集与 D20 探针图（剔除统计一并写进配置）；NYU 点对取
configs/e1_nyu_pairs.json 的 test 划分。抽样：src/data/subsets.seeded_sample，每类子集一个独立的盐
（src/data/constants.EVAL_SUBSET_SALTS），用途串就是子集名。不足目标数时全取，并标 all_taken。

子集（名字 → 总体）：
- poly:<数据集_划分>：RefCOCO/+ 的 unc val/testA/testB、RefCOCOg 的 umd val/test，每个 1,000 句。
- grid16:refcoco_unc_val：RefCOCO unc val 500 句，与 poly 独立抽，允许重叠。
- think:<任务>：每个任务 200 条。refexp_box ← RefCOCO unc val；refexp_poly ← poly:refcoco_unc_val；count ← CountBench；
  ovd ← COCO 500；rel_depth_nyu ← NYU 点对（test）；rel_depth_blink ← BLINK（112 条全取）；spatial_cvbench；
  spatial_embspatial；grid16 ← grid16:refcoco_unc_val。
- think8192:<任务>：从 think:<任务> 再抽 50 条。
- extra:refcoco_unc_val（500 句）、extra:coco_val2017（100 张）：附加对照（Qwen 约 1K token；InternVL 原生动态 tile）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import loaders as L
from src.data.constants import (
    EVAL_EXTRA_COCO_N,
    EVAL_EXTRA_REFCOCO_N,
    EVAL_GRID16_N,
    EVAL_POLY_PER_SPLIT,
    EVAL_SUBSET_SALTS,
    EVAL_THINK8192_N,
    EVAL_THINK_N,
)
from src.data.subsets import population_digest, seeded_sample, sort_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SUBSETS_PATH = REPO_ROOT / "configs" / "e1_eval_subsets.json"
EVAL_SUBSETS_VERSION = 1

POLY_SPLITS = (
    ("refcoco", "unc", "val"),
    ("refcoco", "unc", "testA"),
    ("refcoco", "unc", "testB"),
    ("refcoco+", "unc", "val"),
    ("refcoco+", "unc", "testA"),
    ("refcoco+", "unc", "testB"),
    ("refcocog", "umd", "val"),
    ("refcocog", "umd", "test"),
)

# 思考子集：(任务, 总体名或父子集名)
THINK_TASKS = (
    ("refexp_box", "refcoco_unc_val"),
    ("refexp_poly", "poly:refcoco_unc_val"),
    ("count", "countbench"),
    ("ovd", "coco_val2017"),
    ("rel_depth_nyu", "nyu_pairs"),
    ("rel_depth_blink", "blink_reldepth"),
    ("spatial_cvbench", "cvbench"),
    ("spatial_embspatial", "embspatial"),
    ("grid16", "grid16:refcoco_unc_val"),
)

UNITS = {
    "refcoco": "sent_id",
    "countbench": "row（官方 CountBench.json 的 0 起行号）",
    "coco_val2017": "image_id",
    "nyu_pairs": "点对 id",
    "blink_reldepth": "idx",
    "cvbench": "idx",
    "embspatial": "question_id",
}


def _stats(res) -> dict[str, Any]:
    st = res.stats
    return {
        "n_total": st["n_total"],
        "n_missing_image": st["n_missing_image"],
        "excluded_by_reason": st["exclusion"]["excluded_by_reason"],
        "n_eval": st["n_kept"],
    }


def eval_populations(root: str | Path, *, dev_config: str | Path | None = None, pairs_config: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """各评测总体：{名字: {ids, unit, provenance, stats}}。图像必须落全（strict 加载）。"""
    from src.data.nyu_pairs import PAIRS_PATH, load_pairs

    pops: dict[str, dict[str, Any]] = {}
    for ds, split_by, split in POLY_SPLITS:
        res = L.load_refcoco(root, ds, split, subset="eval", dev_config=dev_config)
        pops[f"{ds}_{split_by}_{split}"] = {"ids": [str(s.fields["sent_id"]) for s in res.records], "unit": UNITS["refcoco"], "provenance": f"load_refcoco({ds!r}, {split!r}, subset='eval')", "stats": _stats(res)}
    for name, fn, attr in (
        ("countbench", L.load_countbench, "row"),
        ("blink_reldepth", L.load_blink_reldepth, "idx"),
        ("cvbench", L.load_cvbench, "idx"),
        ("embspatial", L.load_embspatial, "question_id"),
    ):
        res = fn(root, subset="eval", dev_config=dev_config)
        pops[name] = {"ids": [str(s.fields[attr]) for s in res.records], "unit": UNITS[name], "provenance": f"{fn.__name__}(subset='eval')", "stats": _stats(res)}
    res = L.load_coco_ovd(root)
    pops["coco_val2017"] = {"ids": [str(s.fields["image_id"]) for s in res.records], "unit": UNITS["coco_val2017"], "provenance": "load_coco_ovd()", "stats": _stats(res)}
    cfg = load_pairs(pairs_config or PAIRS_PATH)
    ids = [p["id"] for p in cfg["pairs"] if p["split"] == "test"]
    pops["nyu_pairs"] = {"ids": ids, "unit": UNITS["nyu_pairs"], "provenance": "e1_nyu_pairs.json 的 test 划分（Eigen 654 上的点对）", "stats": {"n_eval": len(ids)}}
    return pops


def build_eval_subsets(pops: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    subsets: dict[str, dict[str, Any]] = {}

    def add(name: str, kind: str, task: str, parent: str, parent_ids: Sequence[str], k: int, note: str = "") -> list[str]:
        ids = seeded_sample(parent_ids, k, purpose=name, salt=EVAL_SUBSET_SALTS[kind])
        subsets[name] = {
            "kind": kind,
            "task": task,
            "population": parent,
            "n_population": len(parent_ids),
            "population_sha256": population_digest(parent_ids),
            "k_target": k,
            "k": len(ids),
            "all_taken": len(ids) == len(parent_ids),
            "note": note,
            "ids": ids,
        }
        return ids

    for ds, split_by, split in POLY_SPLITS:
        key = f"{ds}_{split_by}_{split}"
        add(f"poly:{key}", "poly", "refexp_poly", key, pops[key]["ids"], EVAL_POLY_PER_SPLIT)
    add("grid16:refcoco_unc_val", "grid16", "grid16", "refcoco_unc_val", pops["refcoco_unc_val"]["ids"], EVAL_GRID16_N, "与 poly 独立抽，允许重叠")
    for task, parent in THINK_TASKS:
        parent_ids = subsets[parent]["ids"] if parent in subsets else pops[parent]["ids"]
        t = add(f"think:{task}", "think", task, parent, parent_ids, EVAL_THINK_N)
        if len(t) < EVAL_THINK_N:
            subsets[f"think:{task}"]["note"] = f"总体只有 {len(t)} 条，全取（目标 {EVAL_THINK_N}）"
        add(f"think8192:{task}", "think8192", task, f"think:{task}", t, EVAL_THINK8192_N)
    add("extra:refcoco_unc_val", "extra", "refexp_box", "refcoco_unc_val", pops["refcoco_unc_val"]["ids"], EVAL_EXTRA_REFCOCO_N, "附加对照：Qwen 约 1K 视觉 token；InternVL 原生动态 tile")
    add("extra:coco_val2017", "extra", "ovd", "coco_val2017", pops["coco_val2017"]["ids"], EVAL_EXTRA_COCO_N, "附加对照：Qwen 约 1K 视觉 token；InternVL 原生动态 tile")
    return {
        "version": EVAL_SUBSETS_VERSION,
        "salts": dict(EVAL_SUBSET_SALTS),
        "method": "按 sha256(该类的盐|子集名|id) 升序取前 k 个，再按 id 自然顺序列出（src/data/subsets.py、src/data/eval_subsets.py）；不足 k 全取",
        "populations": {
            name: {"unit": p["unit"], "n": len(p["ids"]), "sha256": population_digest(p["ids"]), "provenance": p["provenance"], "stats": p["stats"]}
            for name, p in sorted(pops.items())
        },
        "subsets": subsets,
    }


def compare(new: Mapping[str, Any], old: Mapping[str, Any]) -> list[str]:
    """--check 用：逐个子集比 ids、k、n_population、population_sha256，逐个总体比 n、sha256。返回差异描述。"""
    bad = []
    for name, p in new["populations"].items():
        o = old["populations"].get(name)
        if o is None or (o["n"], o["sha256"]) != (p["n"], p["sha256"]):
            bad.append(f"总体 {name}：入库 {o and (o['n'], o['sha256'][:12])} ≠ 重算 {(p['n'], p['sha256'][:12])}")
    for name, sub in new["subsets"].items():
        o = old["subsets"].get(name)
        if o is None:
            bad.append(f"子集 {name}：入库文件里没有")
            continue
        for k in ("ids", "k", "n_population", "population_sha256"):
            if o[k] != sub[k]:
                bad.append(f"子集 {name}.{k}：入库 {str(o[k])[:60]} ≠ 重算 {str(sub[k])[:60]}")
    for name in set(old["subsets"]) - set(new["subsets"]):
        bad.append(f"子集 {name}：入库文件里多出来")
    return bad


def load_eval_subsets(path: str | Path = EVAL_SUBSETS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def subset_ids(name: str, path: str | Path = EVAL_SUBSETS_PATH) -> list[str]:
    """某个评测子集的 id（按自然顺序）；配置里没有就报错。"""
    cfg = load_eval_subsets(path)
    if name not in cfg["subsets"]:
        raise KeyError(f"{path} 里没有评测子集 {name}")
    return sort_ids(cfg["subsets"][name]["ids"])
