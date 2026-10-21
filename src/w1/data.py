"""W1 的数据划分（DEV-0051；D103 §二.5；规则在 configs/w1.json 的 splits，id 清单在 configs/w1_splits.json）。

划分（名字 → 样本）：
- devtune（开发-调参，300 句）与 devcheck（开发-检查，500 句）：E1 的 poly:refcoco_unc_val（configs/e1_eval_subsets.json，1,000 句）
  按 sha256(salt|purpose|sent_id) 升序排成一列（src/data/subsets.hash_rank，salt = w1_dev_v1），前 300 句是 devtune，第 301–800 句是
  devcheck，不重叠；各自按 id 自然顺序列出。用于 refseg 与 refexp_box（同一批句子）。
- test_<数据集_划分方式_划分>：E1 的 poly:<同名> 子集原样（每个 1,000 句）：refcoco unc testA/testB、refcoco+ unc testA/testB、
  refcocog umd test。
- test_countbench：E1 的 CountBench 评测条目（load_countbench(subset='eval') 全部，442 条）。开发机没有数据，id 清单不入库，只记条数与
  E1 评测总体的 sha256（configs/e1_eval_subsets.json 的 populations.countbench），运行时核对。
- devtune_countbench：E1 的 CountBench 开发子集（configs/e1_dev_subsets.json 的 countbench，49 条）——计数调提示用（DEV-0051 的提议，待裁）。
去重照 E1：样本一律经 E1 的加载器（subset='eval' / 'dev'），已剔除开发子集与 D20 探针图；计数的问句照 E1 默认（countbenchqa，D36-2）。

w1_items(task, split, paths) 按清单取样本（加载器的固定顺序），清单里的 id 在总体里找不到就报错，取出的 id 指纹与清单不符也报错。
生成与核对清单：python -m src.w1.data --write | --check（只读入库的 E1 清单，不要数据）。
--tiny（tests/fixtures 的小数据）：tiny_splits() 用同一套规则在 fixture 的 E1 清单上现算（大小取 configs/w1.json 的 splits.tiny）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import loaders as L
from src.data.eval_subsets import EVAL_SUBSETS_PATH, POLY_SPLITS, load_eval_subsets
from src.data.records import Sample
from src.data.subsets import DEV_SUBSETS_PATH, dump_config, hash_rank, load_dev_subsets, population_digest, sort_ids
from src.e1.tasks import FIXTURE_DIR, DataPaths, _pick, _sent_id, select
from src.w1.config import W1Config, load_w1_config

REPO_ROOT = Path(__file__).resolve().parents[2]
W1_SPLITS_PATH = REPO_ROOT / "configs" / "w1_splits.json"
W1_SPLITS_VERSION = 1
REFCOCO_TASKS: tuple[str, ...] = ("refseg", "refexp_box")
COUNT_TASKS: tuple[str, ...] = ("count",)
_POLY_KEYS = {f"{ds}_{sb}_{sp}": (ds, sb, sp) for ds, sb, sp in POLY_SPLITS}


def poly_test_split(key: str) -> str:
    return f"test_{key}"


def _rel(p: str | Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def build_splits(cfg: W1Config, eval_subsets: Mapping[str, Any], dev_subsets: Mapping[str, Any], *, tiny: bool = False,
                 sources: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """由 E1 的评测子集与开发子集清单生成 W1 的划分（规则见模块说明）。返回可直接 dump_config 的 dict（键序固定）。"""
    sp = cfg.splits
    sizes = sp["tiny"] if tiny else sp
    n_tune, n_check = int(sizes["devtune_n"]), int(sizes["devcheck_n"])
    pop_name = sp["population"]
    sub = eval_subsets["subsets"][pop_name]
    pop = list(sub["ids"])
    if len(pop) < n_tune + n_check:
        raise ValueError(f"{pop_name} 只有 {len(pop)} 句，不够 devtune {n_tune} + devcheck {n_check}")
    ranked = hash_rank(pop, sp["purpose"], sp["salt"])
    ds, sb, split = _POLY_KEYS[pop_name.split(":", 1)[1]]
    splits: dict[str, dict[str, Any]] = {}

    def refcoco_entry(ids: Sequence[str], dsx: str, sbx: str, splitx: str, rule: str) -> dict[str, Any]:
        ids = sort_ids(ids)
        return {"kind": "refcoco", "tasks": list(REFCOCO_TASKS), "dataset": dsx, "split_by": sbx, "split": splitx,
                "unit": "sent_id", "rule": rule, "n": len(ids), "sha256": population_digest(ids), "ids": ids}

    splits["devtune"] = refcoco_entry(ranked[:n_tune], ds, sb, split,
                                      f"{pop_name} 按 sha256(salt|purpose|sent_id) 排序的第 1–{n_tune} 句")
    splits["devcheck"] = refcoco_entry(ranked[n_tune:n_tune + n_check], ds, sb, split,
                                       f"{pop_name} 按 sha256(salt|purpose|sent_id) 排序的第 {n_tune + 1}–{n_tune + n_check} 句")
    for key in sp["test"]:
        if key not in _POLY_KEYS:
            raise ValueError(f"splits.test 里的 {key!r} 不是 E1 的 poly 子集")
        dsx, sbx, splitx = _POLY_KEYS[key]
        splits[poly_test_split(key)] = refcoco_entry(eval_subsets["subsets"][f"poly:{key}"]["ids"], dsx, sbx, splitx,
                                                     f"E1 的 poly:{key} 子集原样")
    cd = dev_subsets["subsets"]["countbench"]
    ids = sort_ids(cd["ids"])
    splits["devtune_countbench"] = {"kind": "countbench_dev", "tasks": list(COUNT_TASKS), "unit": "row",
                                    "rule": "E1 的 CountBench 开发子集（e1_dev_subsets.json 的 countbench；DEV-0051 的提议，待裁）",
                                    "n": len(ids), "sha256": population_digest(ids), "ids": ids}
    cp = eval_subsets["populations"]["countbench"]
    splits["test_countbench"] = {"kind": "countbench_eval", "tasks": list(COUNT_TASKS), "unit": "row",
                                 "rule": "E1 的 CountBench 评测条目（load_countbench(subset='eval') 全部）；id 不入库，运行时按 E1 评测总体的 "
                                         "sha256 核对", "n": int(cp["n"]), "sha256": cp["sha256"], "ids": None}
    return {"version": W1_SPLITS_VERSION, "salt": sp["salt"], "purpose": sp["purpose"], "population": pop_name,
            "method": "按 sha256(salt|purpose|id) 升序排成一列（src/data/subsets.hash_rank），按名次切成 devtune 与 devcheck，各自按 id 自然顺序"
                      "列出；sha256 = 自然顺序、换行连接后的 sha256（src/data/subsets.population_digest）",
            "tiny": tiny, "sources": dict(sources or {}), "splits": splits}


def splits_sha256(doc: Mapping[str, Any]) -> str:
    """划分清单的指纹 = 规范文本（src/data/subsets.dump_config）的 sha256；入库文件就是这份文本，所以等于文件的 sha256。"""
    return hashlib.sha256(dump_config(doc).encode("utf-8")).hexdigest()


def _sources(eval_path: Path, dev_path: Path) -> dict[str, Any]:
    return {"e1_eval_subsets": {"path": _rel(eval_path), "sha256": hashlib.sha256(Path(eval_path).read_bytes()).hexdigest()},
            "e1_dev_subsets": {"path": _rel(dev_path), "sha256": hashlib.sha256(Path(dev_path).read_bytes()).hexdigest()}}


def build_from_files(cfg: W1Config | None = None, *, eval_path: Path = EVAL_SUBSETS_PATH, dev_path: Path = DEV_SUBSETS_PATH,
                     tiny: bool = False) -> dict[str, Any]:
    cfg = cfg or load_w1_config()
    return build_splits(cfg, load_eval_subsets(eval_path), load_dev_subsets(dev_path), tiny=tiny,
                        sources=_sources(eval_path, dev_path))


def tiny_splits(cfg: W1Config | None = None) -> dict[str, Any]:
    """--tiny：fixture 的 E1 清单（tests/fixtures/e1）上按同一套规则现算（大小取 splits.tiny）。"""
    return build_from_files(cfg, eval_path=FIXTURE_DIR / "eval_subsets.json", dev_path=FIXTURE_DIR / "dev_subsets.json", tiny=True)


def load_splits(path: str | Path = W1_SPLITS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_names(task: str, doc: Mapping[str, Any]) -> list[str]:
    return [k for k, v in doc["splits"].items() if task in v["tasks"]]


def w1_items(task: str, split: str, paths: DataPaths, *, splits: Mapping[str, Any] | None = None) -> list[Sample]:
    """task × 划分 → 样本（加载器的固定顺序）。splits 默认读 configs/w1_splits.json。规则见模块说明。"""
    doc = splits if splits is not None else load_splits()
    if split not in doc["splits"]:
        raise KeyError(f"没有划分 {split!r}；可用：{list(doc['splits'])}")
    e = doc["splits"][split]
    if task not in e["tasks"]:
        raise ValueError(f"划分 {split} 只用于 {e['tasks']}，不用于 {task}")
    if e["kind"] == "refcoco":
        res = L.load_refcoco(paths.root, e["dataset"], e["split"], subset="eval", dev_config=paths.dev_subsets, strict=paths.strict)
        samples = _pick(res.records, list(e["ids"]), _sent_id, f"w1:{split}")
        got = [_sent_id(s) for s in samples]
    elif e["kind"] == "countbench_dev":
        samples = select("count", "dev", paths).samples
        got = [str(s.fields["row"]) for s in samples]
        if sort_ids(got) != list(e["ids"]):
            raise ValueError(f"w1:{split}：E1 的 CountBench 开发子集与清单不同")
    elif e["kind"] == "countbench_eval":
        samples = select("count", "eval", paths).samples
        got = [str(s.fields["row"]) for s in samples]
    else:
        raise ValueError(f"未知的划分种类 {e['kind']!r}")
    if len(got) != e["n"] or population_digest(got) != e["sha256"]:
        raise ValueError(f"w1:{split}：取出的 {len(got)} 条 id 指纹 {population_digest(got)[:12]}… 与清单的 {e['n']} 条 "
                         f"{e['sha256'][:12]}… 不符")
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成或核对 configs/w1_splits.json（只读入库的 E1 清单，不要数据）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = dump_config(build_from_files())
    if a.write:
        W1_SPLITS_PATH.write_text(text, encoding="utf-8")
        print(f"写出 {W1_SPLITS_PATH}（sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}）")
        return 0
    same = W1_SPLITS_PATH.exists() and W1_SPLITS_PATH.read_text(encoding="utf-8") == text
    print("一致" if same else f"{W1_SPLITS_PATH} 与重算的结果不同")
    return 0 if same else 1


__all__ = [
    "COUNT_TASKS",
    "REFCOCO_TASKS",
    "W1_SPLITS_PATH",
    "build_from_files",
    "build_splits",
    "load_splits",
    "split_names",
    "splits_sha256",
    "poly_test_split",
    "tiny_splits",
    "w1_items",
]


if __name__ == "__main__":
    sys.exit(main())
