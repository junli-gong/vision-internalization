"""固定种子的抽样与开发子集（D11-6、DEV-0012 任务书 §2）。

抽样一律用哈希排序（src/data/constants.SUBSET_SALT）：与 Python/numpy 版本、平台无关，逐字节可复现。
开发子集的 id 清单入库在 configs/e1_dev_subsets.json，是协议的一部分；scripts/data/build_e1_configs.py 负责生成与核对。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.constants import (
    DEV_SUBSET_FRAC_DEN,
    DEV_SUBSET_FRAC_NUM,
    DEV_SUBSET_MAX,
    NYU_DEV_IMAGES,
    REFCOCO_DEV_SENTENCES,
    SUBSET_SALT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_SUBSETS_PATH = REPO_ROOT / "configs" / "e1_dev_subsets.json"
DEV_SUBSETS_VERSION = 1


def hash_key(purpose: str, item_id: str, salt: str = SUBSET_SALT) -> str:
    return hashlib.sha256(f"{salt}|{purpose}|{item_id}".encode("utf-8")).hexdigest()


def hash_rank(ids: Iterable[str], purpose: str, salt: str = SUBSET_SALT) -> list[str]:
    """按 sha256(salt|purpose|id) 升序排列 ids（哈希相同按 id 排，实际不会发生）。id 必须唯一。"""
    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{purpose}: id 有重复")
    return sorted(ids, key=lambda i: (hash_key(purpose, i, salt), i))


def seeded_sample(ids: Iterable[str], k: int, purpose: str, salt: str = SUBSET_SALT) -> list[str]:
    """哈希排序后取前 k 个（k ≥ 总数时全取），按 id 的自然顺序返回。"""
    ranked = hash_rank(ids, purpose, salt)
    return sort_ids(ranked[: max(k, 0)])


def sort_ids(ids: Iterable[str]) -> list[str]:
    """id 的自然顺序：能整体按整数比较时按整数，否则按 (前缀, 末尾数字, 原串)。"""

    def key(s: str):
        if s.lstrip("-").isdigit():
            return (0, "", int(s), s)
        head = s.rstrip("0123456789")
        tail = s[len(head) :]
        return (1, head, int(tail) if tail else -1, s)

    return sorted((str(i) for i in ids), key=key)


def dev_subset_size(n: int) -> int:
    """只有测试划分的基准：min(50, floor(10% n))。"""
    return min(DEV_SUBSET_MAX, (n * DEV_SUBSET_FRAC_NUM) // DEV_SUBSET_FRAC_DEN)


def population_digest(ids: Iterable[str]) -> str:
    """总体 id 集合的指纹：自然顺序、换行连接后的 sha256。"""
    return hashlib.sha256("\n".join(sort_ids(ids)).encode("utf-8")).hexdigest()


# 每个开发子集的规则：(单位, 抽取个数的函数, 说明)。键就是 configs/e1_dev_subsets.json 里 subsets 的键。
DEV_SUBSET_RULES: dict[str, dict[str, Any]] = {
    "cvbench": {"unit": "question(idx)", "rule": "min(50, floor(10% n))，n = test 全部 2,638 题"},
    "embspatial": {"unit": "question(question_id)", "rule": "min(50, floor(10% n))，n = test 全部 3,640 题"},
    "blink_reldepth": {"unit": "question(idx)", "rule": "min(50, floor(10% n))，n = Relative_Depth val 124 题"},
    "countbench": {"unit": "row(官方 CountBench.json 的 0 起行号)", "rule": "min(50, floor(10% n))，n = 有图的 491 行"},
    "refcoco_unc_val": {"unit": "sentence(sent_id)", "rule": f"val 的 {REFCOCO_DEV_SENTENCES} 句"},
    "refcoco+_unc_val": {"unit": "sentence(sent_id)", "rule": f"val 的 {REFCOCO_DEV_SENTENCES} 句"},
    "refcocog_umd_val": {"unit": "sentence(sent_id)", "rule": f"val 的 {REFCOCO_DEV_SENTENCES} 句"},
    "nyuv2_train": {"unit": "image(labeled 集 1-based 下标)", "rule": f"官方 train（splits.mat trainNdxs，795 张）里的 {NYU_DEV_IMAGES} 张"},
}


def dev_subset_k(name: str, n: int) -> int:
    if name.startswith("refcoco"):
        return min(REFCOCO_DEV_SENTENCES, n)
    if name == "nyuv2_train":
        return min(NYU_DEV_IMAGES, n)
    return dev_subset_size(n)


def build_dev_subsets(populations: Mapping[str, Sequence[str]], provenance: Mapping[str, str] | None = None) -> dict[str, Any]:
    """由各基准的总体 id 清单生成开发子集配置（可 JSON 化的 dict，键序固定）。"""
    unknown = set(populations) - set(DEV_SUBSET_RULES)
    if unknown:
        raise ValueError(f"未知的开发子集：{sorted(unknown)}")
    subsets = {}
    for name in sorted(populations):
        pop = [str(i) for i in populations[name]]
        k = dev_subset_k(name, len(pop))
        subsets[name] = {
            **DEV_SUBSET_RULES[name],
            "n_population": len(pop),
            "population_sha256": population_digest(pop),
            "population_provenance": (provenance or {}).get(name, ""),
            "k": k,
            "n_remaining": len(pop) - k,
            "ids": seeded_sample(pop, k, purpose=f"dev:{name}"),
        }
    return {
        "version": DEV_SUBSETS_VERSION,
        "salt": SUBSET_SALT,
        "method": "按 sha256(salt|dev:<子集名>|id) 升序取前 k 个，再按 id 自然顺序列出（src/data/subsets.py）",
        "usage": "只用于解析器、提示格式开发与批处理一致性验收（D11-6、D28-5）；从报告的数字里剔除，报告写明剩余的 n",
        "subsets": subsets,
    }


def dump_config(cfg: Mapping[str, Any]) -> str:
    """配置文件的规范文本：缩进 1、保留中文、末尾换行（逐字节复现靠它）。"""
    return json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"


def load_dev_subsets(path: str | Path = DEV_SUBSETS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def dev_ids(name: str, path: str | Path = DEV_SUBSETS_PATH) -> set[str]:
    """某个开发子集的 id 集合；配置里没有这个子集时报错（不静默当成空集）。"""
    cfg = load_dev_subsets(path)
    if name not in cfg["subsets"]:
        raise KeyError(f"{path} 里没有开发子集 {name}")
    return set(cfg["subsets"][name]["ids"])
