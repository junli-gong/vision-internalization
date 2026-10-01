"""开发子集与抽样（src/data/subsets.py、configs/e1_dev_subsets.json；D11-6、DEV-0012 任务书 §2）。

固定种子下逐字节复现：哈希排序与配置文本的 sha256 在这里钉死，任何机器上都必须相同。
入库配置里 CV-Bench、BLINK、EmbSpatial 的总体可以由已知的 id 规律重建（总体指纹与配置一致），据此重算并核对 ids；
其余子集（CountBench、RefCOCO、NYUv2）的总体要落地数据才有，服务器上用 build_e1_configs.py dev-subsets --check 核对。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.data import constants as C
from src.data.subsets import (
    DEV_SUBSET_RULES,
    DEV_SUBSETS_PATH,
    build_dev_subsets,
    dev_subset_k,
    dev_subset_size,
    dump_config,
    hash_rank,
    population_digest,
    seeded_sample,
    sort_ids,
)

REPO = Path(__file__).resolve().parents[1]
FX = Path(__file__).resolve().parent / "fixtures" / "e1"

KNOWN_POPULATIONS = {
    "cvbench": [str(i) for i in range(2638)],
    "blink_reldepth": [f"val_Relative_Depth_{i}" for i in range(1, 125)],
    "embspatial": [f"mp3d_{i}" for i in range(1201)] + [f"ai2thor_{i}" for i in range(1239)] + [f"scannet_{i}" for i in range(1200)],
}


def test_hash_rank_is_pinned():
    assert C.SUBSET_SALT == "visint-e1-data-v1"
    assert hash_rank(["a", "b", "c", "d", "e"], "t") == ["b", "a", "d", "c", "e"]
    assert hash_rank(["e", "d", "c", "b", "a"], "t") == ["b", "a", "d", "c", "e"]  # 与输入顺序无关
    with pytest.raises(ValueError, match="重复"):
        hash_rank(["a", "a"], "t")


def test_seeded_sample_nested_and_natural_order():
    ids = [str(i) for i in range(300)]
    s10 = seeded_sample(ids, 10, "p")
    s20 = seeded_sample(ids, 20, "p")
    assert set(s10) <= set(s20) and s10 == sort_ids(s10)
    assert seeded_sample(ids, 10, "q") != s10  # 用途不同，样本不同
    assert seeded_sample(ids[:5], 50, "p") == ids[:5]


def test_dev_subset_size_rule():
    assert (C.DEV_SUBSET_MAX, C.DEV_SUBSET_FRAC_NUM, C.DEV_SUBSET_FRAC_DEN) == (50, 1, 10)
    assert [dev_subset_size(n) for n in (124, 491, 2638, 3640, 499, 500, 9)] == [12, 49, 50, 50, 49, 50, 0]
    assert dev_subset_k("refcoco_unc_val", 10834) == 200 and dev_subset_k("nyuv2_train", 795) == 50


def test_sort_ids_natural():
    assert sort_ids(["10", "9", "2"]) == ["2", "9", "10"]
    assert sort_ids(["val_Relative_Depth_10", "val_Relative_Depth_2"]) == ["val_Relative_Depth_2", "val_Relative_Depth_10"]
    assert sort_ids(["scannet_1", "ai2thor_3", "mp3d_0"]) == ["ai2thor_3", "mp3d_0", "scannet_1"]


def test_build_dev_subsets_bytewise_reproducible():
    pops = {k: KNOWN_POPULATIONS[k] for k in ("cvbench", "blink_reldepth")}
    a = dump_config(build_dev_subsets(pops))
    b = dump_config(build_dev_subsets({k: list(reversed(v)) for k, v in pops.items()}))
    assert a == b
    assert hashlib.sha256(a.encode()).hexdigest() == "9d6e139d5df0ae3039bc03cfc843b56007512e3c198189b9b0bab83430a3ea88"
    with pytest.raises(ValueError, match="未知"):
        build_dev_subsets({"nope": ["1"]})


def test_committed_dev_subsets_config():
    text = DEV_SUBSETS_PATH.read_text()
    cfg = json.loads(text)
    assert text == dump_config(cfg)  # 规范格式
    assert cfg["salt"] == C.SUBSET_SALT and cfg["version"] == 1
    assert set(cfg["subsets"]) == set(DEV_SUBSET_RULES)
    for name, sub in cfg["subsets"].items():
        assert sub["k"] == dev_subset_k(name, sub["n_population"]) == len(sub["ids"]), name
        assert sub["n_remaining"] == sub["n_population"] - sub["k"]
        assert sub["ids"] == sort_ids(sub["ids"]) and len(set(sub["ids"])) == len(sub["ids"])
        assert sub["population_provenance"]
    expect_n = {"cvbench": 2638, "embspatial": 3640, "blink_reldepth": 124, "countbench": 491, "refcoco_unc_val": 10834, "refcoco+_unc_val": 10758, "refcocog_umd_val": 4896, "nyuv2_train": 795}
    assert {k: v["n_population"] for k, v in cfg["subsets"].items()} == expect_n
    assert all(1 <= int(i) <= 1449 for i in cfg["subsets"]["nyuv2_train"]["ids"])
    assert all(0 <= int(i) < 540 for i in cfg["subsets"]["countbench"]["ids"])


@pytest.mark.parametrize("name", sorted(KNOWN_POPULATIONS))
def test_committed_ids_recompute_from_known_population(name):
    sub = json.loads(DEV_SUBSETS_PATH.read_text())["subsets"][name]
    pop = KNOWN_POPULATIONS[name]
    assert population_digest(pop) == sub["population_sha256"]
    assert seeded_sample(pop, sub["k"], f"dev:{name}") == sub["ids"]


def test_build_script_dev_subsets_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location("_build_e1_configs", REPO / "scripts/data/build_e1_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "dev.json"
    names = ["cvbench", "embspatial", "blink_reldepth", "countbench", "refcoco_unc_val"]
    assert mod.main(["dev-subsets", "--root", str(FX / "root"), "--out", str(out), "--only", *names]) == 0
    assert mod.main(["dev-subsets", "--root", str(FX / "root"), "--out", str(out), "--only", *names, "--check"]) == 0
    cfg = json.loads(out.read_text())
    cfg["subsets"]["cvbench"]["ids"] = ["999"]
    out.write_text(json.dumps(cfg))
    assert mod.main(["dev-subsets", "--root", str(FX / "root"), "--out", str(out), "--only", *names, "--check"]) == 1
